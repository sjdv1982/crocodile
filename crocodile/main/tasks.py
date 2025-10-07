from __future__ import annotations

from typing import Dict, Iterable, Iterator, List, Optional, Sequence, Set, Tuple

import numpy as np
from scipy.spatial.transform import Rotation


class Task:
    """Represents a single workload unit for rotaconformer processing."""

    def __init__(
        self,
        *,
        nr: int,
        prototype: int,
        target_conformers: Iterable[int] | None = None,
        source_conformers: Iterable[int] | None = None,
        rc_source: int = 0,
        rc_target: int = 0,
        context: Optional[Dict[str, object]] = None,
    ) -> None:
        self.nr = nr
        self.prototype = prototype
        self.rc_source = int(rc_source)
        self.rc_target = int(rc_target)
        self._context = context

        self._target_conformers: List[int] = [int(c) for c in (target_conformers or [])]
        self._source_conformers: Set[int] = {int(c) for c in (source_conformers or [])}

        self.target_conformers: Optional[np.ndarray] = None
        self.source_conformers: Optional[np.ndarray] = None
        self._finalized = False

    def add_target_conformer(self, conformer: int, rc_count: int) -> None:
        if self._finalized:
            self._raise_finalized()
        self._target_conformers.append(int(conformer))
        self.rc_target += int(rc_count)

    def add_source_conformer(self, conformer: int, rc_count: int) -> None:
        if self._finalized:
            self._raise_finalized()
        conformer = int(conformer)
        if conformer not in self._source_conformers:
            self._source_conformers.add(conformer)
            self.rc_source += int(rc_count)

    def has_source_conformer(self, conformer: int) -> bool:
        return int(conformer) in self._source_conformers

    def iter_source_conformers(self) -> Iterator[int]:
        if self._finalized and self.source_conformers is not None:
            return iter(self.source_conformers.tolist())
        return iter(self._source_conformers)

    def finalize(self) -> None:
        if self._finalized:
            return
        self._target_conformers = sorted(self._target_conformers)
        source_list = sorted(self._source_conformers)
        self.target_conformers = np.array(self._target_conformers, dtype=int)
        self.source_conformers = np.array(source_list, dtype=int)
        self._finalized = True

    def set_context(self, context: Dict[str, object]) -> None:
        self._context = context

    def _raise_finalized(self) -> None:
        raise RuntimeError("Cannot modify task after finalization")

    def prepare_task(self, semaphore) -> Tuple["Task", np.ndarray, np.ndarray, np.ndarray]:
        if not self._finalized:
            raise RuntimeError("Task must be finalized before preparation")
        if self._context is None:
            raise RuntimeError("Task context not initialised")

        semaphore.acquire()

        ctx = self._context
        proto = self.prototype
        csource_conformers = self.source_conformers
        ctarget_conformers = self.target_conformers

        prototype_clusters = ctx["prototype_clusters"]
        prototypes_scalevec = ctx["prototypes_scalevec"]
        origin_rotaconformers = ctx["origin_rotaconformers"]
        proto_align = ctx["proto_align"]
        conf_prototypes = ctx["conf_prototypes"]
        load_membership = ctx["load_membership"]
        motif = ctx["motif"]
        nucpos = ctx["nucpos"]
        ovRMSD = ctx["ovRMSD"]
        membership_bins = ctx["membership_bins"]

        clusters = prototype_clusters[proto]
        scalevec = prototypes_scalevec[proto]

        csource_rotaconformers_align = Rotation.concatenate(
            [
                origin_rotaconformers[conf] * proto_align[proto, conf]
                for conf in csource_conformers
            ]
        )

        csource_rotaconformers = Rotation.concatenate(
            [origin_rotaconformers[conf] for conf in csource_conformers]
        )

        rmsd = np.empty((len(clusters), len(csource_rotaconformers)))

        for n, cluster in enumerate(clusters):
            rr = csource_rotaconformers_align * cluster.inv()
            ax = rr.as_rotvec()
            ang = np.linalg.norm(ax, axis=1)
            ang = np.maximum(ang, 0.0001)
            ax /= ang[:, None]
            fac = (np.cos(ang) - 1) ** 2 + np.sin(ang) ** 2
            cross = (scalevec * scalevec) * (1 - ax * ax)
            curr_rmsd = np.sqrt(fac * cross.sum(axis=-1))
            rmsd[n, :] = curr_rmsd

        rmsd_upper = np.digitize(rmsd + ovRMSD, membership_bins).astype(np.uint8)
        rmsd_lower = np.digitize(rmsd - ovRMSD, membership_bins).astype(np.uint8)

        membership: List[np.ndarray] = []
        for conformer in ctarget_conformers:
            assert conf_prototypes[conformer] == proto, (
                conformer,
                conf_prototypes[conformer],
                proto,
            )
            nclust = len(prototype_clusters[proto])
            curr_membership = load_membership(motif, nucpos, conformer, nclust)
            membership.append(curr_membership)

        membership = np.concatenate(membership, axis=1)
        assert membership.shape[1] == self.rc_target, (
            membership.shape[1],
            self.rc_target,
        )
        assert rmsd.shape[1] == self.rc_source, (rmsd.shape[1], self.rc_source)

        return self, membership, rmsd_upper, rmsd_lower


class TaskList:
    """Container around a collection of tasks with shared context."""

    MEMBERSHIP_BINS = np.array(
        [1.0, 1.25, 1.5, 1.75, 2.0, 2.25, 2.5, 2.75, 3.0, 3.25, 3.5, 3.75, 4, 4.5, 5]
    )

    def __init__(
        self,
        all_proto: Sequence[int],
        *,
        source_confs: Dict[int, Iterable[int]],
        conf_prototypes: Sequence[int],
        prototypes: np.ndarray,
        prev_lib,
        origin_rotaconformers: Dict[int, Rotation],
        lib,
        crmsd_ok: np.ndarray,
        superimpose,
        prototype_clusters: Dict[int, Rotation],
        prototypes_scalevec: np.ndarray,
        load_membership,
        motif: str,
        nucpos: int,
        ovRMSD: float,
    ) -> None:
        proto_align: Dict[Tuple[int, int], Rotation] = {}
        tasks: List[Task] = []

        for proto in all_proto:
            conf_pairs = {
                tconf: oriconfs
                for tconf, oriconfs in source_confs.items()
                if conf_prototypes[tconf] == proto
            }
            all_oriconfs = set()
            for tconf in conf_pairs:
                all_oriconfs.update(conf_pairs[tconf])

            prototype = prototypes[proto]
            for oriconf in all_oriconfs:
                mat = superimpose(prev_lib.coordinates[oriconf], prototype)[0]
                proto_align[proto, oriconf] = Rotation.from_matrix(mat)

            def new_task() -> Task:
                curr_task = Task(nr=len(tasks), prototype=proto)
                tasks.append(curr_task)
                return curr_task

            target_conf_list = sorted(
                conf_pairs.keys(), key=lambda k: -len(conf_pairs[k])
            )
            target_conf_done = set()
            for target_conf in target_conf_list:
                if target_conf in target_conf_done:
                    continue
                target_conf_done.add(target_conf)
                curr_task = new_task()
                rc = lib.get_rotamers(target_conf)
                curr_task.add_target_conformer(target_conf, len(rc))
                for source_conf in conf_pairs[target_conf]:
                    rc = origin_rotaconformers[source_conf]
                    curr_task.add_source_conformer(source_conf, len(rc))
                while curr_task.rc_target * curr_task.rc_source < 100e6:
                    best_target_conf = None
                    lowest_waste = None
                    for candidate_conf in target_conf_list:
                        if candidate_conf in target_conf_done:
                            continue
                        waste = 0
                        rc_target_conf = len(lib.get_rotamers(candidate_conf))
                        for source_conf in conf_pairs[candidate_conf]:
                            if not curr_task.has_source_conformer(source_conf):
                                rc = origin_rotaconformers[source_conf]
                                assert len(rc) and curr_task.rc_target
                                waste += len(rc) * curr_task.rc_target
                            else:
                                assert crmsd_ok[source_conf, candidate_conf]
                        for source_conf in curr_task.iter_source_conformers():
                            if source_conf not in conf_pairs[candidate_conf]:
                                rc = origin_rotaconformers[source_conf]
                                assert len(rc) and rc_target_conf
                                waste += len(rc) * rc_target_conf
                            else:
                                assert crmsd_ok[source_conf, candidate_conf]

                        if lowest_waste is None or waste < lowest_waste:
                            best_target_conf = candidate_conf
                            lowest_waste = waste

                    if best_target_conf is None:
                        break

                    target_conf_done.add(best_target_conf)
                    rc = lib.get_rotamers(best_target_conf)
                    curr_task.add_target_conformer(best_target_conf, len(rc))
                    for source_conf in conf_pairs[best_target_conf]:
                        assert crmsd_ok[source_conf, best_target_conf]
                        if not curr_task.has_source_conformer(source_conf):
                            rc = origin_rotaconformers[source_conf]
                            curr_task.add_source_conformer(source_conf, len(rc))

        context = {
            "origin_rotaconformers": origin_rotaconformers,
            "proto_align": proto_align,
            "prototype_clusters": prototype_clusters,
            "prototypes_scalevec": prototypes_scalevec,
            "conf_prototypes": conf_prototypes,
            "load_membership": load_membership,
            "motif": motif,
            "nucpos": nucpos,
            "ovRMSD": ovRMSD,
            "membership_bins": self.MEMBERSHIP_BINS,
        }

        for task in tasks:
            task.finalize()
            task.set_context(context)

        self._proto_align = proto_align
        self._tasks = tasks

    @property
    def proto_align(self) -> Dict[Tuple[int, int], Rotation]:
        return self._proto_align

    def __len__(self) -> int:
        return len(self._tasks)

    def __getitem__(self, index: int) -> Task:
        return self._tasks[index]

    def __iter__(self):
        return iter(self._tasks)
