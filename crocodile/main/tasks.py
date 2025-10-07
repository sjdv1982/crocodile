from __future__ import annotations

from typing import Dict, Iterable, List, Sequence, Tuple

import numpy as np
from scipy.spatial.transform import Rotation


class Task:
    """Represents a single workload unit for rotaconformer processing."""

    def __init__(
        self,
        *,
        nr: int,
        prototype: int,
        target_conformers: np.ndarray,
        source_conformers: np.ndarray,
        rc_source: int,
        rc_target: int,
        context: Dict[str, object],
    ) -> None:
        self.nr = nr
        self.prototype = prototype
        self.target_conformers = np.asarray(target_conformers, dtype=int)
        self.source_conformers = np.asarray(source_conformers, dtype=int)
        self.rc_source = rc_source
        self.rc_target = rc_target
        self._context = context

    def prepare_task(self, semaphore) -> Tuple["Task", np.ndarray, np.ndarray, np.ndarray]:
        """Prepare membership and RMSD data for this task."""

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
        task_dicts: List[Dict[str, object]] = []

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

            def new_task() -> Dict[str, object]:
                curr_task: Dict[str, object] = {
                    "nr": len(task_dicts),
                    "prototype": proto,
                    "target_conformers": [],
                    "source_conformers": set(),
                    "rc_source": 0,
                    "rc_target": 0,
                }
                task_dicts.append(curr_task)
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
                curr_task["rc_target"] = len(rc)
                curr_task["target_conformers"].append(int(target_conf))
                for source_conf in conf_pairs[target_conf]:
                    curr_task["source_conformers"].add(int(source_conf))
                    rc = origin_rotaconformers[source_conf]
                    curr_task["rc_source"] = int(curr_task["rc_source"]) + len(rc)
                while curr_task["rc_target"] * curr_task["rc_source"] < 100e6:
                    best_target_conf = None
                    lowest_waste = None
                    for target_conf in target_conf_list:
                        if target_conf in target_conf_done:
                            continue
                        waste = 0
                        rc_target_conf = len(lib.get_rotamers(target_conf))
                        for source_conf in conf_pairs[target_conf]:
                            if source_conf not in curr_task["source_conformers"]:
                                rc = origin_rotaconformers[source_conf]
                                assert len(rc) and curr_task["rc_target"]
                                waste += len(rc) * curr_task["rc_target"]
                            else:
                                assert crmsd_ok[source_conf, target_conf]
                        for source_conf in curr_task["source_conformers"]:
                            if source_conf not in conf_pairs[target_conf]:
                                rc = origin_rotaconformers[source_conf]
                                assert len(rc) and rc_target_conf
                                waste += len(rc) * rc_target_conf
                            else:
                                assert crmsd_ok[source_conf, target_conf]

                        if lowest_waste is None or waste < lowest_waste:
                            best_target_conf = target_conf
                            lowest_waste = waste

                    if best_target_conf is None:
                        break

                    target_conf_done.add(best_target_conf)
                    rc = lib.get_rotamers(best_target_conf)
                    curr_task["rc_target"] += len(rc)
                    curr_task["target_conformers"].append(int(best_target_conf))
                    for source_conf in conf_pairs[best_target_conf]:
                        assert crmsd_ok[source_conf, best_target_conf]
                        source_conf = int(source_conf)
                        if source_conf not in curr_task["source_conformers"]:
                            curr_task["source_conformers"].add(source_conf)
                            rc = origin_rotaconformers[source_conf]
                            curr_task["rc_source"] += len(rc)

        for curr_task in task_dicts:
            target = np.array(curr_task["target_conformers"], dtype=int)
            curr_task["target_conformers"] = np.sort(target)
            source = np.array(list(curr_task["source_conformers"]), dtype=int)
            curr_task["source_conformers"] = np.sort(source)

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

        self._proto_align = proto_align
        self._tasks = [
            Task(
                nr=curr_task["nr"],
                prototype=curr_task["prototype"],
                target_conformers=curr_task["target_conformers"],
                source_conformers=curr_task["source_conformers"],
                rc_source=curr_task["rc_source"],
                rc_target=curr_task["rc_target"],
                context=context,
            )
            for curr_task in task_dicts
        ]

    @property
    def proto_align(self) -> Dict[Tuple[int, int], Rotation]:
        return self._proto_align

    def __len__(self) -> int:
        return len(self._tasks)

    def __getitem__(self, index: int) -> Task:
        return self._tasks[index]

    def __iter__(self):
        return iter(self._tasks)
