from __future__ import annotations

from typing import Dict, Iterable, Iterator, List, Optional, Sequence, Set, Tuple

import numpy as np
from scipy.spatial.transform import Rotation

from crocodile.main.superimpose import superimpose_array, superimpose


class Task:
    """Represents a single workload unit for rotaconformer processing."""

    def __init__(
        self,
        *,
        nr: int,
        prototype: int,
        context: Optional[Dict[str, object]],
        target_conformers: Iterable[int] | None = None,
        source_conformers: Iterable[int] | None = None,
        rc_source: int = 0,
        rc_target: int = 0,
    ) -> None:
        self.nr = nr
        self.prototype = prototype
        self.rc_source = int(rc_source)
        self.rc_target = int(rc_target)
        self._context = context

        if isinstance(target_conformers, np.ndarray):
            target_conformers = target_conformers.tolist()
        if isinstance(source_conformers, np.ndarray):
            source_conformers = source_conformers.tolist()

        self._target_conformers: List[int] = [int(c) for c in (target_conformers or [])]
        self._source_conformers: Set[int] = {int(c) for c in (source_conformers or [])}

        self.target_conformers: Optional[np.ndarray] = None
        self.source_conformers: Optional[np.ndarray] = None
        self.membership: Optional[np.ndarray] = None
        self.rmsd_upper: Optional[np.ndarray] = None
        self.rmsd_lower: Optional[np.ndarray] = None
        self.source_rotaconf_counts: Optional[np.ndarray] = None
        self.candidates: Optional[np.ndarray] = None
        self._prepared = False
        self._has_run = False
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

    def prepare_task(self, semaphore):
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

        origin_rotaconformers2 = {
            conf: origin_rotaconformers[conf].as_matrix()
            for conf in origin_rotaconformers
        }
        csource_rotaconformers_align = np.concatenate(
            [
                proto_align[proto, conf]
                .dot(origin_rotaconformers2[conf])
                .swapaxes(0, 1)
                for conf in csource_conformers
            ]
        )

        csource_rotaconformers = Rotation.concatenate(
            [origin_rotaconformers[conf] for conf in csource_conformers]
        )

        rmsd = np.empty((len(clusters), len(csource_rotaconformers)))

        for n, cluster in enumerate(clusters):
            rr = csource_rotaconformers_align.dot(cluster.inv().as_matrix())
            ax = Rotation.from_matrix(rr).as_rotvec()

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

        self.membership = membership
        self.rmsd_upper = rmsd_upper
        self.rmsd_lower = rmsd_lower
        self._prepared = True

    def run_task(self) -> None:
        from .julia_import import Main

        if not self._prepared:
            raise RuntimeError("Task must be prepared before processing")
        if self._has_run:
            return
        assert self.membership is not None
        assert self.rmsd_upper is not None
        assert self.rmsd_lower is not None

        source_rotaconf_counts, candidates = Main.CrocoCandidates.compute_candidates(
            self.membership,
            self.rmsd_upper,
            self.rmsd_lower,
        )
        self.membership = None
        self.rmsd_upper = None
        self.rmsd_lower = None
        self.source_rotaconf_counts = source_rotaconf_counts.to_numpy()
        self.candidates = candidates.to_numpy() - 1
        self._has_run = True

    def _run_task_tensor(self) -> None:
        lib, prev_lib = self._context["lib"], self._context["prev_lib"]
        prototype = self._context["prototypes"][self.prototype]
        scalevec = self._context["prototypes_scalevec"][self.prototype]

        source_proto_align = superimpose_array(
            prev_lib.coordinates[self.source_conformers], prototype
        )[0]
        source_proto_align = {
            conf: v for conf, v in zip(self.source_conformers, source_proto_align)
        }

        target_proto_align = superimpose_array(
            lib.coordinates[self.target_conformers], prototype
        )[0]
        target_proto_align = {
            conf: v for conf, v in zip(self.target_conformers, target_proto_align)
        }

        target_rotaconformers = {}
        for target_conformer in self.target_conformers:
            rotaconformers = lib.get_rotamers(target_conformer)
            rotaconformers = Rotation.from_rotvec(rotaconformers).as_matrix()
            rotaconformers_align = (
                target_proto_align[target_conformer]
                .T.dot(rotaconformers)
                .swapaxes(0, 1)
            )
            target_rotaconformers[target_conformer] = rotaconformers_align

        origin_rotaconformers = self._context["origin_rotaconformers"]

        source_rotaconformers_align = {}
        candidates = []
        for source_conformer in self.source_conformers:
            source_rotaconformers = origin_rotaconformers[source_conformer].as_matrix()
            source_rotaconformers_align[source_conformer] = (
                source_proto_align[source_conformer]
                .T.dot(source_rotaconformers)
                .swapaxes(0, 1)
            )
        for source_conformer in self.source_conformers:
            for source_rota in source_rotaconformers_align[source_conformer]:
                rota_candidates = []
                pos = 0
                for target_conformer in self.target_conformers:
                    t_rotaconformers = target_rotaconformers[target_conformer]
                    rr = t_rotaconformers.dot(source_rota.T)
                    ax = Rotation.from_matrix(rr).as_rotvec()

                    ang = np.linalg.norm(ax, axis=1)
                    ang = np.maximum(ang, 0.0001)
                    ax /= ang[:, None]
                    fac = (np.cos(ang) - 1) ** 2 + np.sin(ang) ** 2
                    cross = (scalevec * scalevec) * (1 - ax * ax)
                    curr_rmsd = np.sqrt(fac * cross.sum(axis=-1))
                    curr_candidates = np.where(curr_rmsd < self._context["ovRMSD"])[0]
                    rota_candidates.append(curr_candidates + pos)
                    pos += len(t_rotaconformers)
                rota_candidates = np.concatenate(rota_candidates)
                candidates.append(rota_candidates)
        source_rotaconf_counts = np.array([len(cand) for cand in candidates])
        candidates = np.concatenate(candidates)

        self.membership = None
        self.rmsd_upper = None
        self.rmsd_lower = None
        self.source_rotaconf_counts = source_rotaconf_counts
        self.candidates = candidates
        self._has_run = True

    def _run_task_bruteforce(self) -> None:
        from . import GRIDSPACING

        lib, prev_lib = self._context["lib"], self._context["prev_lib"]

        target_rotaconformers = {}
        for target_conformer in self.target_conformers:
            rotaconformers = lib.get_rotamers(target_conformer)
            rotaconformers = Rotation.from_rotvec(rotaconformers).as_matrix()
            target_rotaconformers[target_conformer] = rotaconformers

        origin_rotaconformers = self._context["origin_rotaconformers"]

        target_conformer_coordinates = lib.coordinates[target_conformer]
        candidates = []
        for source_conformer in self.source_conformers:
            source_conformer_coordinates = prev_lib.coordinates[source_conformer]
            for source_rota_nr, source_rota in enumerate(
                origin_rotaconformers[source_conformer]
            ):
                reference = np.einsum(
                    "kj,jl->kl", source_conformer_coordinates, source_rota.as_matrix()
                )
                refe_com = reference.mean(axis=0)
                refe_c = reference - refe_com

                rota_candidates = []
                pos = 0
                for target_conformer in self.target_conformers:
                    target_conformer_coordinates = lib.coordinates[target_conformer]
                    t_rotaconformers = target_rotaconformers[target_conformer]

                    superpositions = np.einsum(
                        "kj,ijl->ikl", target_conformer_coordinates, t_rotaconformers
                    )
                    superpositions_c = (
                        superpositions - superpositions.mean(axis=1)[:, None]
                    )
                    dif = superpositions_c - refe_c
                    curr_rmsd = np.sqrt(np.einsum("ijk,ijk->i", dif, dif) / len(refe_c))

                    curr_candidates = np.where(curr_rmsd < self._context["ovRMSD"])[0]
                    rota_candidates.append(curr_candidates + pos)
                    pos += len(t_rotaconformers)
                rota_candidates = np.concatenate(rota_candidates)
                candidates.append(rota_candidates)
        source_rotaconf_counts = np.array([len(cand) for cand in candidates])
        candidates = np.concatenate(candidates)

        self.membership = None
        self.rmsd_upper = None
        self.rmsd_lower = None
        self.source_rotaconf_counts = source_rotaconf_counts
        self.candidates = candidates
        self._has_run = True


Task.run_task = Task._run_task_tensor


class TaskList:
    """Container around a collection of tasks with shared context."""

    MEMBERSHIP_BINS = np.array(
        [1.0, 1.25, 1.5, 1.75, 2.0, 2.25, 2.5, 2.75, 3.0, 3.25, 3.5, 3.75, 4, 4.5, 5]
    )
    WORKLOAD_THRESHOLD = 100e6

    def __init__(
        self,
        *,
        origin_rotaconformers: Dict[int, Rotation],
        prototypes: np.ndarray,
        prototype_clusters: Dict[int, Rotation],
        prototypes_scalevec: np.ndarray,
        conf_prototypes: Sequence[int],
        load_membership,
        motif: str,
        nucpos: int,
        ovRMSD: float,
        lib,
        prev_lib,
    ) -> None:
        self._initialize_state(
            origin_rotaconformers=origin_rotaconformers,
            prototypes=prototypes,
            prototype_clusters=prototype_clusters,
            prototypes_scalevec=prototypes_scalevec,
            conf_prototypes=conf_prototypes,
            load_membership=load_membership,
            motif=motif,
            nucpos=nucpos,
            ovRMSD=ovRMSD,
            prev_lib=prev_lib,
            lib=lib,
        )

    @property
    def proto_align(self) -> Dict[Tuple[int, int], np.ndarray]:
        return self._proto_align

    def __len__(self) -> int:
        return len(self._tasks)

    def __getitem__(self, index: int) -> Task:
        return self._tasks[index]

    def __iter__(self):
        return iter(self._tasks)

    def to_npz(self, filepath: str, *, compressed: bool = False) -> None:
        """Persist the task list to a single NPZ archive."""

        tasks = self._tasks
        if not tasks:
            raise ValueError("TaskList is empty; nothing to serialize")

        prototypes = np.array([task.prototype for task in tasks], dtype=int)
        rc_source = np.array([task.rc_source for task in tasks], dtype=int)
        rc_target = np.array([task.rc_target for task in tasks], dtype=int)
        prepared = np.array([task._prepared for task in tasks], dtype=bool)
        processed = np.array([task._has_run for task in tasks], dtype=bool)

        def _concat_int(arrays: List[np.ndarray]) -> np.ndarray:
            if not arrays:
                return np.array([], dtype=int)
            return np.concatenate(arrays).astype(int, copy=False)

        target_arrays = [
            task.target_conformers
            for task in tasks
            if task.target_conformers is not None
        ]
        source_arrays = [
            task.source_conformers
            for task in tasks
            if task.source_conformers is not None
        ]

        target_counts = np.array(
            [
                len(task.target_conformers) if task.target_conformers is not None else 0
                for task in tasks
            ],
            dtype=int,
        )
        source_counts = np.array(
            [
                len(task.source_conformers) if task.source_conformers is not None else 0
                for task in tasks
            ],
            dtype=int,
        )

        targets_concat = _concat_int(target_arrays)
        sources_concat = _concat_int(source_arrays)

        membership_shapes = np.zeros((len(tasks), 2), dtype=int)
        membership_data_parts: List[np.ndarray] = []
        for idx, task in enumerate(tasks):
            if task.membership is not None:
                membership_shapes[idx] = task.membership.shape
                membership_data_parts.append(task.membership.ravel())

        membership_data = (
            np.concatenate(membership_data_parts)
            if membership_data_parts
            else np.array([], dtype=np.uint8)
        )

        def _collect_matrix(
            tasks: List[Task], attr: str
        ) -> Tuple[np.ndarray, np.ndarray]:
            shapes = np.zeros((len(tasks), 2), dtype=int)
            parts: List[np.ndarray] = []
            for idx, task in enumerate(tasks):
                value = getattr(task, attr)
                if value is not None:
                    shapes[idx] = value.shape
                    parts.append(value.ravel())
            data = np.concatenate(parts) if parts else np.array([], dtype=np.uint8)
            return data, shapes

        rmsd_upper_data, rmsd_upper_shapes = _collect_matrix(tasks, "rmsd_upper")
        rmsd_lower_data, rmsd_lower_shapes = _collect_matrix(tasks, "rmsd_lower")

        proto_align_items = sorted(self._proto_align.items())
        if proto_align_items:
            proto_align_keys = np.array([k for k, _ in proto_align_items], dtype=int)
            proto_align_rotmat = np.stack([v for _, v in proto_align_items])
        else:
            proto_align_keys = np.empty((0, 2), dtype=int)
            proto_align_rotmat = np.empty((0, 3, 3))

        src_rotaconf_count_parts = [
            task.source_rotaconf_counts
            for task in tasks
            if task.source_rotaconf_counts is not None
        ]
        cand_parts = [task.candidates for task in tasks if task.candidates is not None]

        src_rotaconf_counts_counts = np.array(
            [
                (
                    len(task.source_rotaconf_counts)
                    if task.source_rotaconf_counts is not None
                    else 0
                )
                for task in tasks
            ],
            dtype=int,
        )
        candidate_counts = np.array(
            [
                len(task.candidates) if task.candidates is not None else 0
                for task in tasks
            ],
            dtype=int,
        )

        src_rotaconf_counts_data = (
            np.concatenate(src_rotaconf_count_parts).astype(int, copy=False)
            if src_rotaconf_count_parts
            else np.array([], dtype=int)
        )
        candidate_data = (
            np.concatenate(cand_parts).astype(int, copy=False)
            if cand_parts
            else np.array([], dtype=int)
        )

        saver = np.savez_compressed if compressed else np.savez
        print("DO SAVE")
        saver(
            filepath,
            prototype=prototypes,
            rc_source=rc_source,
            rc_target=rc_target,
            prepared=prepared,
            processed=processed,
            target_conformers=targets_concat,
            target_counts=target_counts,
            source_conformers=sources_concat,
            source_counts=source_counts,
            membership_data=membership_data,
            membership_shapes=membership_shapes,
            rmsd_upper_data=rmsd_upper_data,
            rmsd_upper_shapes=rmsd_upper_shapes,
            rmsd_lower_data=rmsd_lower_data,
            rmsd_lower_shapes=rmsd_lower_shapes,
            proto_align_keys=proto_align_keys,
            proto_align_rotmat=proto_align_rotmat,
            source_rotaconf_counts_data=src_rotaconf_counts_data,
            source_rotaconf_counts_counts=src_rotaconf_counts_counts,
            candidate_data=candidate_data,
            candidate_counts=candidate_counts,
        )

    def load_npz(self, filepath: str) -> None:
        """Load tasks from an NPZ archive previously created by `to_npz`."""

        if self._tasks:
            raise RuntimeError("Tasks already built")

        with np.load(filepath) as data:
            prototypes = data["prototype"]
            rc_source = data["rc_source"]
            rc_target = data["rc_target"]
            prepared = data["prepared"].astype(bool)
            processed = data["processed"].astype(bool)
            target_conformers = data["target_conformers"]
            target_counts = data["target_counts"]
            source_conformers = data["source_conformers"]
            source_counts = data["source_counts"]
            membership_data = data["membership_data"]
            membership_shapes = data["membership_shapes"]
            rmsd_upper_data = data["rmsd_upper_data"]
            rmsd_upper_shapes = data["rmsd_upper_shapes"]
            rmsd_lower_data = data["rmsd_lower_data"]
            rmsd_lower_shapes = data["rmsd_lower_shapes"]
            proto_align_keys = data["proto_align_keys"]
            proto_align_rotmat = data["proto_align_rotmat"]
            source_rotaconf_counts_data = data["source_rotaconf_counts_data"]
            source_rotaconf_counts_counts = data["source_rotaconf_counts_counts"]
            candidate_data = data["candidate_data"]
            candidate_counts = data["candidate_counts"]

        n_tasks = len(prototypes)

        target_offsets = np.concatenate(([0], np.cumsum(target_counts)))
        source_offsets = np.concatenate(([0], np.cumsum(source_counts)))

        membership_sizes = np.prod(membership_shapes, axis=1, dtype=int)
        membership_offsets = np.concatenate(([0], np.cumsum(membership_sizes)))

        rmsd_upper_sizes = np.prod(rmsd_upper_shapes, axis=1, dtype=int)
        rmsd_upper_offsets = np.concatenate(([0], np.cumsum(rmsd_upper_sizes)))

        rmsd_lower_sizes = np.prod(rmsd_lower_shapes, axis=1, dtype=int)
        rmsd_lower_offsets = np.concatenate(([0], np.cumsum(rmsd_lower_sizes)))

        src_rotaconf_offsets = np.concatenate(
            ([0], np.cumsum(source_rotaconf_counts_counts))
        )
        candidate_offsets = np.concatenate(([0], np.cumsum(candidate_counts)))

        tasks: List[Task] = []
        for idx in range(n_tasks):
            t_start, t_end = target_offsets[idx : idx + 2]
            s_start, s_end = source_offsets[idx : idx + 2]
            task = Task(
                nr=idx,
                prototype=int(prototypes[idx]),
                target_conformers=target_conformers[t_start:t_end],
                source_conformers=source_conformers[s_start:s_end],
                rc_source=int(rc_source[idx]),
                rc_target=int(rc_target[idx]),
            )
            task.finalize()
            task.set_context(self._context)

            if prepared[idx]:
                m_start, m_end = membership_offsets[idx : idx + 2]
                rmu_start, rmu_end = rmsd_upper_offsets[idx : idx + 2]
                rml_start, rml_end = rmsd_lower_offsets[idx : idx + 2]

                if membership_shapes[idx].prod():
                    task.membership = membership_data[m_start:m_end].reshape(
                        tuple(membership_shapes[idx])
                    )
                else:
                    task.membership = np.array([], dtype=membership_data.dtype).reshape(
                        tuple(membership_shapes[idx])
                    )

                if rmsd_upper_shapes[idx].prod():
                    task.rmsd_upper = rmsd_upper_data[rmu_start:rmu_end].reshape(
                        tuple(rmsd_upper_shapes[idx])
                    )
                else:
                    task.rmsd_upper = np.array([], dtype=rmsd_upper_data.dtype).reshape(
                        tuple(rmsd_upper_shapes[idx])
                    )

                if rmsd_lower_shapes[idx].prod():
                    task.rmsd_lower = rmsd_lower_data[rml_start:rml_end].reshape(
                        tuple(rmsd_lower_shapes[idx])
                    )
                else:
                    task.rmsd_lower = np.array([], dtype=rmsd_lower_data.dtype).reshape(
                        tuple(rmsd_lower_shapes[idx])
                    )

                task._prepared = True

            if processed[idx]:
                sc_start, sc_end = src_rotaconf_offsets[idx : idx + 2]
                cand_start, cand_end = candidate_offsets[idx : idx + 2]
                task.source_rotaconf_counts = source_rotaconf_counts_data[
                    sc_start:sc_end
                ]
                task.candidates = candidate_data[cand_start:cand_end]
                task._has_run = True

            tasks.append(task)

        self._tasks = tasks

        proto_align = {}
        for key, rotmat in zip(proto_align_keys, proto_align_rotmat):
            if len(key) != 2:
                continue
            proto = int(key[0])
            conf = int(key[1])
            proto_align[(proto, conf)] = rotmat

        self._proto_align = proto_align
        self._context["proto_align"] = proto_align

    def _initialize_state(
        self,
        *,
        origin_rotaconformers: Dict[int, Rotation],
        prototypes: np.ndarray,
        prototype_clusters: Dict[int, Rotation],
        prototypes_scalevec: np.ndarray,
        conf_prototypes: Sequence[int],
        load_membership,
        motif: str,
        nucpos: int,
        ovRMSD: float,
        lib,
        prev_lib,
    ) -> None:
        self._tasks: List[Task] = []
        self._proto_align: Dict[Tuple[int, int], Rotation] = {}
        self._context = {
            "origin_rotaconformers": origin_rotaconformers,
            "prototypes": prototypes,
            "prototype_clusters": prototype_clusters,
            "prototypes_scalevec": prototypes_scalevec,
            "conf_prototypes": conf_prototypes,
            "load_membership": load_membership,
            "motif": motif,
            "nucpos": nucpos,
            "ovRMSD": ovRMSD,
            "membership_bins": self.MEMBERSHIP_BINS,
            "lib": lib,
            "prev_lib": prev_lib,
        }

    def build_tasks(
        self,
        *,
        all_proto: Sequence[int],
        source_confs: Dict[int, Iterable[int]],
        conf_prototypes: Sequence[int],
        origin_rotaconformers: Dict[int, Rotation],
        crmsd_ok: np.ndarray,
    ) -> None:
        if self._tasks:
            raise RuntimeError("Tasks already built")

        tasks = self._tasks
        proto_align = self._proto_align
        lib = self._context["lib"]
        prev_lib = self._context["prev_lib"]
        prototypes = self._context["prototypes"]

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
            all_oriconfs_list = sorted(list(all_oriconfs))
            curr_proto_align = superimpose_array(
                prev_lib.coordinates[all_oriconfs_list], prototype
            )[0]
            for oriconf, mat in zip(all_oriconfs_list, curr_proto_align):
                proto_align[proto, oriconf] = mat

            def new_task() -> Task:
                curr_task = Task(nr=len(tasks), prototype=proto, context=self._context)
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
                while (
                    curr_task.rc_target * curr_task.rc_source < self.WORKLOAD_THRESHOLD
                ):
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

        self._context["proto_align"] = proto_align

        for task in tasks:
            task.finalize()
            task.set_context(self._context)
