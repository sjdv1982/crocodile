import numpy as np
from scipy.spatial.transform import Rotation
from . import GRIDSPACING


class CandidatePool:
    def __init__(self, prototypes):
        self._data = {
            proto: {
                "source_conformer": [],
                "source_rotamer": [],
                "source_mat": [],
                "target_conformer": [],
                "target_rotamer": [],
                "target_mat": [],
                "remain_msd": [],
                "candidate_mask": [],
            }
            for proto in prototypes
        }
        self._finalized = False

    def process(
        self,
        task,
        *,
        origin_rotaconformers,
        prev_lib_offset,
        lib_offset,
        lib,
        crmsd,
        ovRMSD,
    ):
        source_rotaconf_counts = task.source_rotaconf_counts
        candidates = task.candidates
        if source_rotaconf_counts is None or candidates is None:
            raise RuntimeError("Task must be prepared before processing candidate pool")

        csource_conformers = task.source_conformers
        ctarget_conformers = task.target_conformers

        source_rotaconf_boundaries = np.cumsum(source_rotaconf_counts)
        source_conf_boundaries = np.cumsum(
            [len(origin_rotaconformers[conf]) for conf in csource_conformers]
        )
        csource_rotaconf = source_conf_boundaries[-1]

        ctarget_rotaconformers0 = [
            lib.get_rotamers(conf) for conf in ctarget_conformers
        ]
        target_conf_boundaries = np.cumsum([len(c) for c in ctarget_rotaconformers0])
        ctarget_rotaconformers = np.concatenate(ctarget_rotaconformers0)

        ctarget_rotaconformers_trueind = np.concatenate(
            [
                np.arange(len(lib.get_rotamers(conf)), dtype=int)
                for conf in ctarget_conformers
            ]
        )

        ind_source_rotaconf = np.searchsorted(
            source_rotaconf_boundaries - 1,
            np.arange(len(candidates)),
        )
        ii_source_conf = np.searchsorted(
            source_conf_boundaries - 1,
            np.arange(csource_rotaconf),
        )
        ind_source_conf = ii_source_conf[ind_source_rotaconf]
        ind_target_conf = np.searchsorted(target_conf_boundaries, candidates)

        cand_source_conf = csource_conformers[ind_source_conf]
        cand_target_conf = ctarget_conformers[ind_target_conf]

        csource_rotaconformers = Rotation.concatenate(
            [origin_rotaconformers[conf] for conf in csource_conformers]
        )
        csource_rotaconformers_trueind = np.concatenate(
            [
                np.arange(len(origin_rotaconformers[conf]), dtype=int)
                for conf in csource_conformers
            ]
        )

        cand_source_rotaconf_trueind = csource_rotaconformers_trueind[
            ind_source_rotaconf
        ]
        source_mat = csource_rotaconformers[ind_source_rotaconf].as_matrix()
        trans_source = np.einsum(
            "ik,ikl->il", prev_lib_offset[cand_source_conf], source_mat
        )
        cand_target_rotaconf_trueind = ctarget_rotaconformers_trueind[candidates]
        target_rotvec = ctarget_rotaconformers[candidates]
        target_mat = Rotation.from_rotvec(target_rotvec).as_matrix()
        trans_target = np.einsum("ik,ikl->il", lib_offset[cand_target_conf], target_mat)
        dif_trans = trans_source - trans_target
        err_disc_trans = dif_trans - GRIDSPACING * np.round(dif_trans / GRIDSPACING)
        rmsd_disc_trans = np.sqrt((err_disc_trans**2).sum(axis=1))

        cand_conf_rmsd = crmsd[cand_source_conf, cand_target_conf]
        cand_msd_remain = ovRMSD**2 - cand_conf_rmsd**2 - rmsd_disc_trans

        cand_mask = cand_msd_remain > 0

        print(len(cand_mask), cand_mask.sum())
        self._append(
            task,
            cand_source_conf,
            cand_source_rotaconf_trueind,
            source_mat,
            cand_target_conf,
            cand_target_rotaconf_trueind,
            target_mat,
            cand_msd_remain,
            cand_mask,
        )

    def _append(
        self,
        task,
        cand_source_conf,
        cand_source_rotamer,
        source_mat,
        cand_target_conf,
        cand_target_rotamer,
        target_mat,
        cand_msd_remain,
        cand_mask,
    ):
        pool = self._data[task.prototype]
        pool["source_conformer"].append(cand_source_conf)
        pool["source_rotamer"].append(cand_source_rotamer)
        pool["source_mat"].append(source_mat)
        pool["target_conformer"].append(cand_target_conf)
        pool["target_rotamer"].append(cand_target_rotamer)
        pool["target_mat"].append(target_mat)
        pool["remain_msd"].append(cand_msd_remain)
        pool["candidate_mask"].append(cand_mask)

    def finalize(self):
        if self._finalized:
            return
        max_items = 0
        for pool in self._data.values():
            max_items = max(max_items, len(pool["candidate_mask"]))
        for pool in self._data.values():
            for key in pool:
                if key == "candidate_mask":
                    continue
                pool[key] = np.concatenate(pool[key])
            for key in pool:
                if key == "candidate_mask":
                    continue
                assert len(pool[key]) == len(pool["remain_msd"]), (
                    key,
                    len(pool[key]),
                    len(pool["remain_msd"]),
                )
            pool["candidate_mask"] = []
        self._finalized = True

    def values(self):
        return self._data.values()

    def total_candidates(self):
        return sum(len(pool["remain_msd"]) for pool in self._data.values())

    def get_item(self, proto, index):
        if self._finalized:
            raise RuntimeError("Cannot access individual items after finalize")
        if proto not in self._data:
            raise KeyError(proto)
        pool = self._data[proto]
        return {key: values[index] for key, values in pool.items()}

    def apply_cand_mask(self):
        if self._finalized:
            raise RuntimeError("Cannot apply mask after finalize")
        for pool in self._data.values():
            for index in range(len(pool["candidate_mask"])):
                mask = pool["candidate_mask"][index]
                if mask is None:
                    continue
                mask = np.asarray(mask, dtype=bool)
                for key in (
                    "source_conformer",
                    "source_rotamer",
                    "target_conformer",
                    "target_rotamer",
                    "remain_msd",
                ):
                    pool[key][index] = pool[key][index][mask]
                pool["source_mat"][index] = pool["source_mat"][index][mask]
                pool["target_mat"][index] = pool["target_mat"][index][mask]
                pool["candidate_mask"][index] = None

    def concatenate_prototypes(self):
        result = {}
        for key in (
            "source_conformer",
            "source_rotamer",
            "target_conformer",
            "target_rotamer",
            "remain_msd",
        ):
            r = np.concatenate([self._data[proto][key] for proto in self._data])
            result[key] = r
        return result
