import os
import sys
from typing import Optional

import numpy as np
from scipy.spatial.transform import Rotation

sys.path.append(os.path.join(os.path.dirname(__file__), "..", "code"))
from library import config
from offsets import get_discrete_offsets, expand_discrete_offsets, gather_discrete_offsets
from superimpose import superimpose


def pick_rotamer_with_com_constraint(
    rotamers: np.ndarray,
    coords: np.ndarray,
    ref_com: Optional[np.ndarray],
    max_dist: float = 3.0,
    max_tries: int = 100000,
) -> tuple[np.ndarray, np.ndarray, int]:
    if rotamers is None or len(rotamers) == 0:
        raise ValueError("No rotamers available for conformer")

    use_rotvec = rotamers.ndim == 2 and rotamers.shape[1] == 3
    for _ in range(max_tries):
        idx = np.random.randint(len(rotamers))
        rotmat = rotamers[idx]
        if use_rotvec:
            rotmat = Rotation.from_rotvec(rotmat).as_matrix()
        coords_rot = coords.dot(rotmat)
        com = coords_rot.mean(axis=0)
        if ref_com is None:
            return rotmat, com, idx
        if np.linalg.norm(com - ref_com) <= max_dist:
            return rotmat, com, idx

    raise RuntimeError(
        "Unable to find rotamer within COM constraint; increase max_tries or relax constraint"
    )


def compatible_conformers(
    coords_a: np.ndarray, coords_b_all: np.ndarray, rmsd_cutoff: float = 0.5
) -> np.ndarray:
    compatible = []
    for idx, coords_b in enumerate(coords_b_all):
        _, rmsd = superimpose(coords_a, coords_b)
        if rmsd < rmsd_cutoff:
            compatible.append(idx)
    return np.array(compatible, dtype=int)


def generate_pose_test_data(seed: int = 0) -> dict[str, object]:
    np.random.seed(seed)

    dinucleotide_libraries, _ = config(verify_checksums=False)

    libf_aa = dinucleotide_libraries["AA"]
    libf_aa.load_rotaconformers()
    lib_aa = libf_aa.create(
        pdb_code=None, nucleotide_mask=[False, True], with_rotaconformers=True
    )

    aa_coords = lib_aa.coordinates
    if len(aa_coords) < 20:
        raise RuntimeError("AA library has fewer than 20 conformers")

    aa_conf_indices = np.random.choice(len(aa_coords), size=20, replace=False)
    aa_coms = []
    ref_com = None
    for i, conf_idx in enumerate(aa_conf_indices):
        coords = aa_coords[conf_idx]
        rotamers = lib_aa.get_rotamers(conf_idx)
        if i == 0:
            _, com, _ = pick_rotamer_with_com_constraint(rotamers, coords, None)
            ref_com = com
        else:
            _, com, _ = pick_rotamer_with_com_constraint(
                rotamers, coords, ref_com, max_dist=3.0
            )
        aa_coms.append(com)

    libf_aa.unload_rotaconformers()

    libf_ac = dinucleotide_libraries["AC"]
    libf_ac.load_rotaconformers()
    lib_ac = libf_ac.create(
        pdb_code=None, nucleotide_mask=[True, False], with_rotaconformers=True
    )

    ac_coords = lib_ac.coordinates

    ac_coms = []
    ac_conf_indices = []
    ac_rot_indices = []
    for conf_idx in aa_conf_indices:
        coords_a = aa_coords[conf_idx]
        compat = compatible_conformers(coords_a, ac_coords, rmsd_cutoff=0.5)
        if compat.size == 0:
            raise RuntimeError("No compatible AC conformers found for AA conformer")
        replace = compat.size < 10
        compat_sel = np.random.choice(compat, size=10, replace=replace)
        ac_conf_indices.append(compat_sel)

        compat_coms = []
        compat_rot_indices = []
        for conf2_idx in compat_sel:
            coords_b = ac_coords[conf2_idx]
            rotamers_b = lib_ac.get_rotamers(conf2_idx)

            coms_b = []
            _, com0, rot_idx0 = pick_rotamer_with_com_constraint(
                rotamers_b, coords_b, None
            )
            coms_b.append(com0)
            rot_idx_list = [rot_idx0]
            for _ in range(9):
                _, com, rot_idx = pick_rotamer_with_com_constraint(
                    rotamers_b, coords_b, com0, max_dist=3.0
                )
                coms_b.append(com)
                rot_idx_list.append(rot_idx)
            compat_coms.append(np.stack(coms_b))
            compat_rot_indices.append(np.array(rot_idx_list, dtype=np.uint16))

        ac_coms.append(np.stack(compat_coms))
        ac_rot_indices.append(np.stack(compat_rot_indices))

    libf_ac.unload_rotaconformers()

    aa_coms = np.stack(aa_coms)
    ac_coms = np.stack(ac_coms)
    ac_conf_indices = np.stack(ac_conf_indices)
    ac_rot_indices = np.stack(ac_rot_indices)

    assert aa_coms.shape == (20, 3)
    assert ac_coms.shape == (20, 10, 10, 3)
    assert ac_conf_indices.shape == (20, 10)
    assert ac_rot_indices.shape == (20, 10, 10)

    displacements = []
    radii = []
    conf_indices = []
    rot_indices = []
    for i in range(20):
        for j in range(10):
            for k in range(10):
                displacements.append(aa_coms[i] - ac_coms[i, j, k])
                radii.append(np.random.uniform(-0.2, 1.2))
                conf_indices.append(ac_conf_indices[i, j])
                rot_indices.append(ac_rot_indices[i, j, k])

    displacements = np.array(displacements, dtype=np.float32)
    radii = np.array(radii, dtype=np.float32)
    conf_indices = np.array(conf_indices, dtype=np.uint16)
    rot_indices = np.array(rot_indices, dtype=np.uint16)

    grid_scale = np.sqrt(3.0) / 3.0
    displacements_scaled = displacements / grid_scale
    radii_scaled = radii / grid_scale

    disp_indices, p_indices, rounded, reverse_map = get_discrete_offsets(
        displacements_scaled, radii_scaled
    )
    translation_lists = expand_discrete_offsets(
        disp_indices, p_indices, rounded, reverse_map
    )
    tinds, tdata = gather_discrete_offsets(translation_lists)

    return {
        "conf_indices": conf_indices,
        "rot_indices": rot_indices,
        "offsets_tuple": (disp_indices, p_indices, rounded, reverse_map),
        "translation_lists": translation_lists,
        "gathered": (tinds, tdata),
    }
