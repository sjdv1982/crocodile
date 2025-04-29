import numpy as np
from typing import Optional
from nefertiti.functions.superimpose import superimpose_array
from tqdm import tqdm

from .library import Library

_mesh_ax = np.arange(-10, 12, 1, dtype=int)
_mx, _my, _mz = np.meshgrid(_mesh_ax, _mesh_ax, _mesh_ax)
_mx2, _my2, _mz2 = (
    np.fix(_mx - 0.1).astype(int),
    np.fix(_my - 0.1).astype(int),
    np.fix(_mz - 0.1).astype(int),
)
_msq = (_mx2**2 + _my2**2 + _mz2**2).astype(int)


def all_fit(
    reference: np.ndarray,
    *,
    fragment_library: Library,
    rmsd_threshold: float,
    conformer_rmsd_threshold: Optional[float] = None,
    rotamer_precision: float,
    grid_spacing: float,
    return_rotation_matrices: Optional[bool] = True,
    return_rotamer_indices: Optional[bool] = False,
) -> np.ndarray:
    """Get all discrete-representation poses within an RMSD threshold.

    reference: (X, 3) numpy array of coordinates, where X is the number of atoms.

    fragment_library: Library instance. conformers must contain X atoms.
    Must have rotaconformer support.

    rmsd_threshold: the maximum RMSD towards the reference.
    If the reference is an anchor, this is the anchor RMSD.
    If the reference is (part of) another discrete-representation pose, it is the overlap RMSD.

    conformer_rmsd_threshold: optional.
    If specified, only consider conformers with direct-superposition-onto-the-reference RMSD
      is below this threshold.
    This is typically specified when the reference is (part of) another discrete-representation pose,
     in which case it is the compatibility RMSD.

    rotamer_precision: maximum RMSD between rotamers.

    grid_spacing: spacing (voxel size) of the translational grid.

    return_rotation_matrices: ...
    return_rotamer_indices: ...
    """
    assert return_rotamer_indices or return_rotation_matrices
    assert conformer_rmsd_threshold is None or conformer_rmsd_threshold < rmsd_threshold

    fields = [("conformer", np.uint16)]
    if return_rotation_matrices:
        fields.append(("rotation", np.float64, (3, 3)))

    if return_rotamer_indices:
        fields.append(("rotamer", np.uint32))

    fields += [
        ("offset", np.float32, 3),
        ("rmsd", np.float32),
    ]
    result_dtype = np.dtype(fields)
    results = np.empty(1000, result_dtype)
    nresults = 0

    _, rmsds = superimpose_array(fragment_library.coordinates, reference)
    if conformer_rmsd_threshold is not None:
        conf_mask = rmsds < conformer_rmsd_threshold
    else:
        conf_mask = rmsds < rmsd_threshold
    if fragment_library.conformer_mask is not None:
        conf_mask &= fragment_library.conformer_mask

    refe_com = reference.mean(axis=0)
    o = np.zeros(3)
    rmsd_threshold_sq = rmsd_threshold * rmsd_threshold
    grid_spacing_sq = grid_spacing * grid_spacing
    rotaconformers_clustering = fragment_library.rotaconformers_clustering

    for conf in tqdm(np.where(conf_mask)[0]):
        conformer_coordinates = fragment_library.coordinates[conf]
        rotamers = fragment_library.get_rotamers(conf)

        if rotaconformers_clustering:
            clus_rotamers_ind, clustering = fragment_library.get_rotamer_clusters(conf)
            assert len(clustering) == len(rotamers)
            """
            assert np.allclose(
                np.unique(clustering),
                np.arange(0, len(rotamers), dtype=clustering.dtype),
            )
            """
            clus_sup = np.einsum(
                "kj,ijl->ikl",
                conformer_coordinates,
                rotamers[clustering[clus_rotamers_ind[:-1]]],
            )
            clus_offsets = refe_com - clus_sup.mean(axis=1)
            dif = clus_sup + clus_offsets[:, None] - reference
            clus_rmsds = np.sqrt(np.einsum("ijk,ijk->i", dif, dif) / len(reference))
            candidate_rotamers = []
            candidate_rotamers0 = []
            for c in np.where(clus_rmsds < 2.0 + 2 * rotamer_precision)[0]:
                i0, i1 = clus_rotamers_ind[c : c + 2]
                cc = clustering[i0:i1]
                candidate_rotamers0 += cc.tolist()
            if candidate_rotamers0:
                candidate_rotamers0 = np.array(candidate_rotamers0).astype(int)
                cand_superpositions = np.einsum(
                    "kj,ijl->ikl",
                    conformer_coordinates,
                    rotamers[candidate_rotamers0],
                )
                cand_offsets = refe_com - cand_superpositions.mean(axis=1)

                offsets_disc = np.round(cand_offsets / grid_spacing) * grid_spacing
                dif = cand_superpositions + offsets_disc[:, None] - reference
                rmsds_disc = np.sqrt(np.einsum("ijk,ijk->i", dif, dif) / len(reference))
                candidate_rotamers_ind = np.where(rmsds_disc < rmsd_threshold)[0]
                candidate_rotamers = candidate_rotamers0[candidate_rotamers_ind]
                candidate_superpositions = cand_superpositions[candidate_rotamers_ind]
                candidate_offsets = cand_offsets[candidate_rotamers_ind]
        else:
            superpositions = np.einsum("kj,ijl->ikl", conformer_coordinates, rotamers)
            offsets = refe_com - superpositions.mean(axis=1)
            offsets_disc = np.round(offsets / grid_spacing) * grid_spacing
            dif = superpositions + offsets_disc[:, None] - reference
            rmsds_disc = np.sqrt(np.einsum("ijk,ijk->i", dif, dif) / len(reference))
            candidate_rotamers = np.where(rmsds_disc < rmsd_threshold)[0]
            candidate_superpositions = superpositions[candidate_rotamers]
            candidate_offsets = offsets[candidate_rotamers]

        for rotamer, sup, off in zip(
            candidate_rotamers, candidate_superpositions, candidate_offsets
        ):
            off_floor = np.floor(off / grid_spacing) * grid_spacing
            dif = sup + off - reference
            rmsd_sq = (dif * dif).sum() / len(reference)
            margin = np.floor((rmsd_threshold_sq - rmsd_sq) / grid_spacing_sq)
            mask = _msq <= margin
            inds = np.stack((_mx[mask], _my[mask], _mz[mask]), axis=1)

            curr_offsets = off_floor + inds * grid_spacing
            curr_offset_errors = curr_offsets - off
            offset_msd = (curr_offset_errors * curr_offset_errors).sum(axis=1)

            rmsds_sq = rmsd_sq + offset_msd
            mask = rmsds_sq < rmsd_threshold_sq
            new_rmsds = rmsds_sq[mask]
            new_offsets = curr_offsets[mask]

            nresults2 = nresults + len(new_rmsds)
            if nresults2 > len(results):
                new_results = np.empty(int(len(results) * 1.1), result_dtype)
                new_results[:nresults] = results[:nresults]
                results = new_results

            curr_results = results[nresults:nresults2]
            curr_results["conformer"] = conf
            if return_rotamer_indices:
                curr_results["rotamer"] = rotamer
            if return_rotation_matrices:
                curr_results["rotation"] = rotamers[rotamer]
            curr_results["rmsd"] = new_rmsds
            curr_results["offset"] = new_offsets
            nresults = nresults2
    return results[:nresults]
