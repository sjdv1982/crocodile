import gc
import sys
from typing import Optional

import numpy as np
from nefertiti.functions.superimpose import superimpose_array
from .reference import Reference
from .library import LibraryFactory, Library


def _fit_conformer(refe_coor, lib):
    rotmats, rmsds = superimpose_array(lib.coordinates, refe_coor)
    conformer = rmsds.argmin()
    rmsd = rmsds[conformer]
    conf_coor = lib.coordinates[conformer]
    refe_com = refe_coor.mean(axis=0)
    conf_com = conf_coor.mean(axis=0)
    offset = refe_com - conf_com
    rotmat = rotmats[conformer]
    if lib.conformer_mapping is not None:
        conformer2 = lib.conformer_mapping[conformer]
    else:
        conformer2 = conformer
    return conformer2, rotmat, offset, rmsd


def _fit_discrete(
    refe_coor, lib, lib2, rotamer_precision, grid_spacing, *, keep_best_conformer
):
    _, rmsds = superimpose_array(lib.coordinates, refe_coor)
    ini_threshold = rmsds.min() + rotamer_precision + grid_spacing
    ini_filter = rmsds < ini_threshold
    sorting = rmsds[ini_filter].argsort()
    conf_indices = np.arange(len(rmsds), dtype=int)[ini_filter][sorting]
    if keep_best_conformer:
        conf_indices = conf_indices[:1]

    refe_com = refe_coor.mean(axis=0)

    best = ini_threshold
    best_conformer = None
    best_rotamer = None
    best_matrix = None
    best_offset = None
    for conf0 in conf_indices:
        if rmsds[conf0] > best:
            continue
        if lib.conformer_mapping is not None:
            conf = lib.conformer_mapping[conf0]
        else:
            conf = conf0
        conformer_coordinates = lib2.coordinates[conf]
        rotamers = lib2.get_rotamers(conf)
        superpositions = np.einsum("kj,ijl->ikl", conformer_coordinates, rotamers)
        offsets = refe_com - superpositions.mean(axis=1)
        offsets_disc = np.round(offsets / grid_spacing) * grid_spacing
        dif = superpositions + offsets_disc[:, None] - refe_coor
        disc_rmsds = np.sqrt(np.einsum("ijk,ijk->i", dif, dif) / len(refe_coor))
        curr_best = disc_rmsds.min()
        if curr_best < best:
            best = curr_best
            best_conformer = conf
            best_rotamer = disc_rmsds.argmin()
            best_matrix = rotamers[best_rotamer].copy()
            best_offset = offsets_disc[best_rotamer]
    return best_conformer, best_matrix, best_rotamer, best_offset, best


def best_fit(
    fraglen: int,
    references: list[Reference],
    pdb_codes: Optional[list[str]],
    fragment_libraries: dict[str:LibraryFactory],
    *,
    discrete_representation: bool,
    keep_best_conformer: Optional[bool] = None,
    rotamer_precision: Optional[float] = None,
    grid_spacing: Optional[float] = None,
):
    """Calculates the best fit onto one or more references, using a fragment library.

    fraglen: fragment length

    pdb_codes (optional): the list containing the PDB code for each reference.
    If present, the fragment library must have PDB origin support.

    fragment_libraries: a dict of fragment library factories (LibraryFactory instances),
      where the keys are the fragment nucleotide sequences.

    discrete_representation: if False, calculate only the best fitting conformer.
    If True, also consider discrete rotamers and grid positioning.
    In that case:
    - The fragment library must have rotaconformer support
    - The rotamer precision (the maximum RMSD between rotamers)
       and the grid spacing (voxelsize) must be specified.
    - if keep_best_conformer, only consider the best conformer
       and calculate its best rotamer and offset

    Returns: a numpy array containing the best fit for each valid fragment.
    Its fields are:
        - fragment: the position of the fragment (starting from 1)
        - conformer: conformer index (starting from 0)
        - rotation: rotation matrix of superposition
        - rotamer: (only with discrete_representation). rotamer index for the conformer.
        - offset: conformer translation of superposition.
           With discrete_representation, offset is discretized to grid_spacing
    """
    if discrete_representation:
        assert rotamer_precision is not None
        assert grid_spacing is not None

    if pdb_codes is not None:
        assert len(pdb_codes) == len(references)
    else:
        pdb_codes = [None] * len(references)

    fragments = {seq: [] for seq in fragment_libraries}
    for refe_ind, (reference, pdb_code) in enumerate(zip(references, pdb_codes)):
        for fragpos in reference.get_fragment_positions(fraglen):
            seq = reference.sequence[fragpos - 1 : fragpos - 1 + fraglen]
            fragments[seq].append((refe_ind, fragpos))

    results0 = {}
    for seq in fragment_libraries:
        results0[seq] = []
        frags = fragments[seq]
        if not frags:
            continue

        libf: LibraryFactory = fragment_libraries[seq]
        if discrete_representation:
            print(f"Loading rotaconformers for {seq}...", file=sys.stderr)
            libf.load_rotaconformers()
            print("...loading done", file=sys.stderr)

        curr_pdb_code = None
        lib: Library = None
        lib2: Library = None
        for refe_ind, fragpos in frags:
            reference = references[refe_ind]
            refe_coor = reference.get_coordinates(fragpos, fraglen)
            pdb_code = pdb_codes[refe_ind]
            if pdb_code != curr_pdb_code:
                lib = libf.create(pdb_code=pdb_code, prune_conformers=True)
                if discrete_representation:
                    lib2 = libf.create(
                        pdb_code=pdb_code,
                        prune_conformers=False,
                        with_rotaconformers=True,
                    )
            assert lib.coordinates.shape[1] == len(refe_coor)

            if not discrete_representation:
                curr_result0 = _fit_conformer(refe_coor, lib)
            else:
                curr_result0 = _fit_discrete(
                    refe_coor,
                    lib,
                    lib2,
                    rotamer_precision,
                    grid_spacing,
                    keep_best_conformer=keep_best_conformer,
                )
            results0[seq].append(curr_result0)

        libf.unload_rotaconformers()
        del lib, lib2
        gc.collect()

    fields = [
        ("fragment", np.uint16),
        ("conformer", np.uint16),
        ("rotation", np.float64, (3, 3)),
    ]
    if discrete_representation:
        fields.append(("rotamer", np.uint32))

    fields += [
        ("offset", np.float32, 3),
        ("rmsd", np.float32),
    ]
    result_dtype = np.dtype(fields)
    results = []

    seq_counter = {seq: 0 for seq in fragments}
    for refe_ind, reference in enumerate(references):
        fragment_positions = reference.get_fragment_positions(fraglen)
        if not fragment_positions:
            results.append(None)
            continue
        curr_results = np.empty(len(fragment_positions), dtype=result_dtype)
        for frag_ind, fragpos in enumerate(fragment_positions):
            seq = reference.sequence[fragpos - 1 : fragpos - 1 + fraglen]
            cpos = seq_counter[seq]
            assert fragments[seq][cpos] == (refe_ind, fragpos), (
                fragments[seq][cpos],
                (refe_ind, fragpos),
            )
            seq_counter[seq] += 1
            r1 = results0[seq][cpos]
            r2 = curr_results[frag_ind]
            r2["fragment"] = fragpos
            if discrete_representation:
                conformer, rotmat, rotamer, offset, rmsd = r1
            else:
                conformer, rotmat, offset, rmsd = r1
            r2["conformer"] = conformer
            r2["rotation"] = rotmat
            if discrete_representation:
                r2["rotamer"] = rotamer
            r2["offset"] = offset
            r2["rmsd"] = rmsd

        results.append(curr_results)
    return results
