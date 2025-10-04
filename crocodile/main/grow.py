import numpy as np
from tqdm import tqdm, trange
from scipy.spatial.transform import Rotation

"""
from crocodile.nuc.all_fit import (
    all_fit,
    conformer_mask_from_crmsd,
    conformer_mask_from_general_pairing,
    conformer_masks_from_specific_pairing,
)
"""
from crocodile.nuc.library import LibraryFactory
from crocodile.main.superimpose import superimpose_array
from crocodile.main import tensorlib


def _grow_from_anchor(command, constraints, state):
    from crocodile_library_config import dinucleotide_libraries

    origin = command["origin"]
    assert origin in ("anchor-up", "anchor-down")

    fragment = command["fragment"]
    key = "frag" + str(fragment)
    assert key in constraints

    if origin == "anchor-up":
        prev_frag = fragment - 1
        nucleotide_mask = [True, False]
    else:
        prev_frag = fragment + 1
        nucleotide_mask = [False, True]

    anchor_key = origin.replace("-", "_")
    assert anchor_key in constraints[key], constraints[key]
    anchor = constraints[key][anchor_key]
    if "base" in anchor:
        only_base = True
        cRMSD = None
        ovRMSD = anchor["base"]
    else:
        only_base = False
        cRMSD = anchor["full"]["cRMSD"]
        ovRMSD = anchor["full"]["ovRMSD"]

    refe = state["reference"]
    seq = refe.get_sequence(fragment, fraglen=2)

    pdb_code = constraints["pdb_code"]
    libf: LibraryFactory = dinucleotide_libraries[seq]
    libf.load_rotaconformers()
    print("START")

    lib = libf.create(
        pdb_code=pdb_code,
        nucleotide_mask=nucleotide_mask,
        only_base=only_base,
        with_rotaconformers=True,
    )

    anchor_coors = refe.get_coordinates(fragment, fraglen=2)[lib.atom_mask]
    anchor_offset = anchor_coors.mean(axis=0)
    anchor_coors = anchor_coors - anchor_offset

    lib_offset = lib.coordinates.mean(axis=1)
    lib_coors = lib.coordinates - lib_offset[:, None]

    lib_cmat, lib_crmsd = superimpose_array(lib_coors, anchor_coors)
    anchor_tensor, anchor_scalevec = tensorlib.get_structure_tensor(anchor_coors)
    lib_indices = np.arange(len(lib_coors)).astype(int)
    if not only_base:
        # 1. calculate merged scalevecs => lib_scalevec
        # 2. apply cRMSD filter to lib_cmat, lib_indices
        raise NotImplementedError
    else:
        lib_scalevec = np.repeat(anchor_scalevec, len(lib_coors))
    lib_cmat_t = lib_cmat.dot(anchor_tensor)

    for n in trange(len(lib_indices)):
        conf = lib_indices[n]
        scalevec = lib_scalevec[n]
        curr_lib_cmat_t = lib_cmat_t[n]
        rotamers0 = lib.get_rotamers(conf)
        rotamers = Rotation.from_rotvec(rotamers0)
        rotamers_t = rotamers.as_matrix().dot(lib_cmat_t[n])
        continue
        msd_conformational = lib_crmsd[n] ** 2
        msd_rotational = tensorlib.get_msd(rotamers_t, None, scalevec)
        msd_maxtrans = ovRMSD**2 - msd_conformational - msd_rotational
        # print(msd_maxtrans.shape, (msd_maxtrans>0).sum(), msd_maxtrans.max())


def _grow_from_fragment(command, constraints, state):
    from crocodile_library_config import dinucleotide_libraries

    print(command)

    fragment = command["fragment"]
    key = "frag" + str(fragment)
    assert key in constraints

    origin = command["origin"]
    origin_poses = np.load(f"state/{origin}.npy")  ###
    prev_frag = int(open(f"state/{origin}.FRAG").read())  ###
    prev_key = "frag" + str(prev_frag)

    if prev_frag == fragment - 1:
        nucleotide_mask = [True, False]
        prev_nucleotide_mask = [False, True]
        frag0, frag1 = prev_key, key
    elif prev_frag == fragment + 1:
        nucleotide_mask = [False, True]
        prev_nucleotide_mask = [True, False]
        frag0, frag1 = key, prev_key
    else:
        raise AssertionError((fragment, prev_frag))

    for p in constraints["pairs"]:
        if p["down"] == frag0 and p["up"] == frag1:
            ovRMSD = p["ovRMSD"]
            cRMSD = p["cRMSD"]
            constraint_pair = p
            break
    else:
        exit("Cannot find constraint fragment pair")

    refe = state["reference"]
    seq = refe.get_sequence(fragment, fraglen=2)

    pdb_code = constraints["pdb_code"]
    prev_seq = refe.get_sequence(prev_frag, fraglen=2)
    prev_libf: LibraryFactory = dinucleotide_libraries[prev_seq]

    prev_libf.load_rotaconformers()
    prev_lib = prev_libf.create(
        pdb_code=pdb_code,
        nucleotide_mask=prev_nucleotide_mask,
        with_rotaconformers=True,
    )

    print("START")

    libf: LibraryFactory = dinucleotide_libraries[seq]
    lib = libf.create(
        pdb_code=pdb_code,
        nucleotide_mask=nucleotide_mask,
        with_rotaconformers=False,
    )

    lib_offset = lib.coordinates.mean(axis=1)
    lib_coors = lib.coordinates - lib_offset[:, None]

    prev_lib_offset = prev_lib.coordinates.mean(axis=1)
    prev_lib_coors = prev_lib.coordinates - prev_lib_offset[:, None]

    print(f"Origin poses: {len(origin_poses)}")

    origin_conformers = np.unique(origin_poses["conformer"])
    print(f"Unique origin conformers: {len(origin_conformers)}")

    rc = 100000 * origin_poses["conformer"].astype(np.uint64) + origin_poses["rotamer"]
    rc = np.array(list(set(rc)))
    origin_rotaconformers = np.stack((rc / 100000, rc % 100000), axis=1)
    print(f"Unique origin rotaconformers: {len(origin_rotaconformers)}")
    lib_crmsd = np.empty((len(origin_conformers), len(lib_coors)))
    lib_cmat = np.empty((len(origin_conformers), len(lib_coors), 3, 3))
    for confnr, conf in enumerate(tqdm(origin_conformers)):
        curr_lib_cmat, curr_lib_crmsd = superimpose_array(
            lib_coors, prev_lib_coors[conf]
        )
        lib_cmat[confnr] = curr_lib_cmat
        lib_crmsd[confnr] = curr_lib_crmsd
    lib_crmsd_ok = lib_crmsd <= cRMSD
    ori_cconf, target_cconf = np.where(lib_crmsd_ok)
    target_confs = []
    ori_cconf_pos0 = np.searchsorted(ori_cconf, np.arange(len(origin_conformers)))
    ori_cconf_pos0 = np.append(ori_cconf_pos0, [len(ori_cconf)])
    ori_cconf_pos = np.stack((ori_cconf_pos0[:-1], ori_cconf_pos0[1:]), axis=1)
    for conf, (p1, p2) in enumerate(ori_cconf_pos):
        if p1 == p2:
            continue
        assert np.all(ori_cconf[p1:p2] == conf), (
            ori_cconf[p1:p2],
            conf,
        )
        target_confs.append(target_cconf[p1:p2])
    origin_conformers_mask = np.unique(ori_cconf)
    origin_conformers = origin_conformers[origin_conformers_mask]
    assert len(origin_conformers) == len(target_confs)
    print("Post-cRMSD: ")
    print(f"Unique origin conformers: {len(origin_conformers)}")
    origin_rotaconformer_rotvecs = []
    for conf in origin_conformers:
        r = prev_lib.get_rotamers(conf)
        mask = origin_poses["conformer"] == conf
        rota = origin_poses["rotamer"][mask]
        rota_uniq = np.unique(rota)
        origin_rotaconformer_rotvecs.append(r[rota_uniq])
    print(
        f"Unique origin rotaconformers: {sum([len(r) for r in origin_rotaconformer_rotvecs])}"
    )

    # Unload rotaconformers from origin library and load them in target library
    if prev_seq == seq:
        prev_libf, libf = libf, prev_libf
        prev_lib, lib = lib, prev_lib
    else:
        prev_libf.unload_rotaconformers()
        prev_lib = prev_libf.create(
            pdb_code=pdb_code,
            nucleotide_mask=prev_nucleotide_mask,
            with_rotaconformers=False,
        )
        libf.load_rotaconformers()
        lib = libf.create(
            pdb_code=pdb_code,
            nucleotide_mask=nucleotide_mask,
            with_rotaconformers=True,
        )

    print(
        f"Target rotaconformers, naive: {len(origin_rotaconformers) * len(lib.rotaconformers):.3e}"
    )
    print(
        f"Average target conformers per origin conformer: {lib_crmsd_ok[origin_conformers_mask].sum(axis=1).mean():.2f}"
    )

    ncand = 0
    for curr_origin_confs, curr_target_confs in zip(
        origin_rotaconformer_rotvecs, target_confs
    ):
        for target_conf in curr_target_confs:
            r = lib.get_rotamers(target_conf)
            ncand += len(r) * len(curr_origin_confs)
    print(f"Target rotaconformers, cRMSD-filtered: {ncand:.3e}")

    exit()
    # ori_rotaconf
    # for

    anchor_tensor, anchor_scalevec = tensorlib.get_structure_tensor(anchor_coors)
    lib_indices = np.arange(len(lib_coors)).astype(int)
    if not only_base:
        # 1. calculate merged scalevecs => lib_scalevec
        # 2. apply cRMSD filter to lib_cmat, lib_indices
        raise NotImplementedError
    else:
        lib_scalevec = np.repeat(anchor_scalevec, len(lib_coors))
    lib_cmat_t = lib_cmat.dot(anchor_tensor)

    for n in trange(len(lib_indices)):
        conf = lib_indices[n]
        scalevec = lib_scalevec[n]
        curr_lib_cmat_t = lib_cmat_t[n]
        rotamers0 = lib.get_rotamers(conf)
        rotamers = Rotation.from_rotvec(rotamers0)
        rotamers_t = rotamers.as_matrix().dot(lib_cmat_t[n])
        continue
        msd_conformational = lib_crmsd[n] ** 2
        msd_rotational = tensorlib.get_msd(rotamers_t, None, scalevec)
        msd_maxtrans = ovRMSD**2 - msd_conformational - msd_rotational
        # print(msd_maxtrans.shape, (msd_maxtrans>0).sum(), msd_maxtrans.max())


def grow(command, constraints, state):
    assert command["type"] == "grow", command
    print(command)
    print("GROW")

    if command["origin"] in ("anchor-up", "anchor-down"):
        return _grow_from_anchor(command, constraints, state)
    else:
        return _grow_from_fragment(command, constraints, state)
