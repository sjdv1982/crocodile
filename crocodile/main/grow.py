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


def _load_membership(motif, nucpos, conformer, nclust):
    digits = f"{conformer%100:02d}"
    outputdir = f"rotamember/{digits}"
    membership_file = (
        f"{outputdir}/lib-nuc-{motif}-nuc{nucpos}-{conformer+1}-cluster-membership.npy"
    )

    membership0 = np.load(membership_file)

    assert int(nclust / 2 + 0.5) == len(membership0), len(membership0)

    membership = np.empty((nclust, membership0.shape[1]), np.uint8)
    membership[::2] = membership0 & 15
    membership[1::2] = membership0 >> 4
    return membership


def _grow_from_fragment(command, constraints, state):
    from crocodile_library_config import dinucleotide_libraries

    print(command)

    fragment = command["fragment"]
    key = "frag" + str(fragment)
    assert key in constraints

    origin = command["origin"]
    origin_poses = np.load(f"state/{origin}.npy")  ###
    origin_poses = origin_poses[:1000]  ###
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
    motif = seq.replace("U", "C").replace("G", "A")
    nucpos = np.where(prev_nucleotide_mask)[0][0] + 1

    conf_prototypes = np.load(
        f"rotaclustering/dinuc-{motif}-nuc{nucpos}-assign-prototypes.npy"
    )
    common_base = motif[0] if nucpos == "1" else motif[1]
    prototypes_scalevec = np.loadtxt(f"monobase-prototypes-{common_base}-scalevec.txt")
    nprototypes = len(prototypes_scalevec)

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

    prototype_clusters = {}
    for prototype_index in range(nprototypes):
        prototype_cluster_file = f"rotaclustering/lib-nuc-{motif}-nuc{nucpos}-prototype-{prototype_index+1}-clusters.npy"
        curr_prototype_cluster_indices = np.load(prototype_cluster_file)
        curr_prototype_clusters = Rotation.from_rotvec(
            libf.rotaconformers[curr_prototype_cluster_indices]
        )
        prototype_clusters[prototype_index] = curr_prototype_clusters
        print(prototype_index, len(curr_prototype_clusters))

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

    origin_conformer_list = np.unique(origin_poses["conformer"])
    print(f"Unique origin conformers: {len(origin_conformer_list)}")

    rc = (origin_poses["conformer"].max() + 1) * origin_poses["rotamer"].astype(
        np.uint64
    ) + origin_poses["conformer"]
    rc = np.array(list(set(rc)))
    origin_rotaconformer_list = np.stack((rc / 100000, rc % 100000), axis=1)
    print(f"Unique origin rotaconformers: {len(origin_rotaconformer_list)}")
    lib_crmsd = np.empty((len(origin_conformer_list), len(lib_coors)))
    lib_cmat = np.empty((len(origin_conformer_list), len(lib_coors), 3, 3))
    for confnr, conf in enumerate(tqdm(origin_conformer_list)):
        curr_lib_cmat, curr_lib_crmsd = superimpose_array(
            lib_coors, prev_lib_coors[conf]
        )
        lib_cmat[confnr] = curr_lib_cmat
        lib_crmsd[confnr] = curr_lib_crmsd
    lib_crmsd_ok = lib_crmsd <= cRMSD
    ori_cconf, target_cconf = np.where(lib_crmsd_ok)
    target_confs = []
    ori_cconf_pos0 = np.searchsorted(ori_cconf, np.arange(len(origin_conformer_list)))
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
    origin_conformers_select = np.unique(ori_cconf)
    np.sort(origin_conformers_select)

    lib_crmsd_ok = lib_crmsd_ok[origin_conformers_select]
    origin_conformer_list = origin_conformer_list[origin_conformers_select]
    assert len(target_confs) == len(origin_conformer_list)
    target_confs = {
        conf: confs for conf, confs in zip(origin_conformer_list, target_confs)
    }
    assert len(origin_conformer_list) == len(target_confs)
    source_confs = {}
    for source_conf in origin_conformer_list:
        for target_conf in target_confs[source_conf]:
            if target_conf not in source_confs:
                source_confs[target_conf] = set()
            source_confs[target_conf].add(source_conf)
    target_conformer_list = list(source_confs.keys())

    print("Post-cRMSD: ")
    print(f"Unique origin conformers: {len(origin_conformer_list)}")
    print(
        f"Unique target conformers: {len(target_conformer_list)} / {len(lib.coordinates)}"
    )
    origin_rotaconformers = {}
    for conf in tqdm(origin_conformer_list):
        r = prev_lib.get_rotamers(conf)
        mask = origin_poses["conformer"] == conf
        rota = origin_poses["rotamer"][mask]
        rota_uniq = np.unique(rota)
        origin_rotaconformers[conf] = Rotation.from_rotvec(r[rota_uniq])
    print(
        f"Unique origin rotaconformers: {sum([len(r) for r in origin_rotaconformers.values()])}"
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
        f"Target rotaconformers, naive: {len(origin_rotaconformer_list) * len(lib.rotaconformers):.3e}"
    )
    print(
        f"Average target conformers per origin conformer: {np.mean([len(v) for v in target_confs.values()]):.2f} / {len(lib.coordinates)}"
    )
    print(
        f"Average origin conformers per target conformer: {np.mean([len(v) for v in source_confs.values()]):.2f} / {len(prev_lib.coordinates)}"
    )

    ncand = 0
    for conf in origin_conformer_list:
        for target_conf in target_confs[conf]:
            r = lib.get_rotamers(target_conf)
            ncand += len(r) * len(origin_rotaconformers[conf])
    print(f"Target rotaconformers, cRMSD-filtered: {ncand:.3e}")

    all_proto = np.unique(conf_prototypes[target_conformer_list])
    tasks = []
    for proto in all_proto:

        # Create tasks consisting of:
        # - One or more target conformers, and the corresponding rotaconformers
        # - All source conformers of the current prototype that are compatible
        # - All corresponding source rotaconformers
        # - A total workload of (source rotaconformers) x (target rotaconformers) >= 50 million

        conf_pairs = {
            tconf: oriconfs
            for tconf, oriconfs in source_confs.items()
            if conf_prototypes[tconf] == proto
        }

        def new_task():
            curr_task = {
                "prototype": proto,
                "target_conformers": [],
                "source_conformers": set(),
                "rc_source": 0,
                "rc_target": 0,
            }
            tasks.append(curr_task)

        target_conf_list = sorted(conf_pairs.keys(), key=lambda k: -len(conf_pairs[k]))
        target_conf_done = set()
        for target_conf in target_conf_list:
            if target_conf in target_conf_done:
                continue
            target_conf_done.add(target_conf)
            new_task()
            curr_task = tasks[-1]
            rc = lib.get_rotamers(target_conf)
            curr_task["rc_target"] = len(rc)
            curr_task["target_conformers"].append(int(target_conf))
            for source_conf in conf_pairs[target_conf]:
                curr_task["source_conformers"].add(int(source_conf))
                rc = origin_rotaconformers[source_conf]
                curr_task["rc_source"] += len(rc)
            while curr_task["rc_target"] * curr_task["rc_source"] < 50e6:
                best_target_conf = None
                lowest_waste = None
                for target_conf in target_conf_list:
                    if target_conf in target_conf_done:
                        continue
                    waste = 0
                    for source_conf in conf_pairs[target_conf]:
                        if source_conf not in curr_task["source_conformers"]:
                            rc = origin_rotaconformers[source_conf]
                            waste += len(rc) * curr_task["rc_target"]
                    if lowest_waste is None or waste < lowest_waste:
                        best_target_conf = target_conf
                        lowest_waste = waste

                if best_target_conf is None:  # target confs is exhausted
                    break

                target_conf_done.add(best_target_conf)
                rc = lib.get_rotamers(best_target_conf)
                curr_task["rc_target"] += len(rc)
                curr_task["target_conformers"].append(int(best_target_conf))
                for source_conf in conf_pairs[best_target_conf]:
                    curr_task["source_conformers"].add(int(source_conf))
                    rc = origin_rotaconformers[source_conf]
                    curr_task["rc_source"] += len(rc)

    membership_bins = np.array(
        [1.0, 1.25, 1.5, 1.75, 2.0, 2.25, 2.5, 2.75, 3.0, 3.25, 3.5, 3.75, 4, 4.5, 5]
    )

    def run_task(tasknr):

        task = tasks[tasknr]
        csource_conformers, ctarget_conformers = (
            sorted(list(task["source_conformers"])),
            task["target_conformers"],
        )
        csource_rotaconformers = Rotation.concatenate(
            [origin_rotaconformers[conf] for conf in csource_conformers]
        )
        ctarget_rotaconformers = np.concatenate(
            [lib.get_rotamers(conf) for conf in ctarget_conformers]
        )

        proto = task["prototype"]
        clusters = prototype_clusters[proto]
        scalevec = prototypes_scalevec[proto]

        rmsd = np.empty((len(csource_rotaconformers), len(clusters)))
        for n, cluster in enumerate(clusters):
            rr = csource_rotaconformers * cluster.inv()
            ax = rr.as_rotvec()
            ang = np.linalg.norm(ax, axis=1)
            ang = np.maximum(ang, 0.0001)
            ax /= ang[:, None]
            fac = (np.cos(ang) - 1) ** 2 + np.sin(ang) ** 2
            cross = (scalevec * scalevec) * (1 - ax * ax)
            curr_rmsd = np.sqrt(fac * cross.sum(axis=-1))
            rmsd[:, n] = curr_rmsd

        rmsd_upper = np.digitize(rmsd + ovRMSD, membership_bins).astype(np.uint8)
        rmsd_lower = np.digitize(rmsd - ovRMSD, membership_bins).astype(np.uint8)
        # up to here, for 20k structures: 10s

        for conformer in ctarget_conformers:
            assert conf_prototypes[conformer] == proto, (
                conformer,
                conf_prototypes[conformer],
                proto,
            )
            conformer = 1200  ##
            # nclust = len(prototype_clusters[proto])
            nclust = len(prototype_clusters[conf_prototypes[conformer]])  ###
            _load_membership(motif, nucpos, conformer, nclust)

        with runner:
            pass

    from threading import Semaphore
    from concurrent.futures import ThreadPoolExecutor

    runner = Semaphore(10)  # run 10 Julia instances at a time

    with ThreadPoolExecutor(max_workers=20) as executor:  # load data for 20
        candidates = list(
            tqdm(executor.map(run_task, range(len(tasks))), total=len(tasks))
        )

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
