import copy
import os
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
from crocodile.main.superimpose import superimpose_array, superimpose
from crocodile.main.tasks import TaskList
from crocodile.main import tensorlib
from crocodile.main.candidate_pool import CandidatePool


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
    digits = f"{(conformer+1)%100:02d}"
    outputdir = f"rotamember/{digits}"
    membership_file = (
        f"{outputdir}/lib-nuc-{motif}-nuc{nucpos}-{conformer+1}-cluster-membership.npy"
    )

    membership0 = np.load(membership_file)

    assert int(nclust / 2 + 0.5) == len(membership0), len(membership0)

    membership = np.empty((nclust, membership0.shape[1]), np.uint8)
    membership[::2] = membership0 & 15
    remain = len(membership[1::2])
    membership[1::2] = membership0[:remain] >> 4
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
    origin_poses = origin_poses[:1]  ###
    prev_frag = int(open(f"state/{origin}.FRAG").read())  ###
    prev_key = "frag" + str(prev_frag)

    if prev_frag == fragment - 1:
        nucleotide_mask = [True, False]
        prev_nucleotide_mask = [False, True]
        frag0, frag1 = prev_key, key
        nucpos = 1
    elif prev_frag == fragment + 1:
        nucleotide_mask = [False, True]
        prev_nucleotide_mask = [True, False]
        frag0, frag1 = key, prev_key
        nucpos = 2
    else:
        raise AssertionError((fragment, prev_frag))

    for p in constraints["pairs"]:
        if p["down"] == frag0 and p["up"] == frag1:
            ovRMSD = p["ovRMSD"]
            cRMSD = p["cRMSD"]
            constraint_pair = p  # TODO: primary pair
            break
    else:
        exit("Cannot find constraint fragment pair")

    refe = state["reference"]
    seq = refe.get_sequence(fragment, fraglen=2)
    motif = seq.replace("U", "C").replace("G", "A")

    conf_prototypes = np.load(
        f"rotaclustering/dinuc-{motif}-nuc{nucpos}-assign-prototypes.npy"
    )
    common_base = motif[0] if nucpos == "1" else motif[1]
    prototypes_scalevec = np.loadtxt(f"monobase-prototypes-{common_base}-scalevec.txt")
    prototypes = np.load(f"monobase-prototypes-{common_base}.npy")
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
    """
    m1 = origin_poses[0]["rotation"]
    m2 = prev_lib.get_rotamers(origin_poses[0]["conformer"])[origin_poses[0]["rotamer"]]
    print(m1)
    print(Rotation.from_rotvec(m2).as_matrix())
    """

    print("START")

    libf: LibraryFactory = dinucleotide_libraries[seq]

    prototype_clusters = {}
    for prototype_index in range(nprototypes):
        prototype_cluster_file = f"rotaclustering/lib-nuc-{motif}-nuc{nucpos}-prototype-{prototype_index+1}-clusters.npy"
        curr_prototype_clusters = Rotation.from_matrix(np.load(prototype_cluster_file))
        prototype_clusters[prototype_index] = curr_prototype_clusters

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

    trimotif = prev_seq[0].replace("G", "A").replace("U", "C") + motif
    crmsd_file = os.path.join(f"rotamember/crmsd_full_matrix_{trimotif}.npy")
    if not os.path.exists(crmsd_file):
        crmsd = []
        for conf_coors in tqdm(
            prev_lib_coors, desc="cRMSD file not found, calculating on the fly..."
        ):
            _, curr_crmsd = superimpose_array(lib_coors, conf_coors)
            crmsd.append(curr_crmsd)
        crmsd = np.stack(crmsd)
        np.save(crmsd_file, crmsd)
    else:
        crmsd = np.load(crmsd_file)
    crmsd_ok = crmsd <= cRMSD

    lib_crmsd_ok = crmsd_ok[origin_conformer_list]
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
    origin_rotaconformer_indices = {}
    for conf in origin_conformer_list:
        r = prev_lib.get_rotamers(conf)
        mask = origin_poses["conformer"] == conf
        rota = origin_poses["rotamer"][mask]
        rota_uniq = np.unique(rota)
        origin_rotaconformer_indices[conf] = rota_uniq
        origin_rotaconformers[conf] = Rotation.from_rotvec(r[rota_uniq])
    print(
        f"Unique origin rotaconformers: {sum([len(r) for r in origin_rotaconformers.values()])}"
    )

    # Unload rotaconformers from origin library and load them in target library
    if prev_seq == seq:
        assert libf is prev_libf
        prev_lib = prev_libf.create(
            pdb_code=pdb_code,
            nucleotide_mask=prev_nucleotide_mask,
            with_rotaconformers=False,
        )
        lib = libf.create(
            pdb_code=pdb_code,
            nucleotide_mask=nucleotide_mask,
            with_rotaconformers=True,
        )

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

    tasks = TaskList(
        origin_rotaconformers=origin_rotaconformers,
        prototypes=prototypes,
        prototype_clusters=prototype_clusters,
        prototypes_scalevec=prototypes_scalevec,
        conf_prototypes=conf_prototypes,
        load_membership=_load_membership,
        motif=motif,
        nucpos=nucpos,
        ovRMSD=ovRMSD,
        prev_lib=prev_lib,
        lib=lib,
    )
    npz_filename = "OUTPUT/test.npz"
    if os.path.exists(npz_filename):
        print("LOAD")
        tasks.load_npz(npz_filename)
        print("/LOAD")

    else:

        tasks.build_tasks(
            all_proto=all_proto,
            source_confs=source_confs,
            conf_prototypes=conf_prototypes,
            origin_rotaconformers=origin_rotaconformers,
            crmsd_ok=crmsd_ok,
        )

        ###from . import julia_import as _
        from threading import Semaphore
        from concurrent.futures import (
            ThreadPoolExecutor,
            wait,
            FIRST_COMPLETED,
            ALL_COMPLETED,
        )

        semaphore = Semaphore(
            10
        )  # pre-load up to 10 tasks. After that, wait until a load has been consumed

        for n in trange(len(tasks)):
            task = tasks[n]
            task.prepare_task(semaphore)
            semaphore.release()
            task.run_task()

        """
        with ThreadPoolExecutor(
            max_workers=min(len(tasks), 20)
        ) as executor:  # load data for 20
            pending = {
                executor.submit(tasks[tasknr].prepare_task, semaphore): tasknr
                for tasknr in range(len(tasks))
            }
            with tqdm(None, total=len(tasks)) as progress:
                while pending:
                    done, _ = wait(set(pending), return_when=FIRST_COMPLETED)
                    for fut in done:
                        idx = pending.pop(fut)
                        fut.result()
                        task = tasks[idx]
                        semaphore.release()
                        task.run_task()
                        progress.update()
        """
        print("SAVE")
        tasks.to_npz(npz_filename)
        print("/SAVE")

    candpool = CandidatePool(all_proto)

    for n in trange(len(tasks)):
        candpool.process(
            tasks[n],
            origin_rotaconformers=origin_rotaconformers,
            origin_rotaconformer_indices=origin_rotaconformer_indices,
            prev_lib_offset=prev_lib_offset,
            lib_offset=lib_offset,
            lib=lib,
            crmsd=crmsd,
            ovRMSD=ovRMSD,
        )

    poses6 = np.load("TEST/frag6-rx/poses-filtered.npy")
    poses7 = np.load("TEST/frag7-fwd/poses.npy")
    origins = np.loadtxt("TEST/frag7-fwd/poses-origins.txt", dtype=int) - 1

    ####
    poses6 = poses6[: len(origin_poses)]
    poses7 = poses7[origins < len(origin_poses)]
    origins = origins[origins < len(origin_poses)]
    print(set(zip(poses7["conformer"].tolist(), poses7["rotamer"].tolist())))
    ####

    poses6 = poses6[np.unique(origins)]
    poses6_conf = poses6["conformer"]
    poses6_rota = poses6["rotamer"]
    poses7_conf = poses7["conformer"]
    poses7_rota = poses7["rotamer"]
    candpool2 = copy.deepcopy(candpool)

    candpool.finalize()
    cand = candpool.concatenate_prototypes()
    print("cand", candpool.total_candidates())

    def check(cand):
        def rc_key(conf, rota):
            assert conf.ndim == 1 and rota.ndim == 1
            return 100000 * conf.astype(np.uint32) + rota

        rc6 = set(rc_key(poses6_conf, poses6_rota))
        rc7 = set(rc_key(poses7_conf, poses7_rota))

        src_confs = np.unique(cand["source_conformer"])
        target_confs = np.unique(cand["target_conformer"])
        src_rc = set(rc_key(cand["source_conformer"], cand["source_rotamer"]))
        target_rc = set(rc_key(cand["target_conformer"], cand["target_rotamer"]))
        for conf in np.unique(poses6_conf):
            if not conf in src_confs:
                print("SRC CONF MISSING", conf)
        for conf in np.unique(poses7_conf):
            if not conf in target_confs:
                print("TARGET CONF MISSING", conf)
        print("SRC RC MISSING", len(rc6.difference(src_rc)), "/", len(rc6))
        print("TARGET RC MISSING", len(rc7.difference(target_rc)), "/", len(rc7))
        print(poses7_conf[0], poses7_rota[0])

    check(cand)
    print("OK")
    print()

    candpool2.apply_cand_mask()
    candpool2.finalize()
    print("cand2", candpool2.total_candidates())
    cand2 = candpool2.concatenate_prototypes()
    check(cand2)


def grow(command, constraints, state):
    assert command["type"] == "grow", command
    print(command)
    print("GROW")

    if command["origin"] in ("anchor-up", "anchor-down"):
        return _grow_from_anchor(command, constraints, state)
    else:
        return _grow_from_fragment(command, constraints, state)
