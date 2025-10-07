import os
import numpy as np
from tqdm import tqdm, trange
from scipy.spatial.transform import Rotation

from juliacall import Main

GRIDSPACING = np.sqrt(3) / 3
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
            constraint_pair = p  # TODO: primary pair
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
    for conf in origin_conformer_list:
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

    tasks = TaskList(
        origin_rotaconformers=origin_rotaconformers,
        prototype_clusters=prototype_clusters,
        prototypes_scalevec=prototypes_scalevec,
        conf_prototypes=conf_prototypes,
        load_membership=_load_membership,
        motif=motif,
        nucpos=nucpos,
        ovRMSD=ovRMSD,
    )
    tasks.build_tasks(
        all_proto=all_proto,
        source_confs=source_confs,
        conf_prototypes=conf_prototypes,
        prototypes=prototypes,
        prev_lib=prev_lib,
        origin_rotaconformers=origin_rotaconformers,
        lib=lib,
        crmsd_ok=crmsd_ok,
        superimpose=superimpose,
    )

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

    candpool = {}
    for proto in all_proto:
        p = {
            "source_conformer": [],
            "source_rotamer": [],
            "source_mat": [],
            "target_conformer": [],
            "target_rotamer": [],
            "target_mat": [],
            "remain_msd": [],
        }
        candpool[proto] = p

    def process_task(task):
        proto_align = tasks.proto_align
        assert task.membership is not None
        assert task.rmsd_upper is not None
        assert task.rmsd_lower is not None

        membership = task.membership
        rmsd_upper = task.rmsd_upper
        rmsd_lower = task.rmsd_lower

        csource_rotaconf = rmsd_upper.shape[1]
        ctarget_rotaconf = membership.shape[1]

        source_rotaconf_counts, candidates = Main.CrocoCandidates.compute_candidates(
            membership,
            rmsd_upper,
            rmsd_lower,
        )
        source_rotaconf_counts = source_rotaconf_counts.to_numpy()
        candidates = candidates.to_numpy() - 1

        csource_conformers, ctarget_conformers = (
            task.source_conformers,
            task.target_conformers,
        )
        source_rotaconf_boundaries = np.cumsum(source_rotaconf_counts)
        source_conf_boundaries = np.cumsum(
            [len(origin_rotaconformers[conf]) for conf in csource_conformers]
        )
        assert source_conf_boundaries[-1] == csource_rotaconf

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

        assert target_conf_boundaries[-1] == ctarget_rotaconf

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
        cand_target_rotaconf_trueind = ctarget_rotaconformers_trueind[
            candidates
        ]  # alternative: subtract conf boundary from candidates for more speed
        target_rotvec = ctarget_rotaconformers[candidates]
        target_mat = Rotation.from_rotvec(target_rotvec).as_matrix()
        trans_target = np.einsum("ik,ikl->il", lib_offset[cand_target_conf], target_mat)
        dif_trans = trans_source - trans_target
        err_disc_trans = dif_trans - GRIDSPACING * np.round(dif_trans / GRIDSPACING)
        assert err_disc_trans.ndim == 2
        rmsd_disc_trans = np.sqrt((err_disc_trans**2).sum(axis=1))

        cand_conf_rmsd = crmsd[cand_source_conf, cand_target_conf]
        cand_msd_remain = ovRMSD**2 - cand_conf_rmsd**2 - rmsd_disc_trans

        cand_mask = cand_msd_remain > 0

        candidates = candidates[cand_mask]
        cand_msd_remain = cand_msd_remain[cand_mask]
        cand_source_conf = cand_source_conf[cand_mask]
        cand_target_conf = cand_target_conf[cand_mask]
        source_mat = source_mat[cand_mask]
        target_mat = target_mat[cand_mask]

        print(len(cand_mask), cand_mask.sum())
        p = candpool[task.prototype]
        p["source_conformer"].append(cand_source_conf)
        p["source_rotamer"].append(cand_source_rotaconf_trueind[cand_mask])
        p["source_mat"].append(source_mat)
        p["target_conformer"].append(cand_target_conf)
        p["target_rotamer"].append(cand_target_rotaconf_trueind[cand_mask])
        p["target_mat"].append(target_mat)
        p["remain_msd"].append(cand_msd_remain)

    for n in trange(len(tasks)):
        task = tasks[n]
        task.prepare_task(semaphore)
        semaphore.release()
        process_task(task)

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
                    semaphore.release()
                    progress.update()
                    process_task(tasks[idx])
    """

    for p in candpool.values():
        for k in p:
            p[k] = np.concatenate(p[k])
        for k in p:
            assert len(p[k]) == len(p["remain_msd"]), (
                k,
                len(p[k]),
                len(p["remain_msd"]),
            )
    print("cand", sum([len(p["remain_msd"]) for p in candpool.values()]))


def grow(command, constraints, state):
    assert command["type"] == "grow", command
    print(command)
    print("GROW")

    if command["origin"] in ("anchor-up", "anchor-down"):
        return _grow_from_anchor(command, constraints, state)
    else:
        return _grow_from_fragment(command, constraints, state)
