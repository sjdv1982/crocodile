import sys
import json
import os
from hashlib import md5

import numpy as np

output_directory = sys.argv[1]
directories = sys.argv[2:]

if os.path.exists(output_directory):
    print(f"Output directory '{output_directory}' already exists")
    exit(1)

fragment_dirs0 = {}
dirs_origin = {}
for direc in directories:
    frag = int(json.load(open(os.path.join(direc, "FRAG"))))
    if frag not in fragment_dirs0:
        fragment_dirs0[frag] = []
    fragment_dirs0[frag].append(direc)
    origin = None
    originfile = os.path.join(direc, "ORIGIN")
    if os.path.exists(originfile):
        origin = int(json.load(open(originfile)))
    dirs_origin[direc] = origin

print("Fragments:")
print(json.dumps(fragment_dirs0, indent=2))
print()
print("Origins:")
print(json.dumps(dirs_origin, indent=2))
print()

merged_fragments = {}
fragment_dir = {}
fragment_origin = {}
for fragment, dirs in fragment_dirs0.items():
    if len(dirs) > 1:
        merged_fragments[fragment] = dirs
        md5sum = None
        for d in dirs:
            posedata = open(os.path.join(d, "poses-filtered.npy"), "rb").read()
            curr_md5sum = md5(posedata, usedforsecurity=False).hexdigest()
            if md5sum is None:
                curr_md5sum = md5sum
            if curr_md5sum != md5sum:
                print(
                    f"'{dirs[0]}' and '{d}' have the same fragment, but don't contain the same poses. Use reindex."
                )
                exit(1)
    else:
        fragment_dir[fragment] = dirs[0]
        fragment_origin[fragment] = dirs_origin[dirs[0]]

for fragment in list(fragment_origin.keys()):
    ori = fragment_origin[fragment]
    if ori not in fragment_dir and ori not in merged_fragments:
        fragment_origin[fragment] = None

subchains = []
done = set()
for frag in fragment_origin:
    if frag in done:
        continue
    if fragment_origin[frag] is not None:
        continue
    has_long_subchains = False
    while 1:
        subchains.append([])
        subchains[-1].append(frag)
        next_ori = frag
        while 1:
            for frag2 in fragment_origin:
                if frag2 in done:
                    continue
                if fragment_origin[frag2] != next_ori:
                    continue
                has_long_subchains = True
                subchains[-1].append(frag2)
                done.add(frag2)
                next_ori = frag2
                break
            else:
                break
        if len(subchains[-1]) == 1:
            if has_long_subchains:
                subchains = subchains[:-1]
            try:
                done.remove(subchains[-1][-1])
            except KeyError:
                pass
            break

subchain_results = []
for subchain in subchains:
    all_poses = None
    all_rmsds = None
    prev_origins = None
    pose_mapping = None
    first_pose_mapping = None
    for fragmentnr, fragment in enumerate(reversed(subchain)):
        direc = fragment_dir[fragment]
        poses = np.load(os.path.join(direc, "poses-filtered.npy"))
        rmsds = np.loadtxt(os.path.join(direc, "poses-filtered.rmsd"), ndmin=1)
        assert len(poses) == len(rmsds), direc
        if fragmentnr == 0:
            all_poses = np.empty((len(poses), len(subchain)), dtype=poses.dtype)
            all_rmsds = np.empty((len(poses), len(subchain)))

        pose_mapping = None
        pose_mapping_file = os.path.join(direc, "pose-mapping.txt")
        if os.path.exists(pose_mapping_file):
            pose_mapping = {}
            for l in open(pose_mapping_file).readlines():
                ll = l.split()
                pose_mapping[int(ll[0])] = int(ll[1])
        if fragmentnr == 0:
            first_pose_mapping = pose_mapping

        next_poses = poses
        next_rmsds = rmsds
        pose_inds = None
        if prev_origins is not None:
            pose_inds = prev_origins.copy()
            if pose_mapping is not None:
                pose_inds = np.array([pose_mapping[ind] for ind in pose_inds])
            next_poses = poses[pose_inds - 1]
            next_rmsds = rmsds[pose_inds - 1]
        all_poses[:, fragmentnr] = next_poses
        all_rmsds[:, fragmentnr] = next_rmsds

        origins = None
        origin_file = os.path.join(direc, "poses-filtered-origins.txt")
        if os.path.exists(origin_file):
            origins = np.loadtxt(origin_file, ndmin=1).astype(int)
            assert len(origins) == len(poses)
            if pose_inds is not None:
                origins = origins[pose_inds - 1]

        prev_origins = origins
    # reverse columns
    all_poses = all_poses[:, ::-1]
    all_rmsds = all_rmsds[:, ::-1]
    subchain_results.append((all_poses, all_rmsds, first_pose_mapping, pose_inds))

print("Initial subchains")
print(subchains)
print()

# lambda/hat (^) merge: add merged fragments
for fragment, fragdirs in merged_fragments.items():
    assert len(fragdirs) == 2, (fragment, fragdirs)
    ori1, ori2 = [dirs_origin[d] for d in fragdirs]
    subch1 = [
        subch for subch, subchain in enumerate(subchains) if subchain[-1] == ori1
    ][0]
    subch2 = [
        subch for subch, subchain in enumerate(subchains) if subchain[-1] == ori2
    ][0]
    if subchains[subch2][0] < subchains[subch1][0]:
        subch1, subch2 = subch2, subch1
        ori1, ori2 = ori2, ori
        fragdirs[:] = fragdirs[1], fragdirs[0]

    sub1 = subchain_results[subch1]
    sub2 = subchain_results[subch2]

    # poses and RMSDs are the same between subdirs
    poses = np.load(os.path.join(fragdirs[0], "poses-filtered.npy"))
    rmsds = np.loadtxt(os.path.join(fragdirs[0], "poses-filtered.rmsd"), ndmin=1)

    all_halfposes = []
    all_halfposes_rmsd = []
    all_subposes_inds = []
    for direc, sub in zip(fragdirs, (sub1, sub2)):

        subpose_mapping = sub[2]
        subpose_rev_mapping = None
        if subpose_mapping is not None:
            subpose_rev_mapping = {v: k for k, v in subpose_mapping.items()}
        if subpose_rev_mapping is not None:
            subpose_inds_mapped = [
                subpose_rev_mapping[ind] for ind in range(1, len(sub[0]) + 1)
            ]
        else:
            subpose_inds_mapped = [ind for ind in range(1, len(sub[0]) + 1)]
        subpose_inds_mapped = np.array(subpose_inds_mapped)
        origin_file = os.path.join(direc, "poses-filtered-origins.txt")
        origins = np.loadtxt(origin_file, ndmin=1).astype(int)

        subpose_inds_mapped = subpose_inds_mapped.tolist()
        subpose_inds = np.array([subpose_inds_mapped.index(ind) + 1 for ind in origins])
        halfposes = sub[0][subpose_inds - 1]
        all_halfposes.append(halfposes)
        halfposes_rmsd = sub[1][subpose_inds - 1]
        all_halfposes_rmsd.append(halfposes_rmsd)
        all_subposes_inds.append(subpose_inds)

    all_halfposes = [
        all_halfposes[0],
        poses[:, None],
        all_halfposes[1],
    ]
    all_halfposes_rmsd = [
        all_halfposes_rmsd[0],
        rmsds[:, None],
        all_halfposes_rmsd[1],
    ]  # reverse order
    new_poses = np.concatenate(all_halfposes, axis=1)
    new_rmsd = np.concatenate(all_halfposes_rmsd, axis=1)
    subchains.append(subchains[subch1] + [fragment] + subchains[subch2])
    subchain_results.append(
        (new_poses, new_rmsd, sub2[2], sub1[3][all_subposes_inds[0] - 1])
    )
    subchains = [v for i, v in enumerate(subchains) if i not in (subch1, subch2)]
    subchain_results = [
        v for i, v in enumerate(subchain_results) if i not in (subch1, subch2)
    ]

print("Subchains after hat merge, before V merge")
print(subchains)
print()

# V merge: Cartesian combination of sub-chains with the same root
while 1:
    change = False
    for s1, subchain1 in enumerate(list(subchains)):
        for s2, subchain2 in enumerate(list(subchains)):
            if s1 == s2:
                continue
            if subchain1[0] == subchain2[0]:
                if min(subchain1[1:]) > min(subchain2[1:]):
                    continue
                sub1 = subchain_results[s1]
                sub2 = subchain_results[s2]
                ind1, ind2 = sub1[3], sub2[3]
                nfrags = len(subchain1) + len(subchain2) - 1
                sub_inds = set(ind1).intersection(ind2)
                all_new_poses = []
                all_new_rmsd = []
                for sub_ind in sub_inds:
                    mask1 = ind1 == sub_ind
                    mask2 = ind2 == sub_ind

                    poses1block = sub1[0][mask1]
                    poses2block = sub2[0][mask2]
                    interpose = poses1block[0, 0]
                    assert interpose == poses2block[0, 0]
                    if len(poses1block) > 1:
                        assert interpose == poses1block[1, 0]
                    new_poses = np.empty(
                        (len(poses1block) * len(poses2block), nfrags), poses1block.dtype
                    )
                    inds1 = np.arange(len(new_poses)) // len(poses2block)
                    inds2 = np.arange(len(new_poses)) % len(poses2block)
                    cols1 = poses1block.shape[1]
                    cols2 = poses2block.shape[1]

                    poses1block_rev = poses1block[:, ::-1]
                    new_poses[:, : cols1 - 1] = poses1block_rev[:, :cols1][inds1][
                        :, :-1
                    ]
                    new_poses[:, cols1 - 1] = interpose
                    new_poses[:, -cols2 + 1 :] = poses2block[:, -cols2:][inds2][:, 1:]

                    rmsd1block = sub1[1][mask1]
                    rmsd2block = sub2[1][mask2]
                    inter_rmsd = rmsd1block[0, 0]
                    assert inter_rmsd == rmsd2block[0, 0]
                    if len(rmsd1block) > 1:
                        assert inter_rmsd == rmsd1block[1, 0]
                    new_rmsd = np.empty(new_poses.shape)

                    rmsd1block_rev = rmsd1block[:, ::-1]
                    new_rmsd[:, : cols1 - 1] = rmsd1block_rev[:, :cols1][inds1][:, :-1]
                    new_rmsd[:, cols1 - 1] = inter_rmsd
                    new_rmsd[:, -cols2 + 1 :] = rmsd2block[:, -cols2:][inds2][:, 1:]

                    all_new_poses.append(new_poses)
                    all_new_rmsd.append(new_rmsd)

                all_new_poses = np.concatenate(all_new_poses)
                all_new_rmsd = np.concatenate(all_new_rmsd)
                new_subchain = (
                    list(reversed(subchain1)) + subchain2[1:]
                )  # ascending order
                subchains.append(new_subchain)
                subchain_results.append((all_new_poses, all_new_rmsd, None))

                change = True
                subchains.pop(s2)
                subchains.pop(s1)
                subchain_results.pop(s2)
                subchain_results.pop(s1)
                break
        if change:
            break
    if not change:
        break


print("Final subchains")
print(subchains)
print()

if len(subchains) != 1:
    print("Fragment directories cannot be gathered into a single continuous chain")
    exit(1)

chain = subchains[0]
chain_result = subchain_results[0]
first_frag, last_frag = min(fragment_dirs0.keys()), max(fragment_dirs0.keys())
poses, rmsd = chain_result[:2]
if chain == list(reversed(range(first_frag, last_frag + 1))):
    chain = list(reversed(chain))
    poses = poses[:, ::-1]
    rmsd = rmsd[:, ::-1]

assert chain == list(range(first_frag, last_frag + 1)), (
    chain,
    first_frag,
    last_frag,
)

os.mkdir(output_directory)
json.dump(chain, open(os.path.join(output_directory, "FRAGS"), "w"))
for pos in range(last_frag - first_frag + 1):
    frag = first_frag + pos
    frag_poses = poses[:, pos]
    frag_rmsd = rmsd[:, pos]
    np.save(os.path.join(output_directory, f"frag{frag}.npy"), frag_poses)
    np.savetxt(
        os.path.join(output_directory, f"frag{frag}.rmsd"), frag_rmsd, fmt="%.3f"
    )
