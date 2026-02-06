import shutil
import sys
import os
import numpy as np
import json

source_directory = sys.argv[1]
target_directory = sys.argv[2]
if os.path.exists(target_directory):
    print("Target directory exists")
    exit(1)

pose_mapping = {}
rev_pose_mapping = {}

poses = np.load(os.path.join(source_directory, "poses-filtered.npy"))
rmsd = np.loadtxt(os.path.join(source_directory, "poses-filtered.rmsd"), ndmin=1)
ene = np.loadtxt(os.path.join(source_directory, "poses-filtered.ene"), ndmin=1)

source_index_file = os.path.join(source_directory, "prop-indices.txt")
source_indices = np.loadtxt(source_index_file, ndmin=1).astype(int)

source_indices_hash_file = os.path.join(source_directory, "prop-indices-hash.txt")
if os.path.exists(source_indices_hash_file):
    source_hash_indices = np.loadtxt(source_indices_hash_file, ndmin=1).astype(int)
    nposes = len(set(source_hash_indices.tolist()))
    for ind, hind in zip(source_indices, source_hash_indices):
        pose_mapping[ind] = hind
        if hind - 1 not in rev_pose_mapping:
            rev_pose_mapping[hind - 1] = ind - 1
else:
    nposes = len(source_indices)
    for indnr, ind in enumerate(source_indices):
        pose_mapping[ind] = indnr + 1
        rev_pose_mapping[indnr] = ind - 1

rev_ind = [rev_pose_mapping[ind] for ind in range(nposes)]
new_poses = poses[rev_ind]
new_poses["rmsd"] = 0
new_rmsd = rmsd[rev_ind]
new_ene = ene[rev_ind]

os.mkdir(target_directory)

with open(os.path.join(target_directory, "pose-mapping.txt"), "w") as f:
    for k, v in pose_mapping.items():
        print(k, v, file=f)

np.savetxt(os.path.join(target_directory, "poses-filtered.rmsd"), new_rmsd, fmt="%.3f")
np.savetxt(os.path.join(target_directory, "poses-filtered.ene"), new_ene, fmt="%.6f")
np.save(os.path.join(target_directory, "poses-filtered.npy"), new_poses)

origin_file = os.path.join(source_directory, "poses-filtered-origins.txt")
if os.path.exists(origin_file):
    origins = np.loadtxt(origin_file).astype(int)
    assert len(origins) == len(poses)
    new_origins = origins[rev_ind]
    np.savetxt(
        os.path.join(target_directory, "poses-filtered-origins.txt"),
        new_origins,
        fmt="%d",
    )

shutil.copy(
    os.path.join(source_directory, "FRAG"), os.path.join(target_directory, "FRAG")
)

origin_source = None
originfile_source = os.path.join(source_directory, "ORIGIN")
if os.path.exists(originfile_source):
    shutil.copy(originfile_source, os.path.join(target_directory, "ORIGIN"))
