import sys
import os
import numpy as np
import json

source_directory = sys.argv[1]
target_directory = sys.argv[2]

frag_source = int(json.load(open(os.path.join(source_directory, "FRAG"))))
frag_target = int(json.load(open(os.path.join(target_directory, "FRAG"))))
assert abs(frag_source - frag_target) == 1

origin_source = None
originfile_source = os.path.join(source_directory, "ORIGIN")
if os.path.exists(originfile_source):
    origin_source = int(json.load(open(originfile_source)))

origin_target = None
originfile_target = os.path.join(target_directory, "ORIGIN")
if os.path.exists(originfile_target):
    origin_target = int(json.load(open(originfile_target)))

if origin_source is not None and origin_source == frag_target:
    # backpropagation: the target dir is the origin of the source dir
    # remove those poses in the target dir that didn't propagate into poses in the source dir
    all_origins = np.loadtxt(
        os.path.join(source_directory, "poses-filtered-origins.txt"), ndmin=1
    ).astype(np.uint)

    source_index_file = os.path.join(source_directory, "prop-indices.txt")
    if os.path.exists(source_index_file):
        source_indices = np.loadtxt(source_index_file, ndmin=1).astype(int)
        all_origins = all_origins[source_indices - 1]

    source_origins = np.unique(all_origins)

    pose_mapping_file = os.path.join(target_directory, "pose-mapping.txt")
    if os.path.exists(pose_mapping_file):
        pose_mapping = {}
        for l in open(pose_mapping_file).readlines():
            ll = l.split()
            pose_mapping[int(ll[0])] = int(ll[1])
        source_origins = np.array([pose_mapping[ind] for ind in source_origins])

    nposes_target = len(np.load(os.path.join(target_directory, "poses-filtered.npy")))
    index_file = os.path.join(target_directory, "prop-indices.txt")
    if os.path.exists(index_file):
        curr_indices = np.loadtxt(index_file, ndmin=1).astype(np.uint)
        print(f"Current prop indices: {len(curr_indices)}")
        new_indices = list(set(curr_indices).intersection(source_origins))
    else:
        new_indices = source_origins
    print(f"New prop indices: {len(new_indices)}/{nposes_target}")
    np.savetxt(index_file, new_indices, fmt="%d")
elif origin_target is not None and origin_target == frag_source:
    # forward propagation: the source dir is the origin of the target dir
    # remove those poses in the target dir whose origin poses have been filtered out
    #  (which is reflected by the prop indices in the origin dir)
    nposes_target = len(np.load(os.path.join(target_directory, "poses-filtered.npy")))
    source_index_file = os.path.join(source_directory, "prop-indices.txt")
    if not os.path.exists(source_index_file):
        print("Prop indices are lacking. No propagation to be done")
        exit(1)
    source_indices = set(
        np.loadtxt(source_index_file, ndmin=1).astype(np.uint).tolist()
    )
    origins = np.loadtxt(
        os.path.join(target_directory, "poses-filtered-origins.txt"), ndmin=1
    ).astype(np.uint)
    origin_indices = [onr + 1 for onr, o in enumerate(origins) if o in source_indices]
    index_file = os.path.join(target_directory, "prop-indices.txt")
    if os.path.exists(index_file):
        curr_indices = np.loadtxt(index_file, ndmin=1).astype(np.uint)
        print(f"Current prop indices: {len(curr_indices)}")

        new_indices = list(set(curr_indices).intersection(origin_indices))
    else:
        new_indices = origin_indices

    print(f"New prop indices: {len(new_indices)}/{nposes_target}")
    np.savetxt(index_file, new_indices, fmt="%d")

else:
    print("Propagation is ineffective")
    exit(1)
