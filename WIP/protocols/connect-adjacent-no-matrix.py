import os
import argparse
import json
import sys
import numpy as np
import random
from scipy.spatial import KDTree
from tqdm import tqdm
from crocodile.nuc.all_fit import (
    conformer_mask_from_crmsd,
    conformer_mask_from_general_pairing,
    conformer_masks_from_specific_pairing,
) # TODO
from crocodile.nuc.reference import Reference
from library_config import mononucleotide_templates, dinucleotide_libraries

parser = argparse.ArgumentParser()
parser.add_argument("fragment", type=int)
parser.add_argument("target_directory")
parser.add_argument("origin_directory1")
parser.add_argument("origin_directory2")

args = parser.parse_args()


def err(*args):
    print(*args, file=sys.stderr)
    exit(1)


if not os.path.exists("constraints.json"):
    err("constraints.json must exist")

poses_file = os.path.join(args.target_directory, "poses.npy")
if os.path.exists(args.target_directory):
    if os.path.exists(poses_file):
        err(f"poses.npy in '{args.target_directory}' already exists")
else:
    os.makedirs(args.target_directory)
    os.link("score-data", os.path.join(args.target_directory, "score-data"))
    os.link("templates", os.path.join(args.target_directory, "templates"))

with open("constraints.json") as f:
    constraints = json.load(f)
pdb_code = constraints["pdb_code"]

fragment = args.fragment
fragmentstr = f"frag{fragment}"
frag_constr = constraints[fragmentstr]

fragfile = os.path.join(args.origin_directory1, "FRAG")
ori1_frag = int(json.load(open(fragfile)))
if ori1_frag != fragment + 1 and ori1_frag != fragment - 1 and ori1_frag != fragment:
    exit("Origin directories must be from the same or an adjacent fragment")

fragfile = os.path.join(args.origin_directory2, "FRAG")
ori2_frag = int(json.load(open(fragfile)))
if ori2_frag != fragment + 1 and ori2_frag != fragment - 1 and ori2_frag != fragment:
    exit("Origin directories must be from the same or an adjacent fragment")

if ori1_frag == ori2_frag:
    exit("Origin directories must be of different fragments")

if ori1_frag != fragment and ori2_frag != fragment:
    exit("One origin directory must be of the same fragment as the first argument")

if ori1_frag == fragment:
    curr_dir = args.origin_directory1
    prev_frag = ori2_frag
    prev_dir = args.origin_directory2
else:
    curr_dir = args.origin_directory2
    prev_frag = ori1_frag
    prev_dir = args.origin_directory1

prev_fragstr = f"frag{prev_frag}"

if prev_frag < fragment:
    frag0, frag1 = prev_fragstr, fragmentstr
else:
    frag0, frag1 = fragmentstr, prev_fragstr
for p in constraints["pairs"]:
    if p["down"] == frag0 and p["up"] == frag1:
        ovRMSD = p["ovRMSD"]
        cRMSD = p["cRMSD"]
        constraint_pair = p
        break
else:
    exit("Cannot find constraint fragment pair")

refe_ppdb = np.load("reference.npy")
refe = Reference(
    ppdb=refe_ppdb,
    mononucleotide_templates=mononucleotide_templates,
    rna=True,
    ignore_unknown=False,
    ignore_missing=False,
    ignore_reordered=False,
)

if prev_frag == fragment - 1:
    nucleotide_mask = [True, False]
    last_nucleotide_mask = [False, True]
else:
    nucleotide_mask = [False, True]
    last_nucleotide_mask = [True, False]

seq = refe.get_sequence(fragment, fraglen=2)
assert seq == frag_constr["sequence"]
libf = dinucleotide_libraries[seq]
libf.load_rotaconformers()
print("START")

curr_lib = libf.create(
    pdb_code=pdb_code,
    nucleotide_mask=nucleotide_mask,
    only_base=False,
    with_rotaconformers=True,
)

curr_poses = np.load(os.path.join(curr_dir, "poses-filtered.npy"))
print(f"Current poses: {len(curr_poses)}")
index_file = os.path.join(curr_dir, "prop-indices.txt")
prop_indices = None
if os.path.exists(index_file):
    prop_indices = np.array(set(np.loadtxt(index_file, ndmin=1).astype(np.uint)))
    curr_poses = curr_poses[prop_indices-1]
curr_libc1 = curr_lib.coordinates[:, 0]

prev_poses = np.load(os.path.join(prev_dir, "poses-filtered.npy"))
print(f"Adjacent poses: {len(prev_poses)}")
index_file = os.path.join(prev_dir, "prop-indices.txt")
prev_prop_indices = None
if os.path.exists(index_file):
    prev_prop_indices = np.array(list(set(np.loadtxt(index_file, ndmin=1).astype(np.uint))))
    prev_poses = prev_poses[prev_prop_indices-1]
prev_seq = refe.get_sequence(prev_frag, fraglen=2)
prev_libf = dinucleotide_libraries[prev_seq]
prev_libf.load_rotaconformers()
print("START2")
prev_lib = prev_libf.create(pdb_code=pdb_code, nucleotide_mask=last_nucleotide_mask, with_rotaconformers=True)

prev_libc1 = prev_lib.coordinates[:, 0]

'''
# TODO trinuc pairing
if prev_frag < fragment:
    trinuc_seq = prev_seq[:-1] + seq
    pair_mask = np.zeros((len(prev_lib.coordinates), len(curr_lib.coordinates)), bool)    
else:
    trinuc_seq = seq + prev_seq[1:]
    pair_mask = np.zeros((len(curr_lib.coordinates), len(prev_lib.coordinates)), bool)    
trinuc_seq = trinuc_seq.replace("G", "A").replace("U", "C")
trinuc_pairing = json.load(
    open(f"dinuc-trinuc-pairs/{trinuc_seq}-pairing.txt")
)
for pairing in trinuc_pairing:
    for (p1, p2), crmsd in pairing:
        if crmsd < cRMSD:
            pair_mask[p1, p2] = 1

if prev_frag < fragment:
    pair_mask = pair_mask.T
/# TODO trinuc pairing
'''

random.seed(0)
prev_poses_sample = prev_poses[random.sample(range(len(prev_poses)), min(len(prev_poses), 10000))]
curr_poses_sample = curr_poses[random.sample(range(len(curr_poses)), min(int(10**8/len(prev_poses_sample)), len(curr_poses)))]

curr_pose_sample_coors = np.einsum("ij,ijl->il",
    curr_libc1[curr_poses_sample["conformer"]], curr_poses_sample["rotation"]
) + curr_poses_sample["offset"]

prev_pose_sample_coors = np.einsum("ij,ijl->il",
    prev_libc1[prev_poses_sample["conformer"]], prev_poses_sample["rotation"]
) + prev_poses_sample["offset"]

curr_pose_sample_tree = KDTree(curr_pose_sample_coors)
prev_pose_sample_tree = KDTree(prev_pose_sample_coors)
MARGIN = min(ovRMSD, 0.5)
contacts = curr_pose_sample_tree.query_ball_tree(prev_pose_sample_tree,r=ovRMSD+MARGIN)
ncontacts = len(sum(contacts, []))

tot_ncontacts = int(ncontacts * len(curr_poses)/len(curr_poses_sample) * len(prev_poses)/len(prev_poses_sample))
print(f"Estimated number of initial contact candidates: {tot_ncontacts}")

nchunks = int(tot_ncontacts / 1000000)
print(f"Current pose chunks: {nchunks}")
assert nchunks < len(curr_poses)
positions = (np.linspace(0, 1, nchunks+1) * len(curr_poses)).astype(int)

new_pose_inds = []
new_pose_origins = []
for chunk in tqdm(list(range(nchunks))):
    curr_curr_poses = curr_poses[positions[chunk]:positions[chunk+1]]
    atom = 0  # first atom

    prev_libc = prev_lib.coordinates[:, atom]
    curr_libc = curr_lib.coordinates[:, atom]
    curr_pose_coors = np.einsum("ij,ijl->il",
        curr_libc[curr_curr_poses["conformer"]], curr_curr_poses["rotation"]
    ) + curr_curr_poses["offset"]

    prev_pose_coors = np.einsum("ij,ijl->il",
        prev_libc[prev_poses["conformer"]], prev_poses["rotation"]
    ) + prev_poses["offset"]

    curr_pose_tree = KDTree(curr_pose_coors)
    prev_pose_tree = KDTree(prev_pose_coors)

    contacts = curr_pose_tree.query_ball_tree(prev_pose_tree,r=ovRMSD+MARGIN)
    col1 = np.repeat(np.arange(len(curr_pose_coors)), [len(s) for s in contacts]).astype(int)
    col2 = np.concatenate(contacts).astype(int)
    candidates = np.stack((col1, col2),axis=1)
    print(f"Chunk {chunk+1}: {len(candidates)} initial contact candidates")
    if len(candidates) == 0:
        continue

    poses1 = curr_curr_poses
    poses2 = prev_poses

    coor1 = curr_lib.coordinates[poses1["conformer"]]
    coor1 = np.einsum("ikj,ijl->ikl",
            coor1, poses1["rotation"]
    ) + poses1["offset"][:, None, :]

    print("X 1")
    coor2 = prev_lib.coordinates[poses2["conformer"]]
    coor2 = np.einsum("ikj,ijl->ikl",
            coor2, poses2["rotation"]
    ) + poses2["offset"][:, None, :]
    print("X 2")


    dif = coor1[candidates[:, 0]] - coor2[candidates[:, 1]]
    rmsd = np.sqrt(np.einsum("ijk,ijk->i", dif, dif) / coor1.shape[1])
    print("X 3")
    mask = (rmsd < ovRMSD)
    print(f"Chunk {chunk+1}: {mask.sum()} final contact candidates")
    print(f"min RMSD: {rmsd.min():.3f}")
    if mask.sum() > 0:
        curr_pose_inds = candidates[mask, 0] + positions[chunk]        
        curr_pose_origins = candidates[mask, 1]
        new_pose_inds.append(curr_pose_inds)
        new_pose_origins.append(curr_pose_origins)

new_pose_inds = np.concatenate(new_pose_inds)
new_pose_origins = np.concatenate(new_pose_origins)
new_poses = curr_poses[new_pose_inds]
np.save(poses_file, new_poses)
np.savetxt(os.path.join(args.target_directory, "poses-merge-origins.txt"), new_pose_inds+1, fmt="%d")
np.savetxt(os.path.join(args.target_directory, "poses-origins.txt"), new_pose_origins+1, fmt="%d")
