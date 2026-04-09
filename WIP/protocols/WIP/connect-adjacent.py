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
index_file = os.path.join(curr_dir, "prop-indices.txt")
prop_indices = None
if os.path.exists(index_file):
    prop_indices = np.array(set(np.loadtxt(index_file, ndmin=1).astype(np.uint)))
    curr_poses = curr_poses[prop_indices-1]
curr_libc1 = curr_lib.coordinates[:, 0]

prev_poses = np.load(os.path.join(prev_dir, "poses-filtered.npy"))
index_file = os.path.join(prev_dir, "prop-indices.txt")
prev_prop_indices = None
if os.path.exists(index_file):
    prev_prop_indices = np.array(set(np.loadtxt(index_file, ndmin=1).astype(np.uint)))
    prev_poses = prev_poses[prev_prop_indices-1]
prev_seq = refe.get_sequence(prev_frag, fraglen=2)
prev_libf = dinucleotide_libraries[prev_seq]
prev_libf.load_rotaconformers()
print("START2")
prev_lib = prev_libf.create(pdb_code=pdb_code, nucleotide_mask=last_nucleotide_mask, with_rotaconformers=True)

prev_libc1 = prev_lib.coordinates[:, 0]

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
contacts = curr_pose_sample_tree.query_ball_tree(prev_pose_sample_tree,r=ovRMSD*2)
ncontacts = len(sum(contacts, []))

tot_ncontacts = int(ncontacts * len(curr_poses)/len(curr_poses_sample) * len(prev_poses)/len(prev_poses_sample))
if tot_ncontacts > 200000000:
    raise NotImplementedError

all_under_2x = None
any_under_1x = None
for atom in tqdm(list(range(curr_lib.coordinates.shape[1]))):
    prev_libc = prev_lib.coordinates[:, atom]
    curr_libc = curr_lib.coordinates[:, atom]
    curr_pose_coors = np.einsum("ij,ijl->il",
        curr_libc[curr_poses["conformer"]], curr_poses["rotation"]
    ) + curr_poses["offset"]

    prev_pose_coors = np.einsum("ij,ijl->il",
        prev_libc[prev_poses["conformer"]], prev_poses["rotation"]
    ) + prev_poses["offset"]

    curr_pose_tree = KDTree(curr_pose_coors)
    prev_pose_tree = KDTree(prev_pose_coors)
    def get_contacts(r):
        contacts = curr_pose_tree.query_ball_tree(prev_pose_tree,r=r)
        col1 = np.repeat(np.arange(len(curr_pose_coors)), [len(s) for s in contacts]).astype(int)
        col2 = np.concatenate(contacts).astype(int)
        return set(zip(col1, col2))
    contacts1 = get_contacts(ovRMSD)
    contacts2 = get_contacts(ovRMSD*2)
    print(len(contacts1), len(contacts2))
    if all_under_2x is None:
        all_under_2x = contacts2
    else:
        all_under_2x = all_under_2x.intersection(contacts2)
    if len(all_under_2x) < 10000000:
        # We have reduced the candidates enough
        any_under_1x = None
        break

    if any_under_1x is None:
        any_under_1x = contacts1
    else:
        any_under_1x = any_under_1x.union(contacts1)

candidates = all_under_2x
if any_under_1x is not None:
    candidates = any_under_1x.intersection(all_under_2x)
candidates = np.array(list(candidates))

poses1 = curr_poses[candidates[:, 0]]
poses2 = prev_poses[candidates[:, 1]]

coor1 = curr_lib.coordinates[poses1["conformer"]]
coor1 = np.einsum("ikj,ijl->ikl",
        coor1, poses1["rotation"]
) + poses1["offset"][:, None, :]

coor2 = prev_lib.coordinates[poses2["conformer"]]
coor2 = np.einsum("ikj,ijl->ikl",
        coor2, poses2["rotation"]
) + poses2["offset"][:, None, :]

dif = coor1 - coor2
rmsd = np.sqrt(np.einsum("ijk,ijk->i", dif, dif) / coor1.shape[1])
print(f"min RMSD: {rmsd.min():.3f}")