"""
Calculate RMSD for the poses; explicit definition of poses file and fragment
"""

import numpy as np
import os
import argparse
import json
import sys
from crocodile.nuc.reference import Reference
from library_config import mononucleotide_templates, dinucleotide_libraries

parser = argparse.ArgumentParser()
parser.add_argument("pose_file")
parser.add_argument("fragment", type=int)
parser.add_argument("output_file")

args = parser.parse_args()


def err(*args):
    print(*args, file=sys.stderr)
    exit(0)


if not os.path.exists("constraints.json"):
    err("constraints.json must exist")

with open("constraints.json") as f:
    constraints = json.load(f)
pdb_code = constraints["pdb_code"]

poses_file = args.pose_file
fragment = args.fragment

refe_ppdb = np.load("reference.npy")
refe = Reference(
    ppdb=refe_ppdb,
    mononucleotide_templates=mononucleotide_templates,
    rna=True,
    ignore_unknown=False,
    ignore_missing=False,
    ignore_reordered=False,
)

seq = refe.get_sequence(fragment, fraglen=2)
libf = dinucleotide_libraries[seq]

poses = np.load(poses_file)

lib2 = libf.create(pdb_code=pdb_code)
refe_coors = refe.get_coordinates(fragment, fraglen=2)

CHUNKSIZE = 100000
rmsd = np.empty(len(poses), np.float32)
for chunk in range(0, len(poses), CHUNKSIZE):
    print(chunk)
    curr_conf = lib2.coordinates[poses["conformer"][chunk : chunk + CHUNKSIZE]]
    curr_poses = poses[chunk : chunk + CHUNKSIZE]
    poses_coors = (
        np.einsum("ikj,ijl->ikl", curr_conf, curr_poses["rotation"])
        + curr_poses["offset"][:, None, :]
    )
    dif = poses_coors - refe_coors
    curr_rmsd = np.sqrt(np.einsum("ijk,ijk->i", dif, dif) / len(refe_coors))
    rmsd[chunk : chunk + CHUNKSIZE] = curr_rmsd
print(len(poses), rmsd.min())

np.savetxt(args.output_file, rmsd, fmt="%.3f")
