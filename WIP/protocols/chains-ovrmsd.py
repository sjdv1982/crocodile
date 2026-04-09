import json
import os
import numpy as np
from crocodile.nuc.reference import Reference
import sys
from library_config import (
    mononucleotide_templates,
    dinucleotide_templates,
    dinucleotide_libraries,
)

chaindir = sys.argv[1]

if not os.path.exists("constraints.json"):
    print("constraints.json must exist")
    exit(0)

with open("constraints.json") as f:
    constraints = json.load(f)
pdb_code = constraints["pdb_code"]

refe_ppdb = np.load("reference.npy")
refe = Reference(
    ppdb=refe_ppdb,
    mononucleotide_templates=mononucleotide_templates,
    rna=True,
    ignore_unknown=False,
    ignore_missing=False,
    ignore_reordered=False,
)

all_pre_coors = {}
all_post_coors = {}
frags = []

for fragpos in refe.get_fragment_positions(2):
    poses_file = f"{chaindir}/frag{fragpos}.npy"
    if not os.path.exists(poses_file):
        continue
    seq = refe.get_sequence(fragpos, 2)
    libf = dinucleotide_libraries[seq]
    pre_lib = libf.create(pdb_code=pdb_code, nucleotide_mask=[False, True])
    post_lib = libf.create(pdb_code=pdb_code, nucleotide_mask=[True, False])
    frags.append(fragpos)
    print(fragpos)
    for lib, all_coors in ((pre_lib, all_pre_coors), (post_lib, all_post_coors)):
        poses = np.load(poses_file)
        pose_coors = (
            np.einsum(
                "ikj,ijl->ikl", lib.coordinates[poses["conformer"]], poses["rotation"]
            )
            + poses["offset"][:, None, :]
        )
        all_coors[fragpos] = pose_coors

for f1 in frags:
    f2 = f1 + 1
    if f2 not in frags:
        continue
    coors1, coors2 = all_pre_coors[f1], all_post_coors[f2]
    d = coors1 - coors2
    ovrmsd = np.sqrt((d * d).sum(axis=2).mean(axis=1))
    np.savetxt(f"{chaindir}/pair-{f1}-{f2}.ovrmsd", ovrmsd, fmt="%.3f")
