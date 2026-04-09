import json
import os
import sys
import numpy as np
from crocodile.nuc.reference import Reference
from library_config import mononucleotide_templates, dinucleotide_libraries
from nefertiti.functions.write_pdb import write_multi_pdb
from nefertiti.functions.parse_pdb import atomic_dtype

target_dir = sys.argv[1]
frags = json.load(open(os.path.join(target_dir, "FRAGS")))


def err(*args):
    print(*args, file=sys.stderr)
    exit(1)


if not os.path.exists("constraints.json"):
    err("constraints.json must exist")

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

avg_coors = {}

nchains = None
for frag in frags:
    seq = refe.get_sequence(frag, 2)
    libf = dinucleotide_libraries[seq]
    lib = libf.create(pdb_code)

    poses_file = os.path.join(target_dir, f"frag{frag}.npy")
    poses = np.load(poses_file)
    if nchains is None:
        nchains = len(poses)
    else:
        assert len(poses) == nchains, frag

    rmsds_file = os.path.join(target_dir, f"frag{frag}.rmsd")
    original_rmsds = np.loadtxt(rmsds_file)
    assert len(original_rmsds) == nchains, frag

    poses_coors_conf = lib.coordinates[poses["conformer"]]
    poses_coors = (
        np.einsum("ikj,ijl->ikl", poses_coors_conf, poses["rotation"])
        + poses["offset"][:, None, :]
    )
    refe_coor = refe.get_coordinates(frag, 2)
    rmsds = np.sqrt(
        np.einsum("ijk->i", (poses_coors - refe_coor) ** 2) / len(refe_coor)
    )
    dif = rmsds - original_rmsds
    wrong = np.where(np.abs(dif) > 0.1)[0]
    for pose in wrong:
        print("WRONG", frag, pose + 1, rmsds[pose], original_rmsds[pose])
