import json
import os
import sys
import numpy as np
from crocodile.nuc.reference import Reference
from library_config import (
    mononucleotide_templates,
    dinucleotide_templates,
    dinucleotide_libraries,
)
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

model_template = []
model_coors = []

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

    poses_coors_conf = lib.coordinates[poses["conformer"]]
    poses_coors = (
        np.einsum("ikj,ijl->ikl", poses_coors_conf, poses["rotation"])
        + poses["offset"][:, None, :]
    )
    model_coors.append(poses_coors)

    tmpl = dinucleotide_templates[seq].copy()
    mask = tmpl["resid"] == tmpl["resid"][0]
    tmpl["resid"][mask] = 100 * frag + 1
    tmpl["resid"][~mask] = 100 * frag + 2
    model_template.append(tmpl)

if nchains is None:
    exit(0)

model_template = np.concatenate(model_template)
model_coors = np.concatenate(model_coors, axis=1)

model = np.tile(model_template.astype(atomic_dtype), nchains).reshape(nchains, -1)

model["x"] = model_coors[:, :, 0]
model["y"] = model_coors[:, :, 1]
model["z"] = model_coors[:, :, 2]

np.save(os.path.join(target_dir, "coor-chains-ppdb.npy"), model)
pdbtxt = write_multi_pdb(model)
with open(os.path.join(target_dir, "coor-chains.pdb"), "w") as f:
    f.write(pdbtxt)
