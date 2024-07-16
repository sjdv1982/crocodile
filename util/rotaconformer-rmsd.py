# Calculate the RMSD of rotamers with respect to a reference
# The rotamers and the reference are each centered, but not superimposed

import numpy as np
import sys

from nefertiti.functions.parse_pdb import parse_pdb

def get_coors(pdbfile):
    with open(pdbfile) as f:
        pdb = parse_pdb(f.read())
    pdb_coors = np.stack((pdb["x"], pdb["y"], pdb["z"]), axis=1)
    return pdb_coors

conformers_coordinate_file = sys.argv[1]
conformers = np.load(conformers_coordinate_file)

rotaconformers_file = sys.argv[2]
rotaconformers = np.load(rotaconformers_file)
assert rotaconformers["conformer"].min() >= 0 and rotaconformers["conformer"].max() <= len(conformers) - 1, (rotaconformers["conformer"].min(), rotaconformers["conformer"].max(), len(conformers) - 1)

bound_pdb = sys.argv[3]
bound = get_coors(bound_pdb)

assert conformers.ndim == 3 and conformers.shape[1] == len(bound) and conformers.shape[2] == 3, (conformers.shape, len(bound))

conformers -= conformers.mean(axis=1)[:, None, :]
bound -= bound.mean(axis=0)


rmsds = np.empty(len(rotaconformers))


chunklen = 1000
for n in range(0, len(rotaconformers), chunklen):
    rotamer_matrices_chunk = rotaconformers["rotmat"][n:n+chunklen]
    conformers_chunk = conformers[rotaconformers["conformer"][n:n+chunklen]]
    rotamer_chunk = np.einsum("ikj,ijl->ikl", conformers_chunk, rotamer_matrices_chunk)
    dif = rotamer_chunk - bound
    
    rmsd_chunk = np.sqrt((dif*dif).sum(axis=2).mean(axis=1))
    
    for r in rmsd_chunk:
        print(r)