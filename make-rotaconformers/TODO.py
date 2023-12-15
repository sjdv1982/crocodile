"""
Cleans up the library files in ORIGINAL
1. Centers them (using the all-atom PDB to compute the center)
2. Orients them along their principal components
3. Writes them out:
- XXX.npy. Coordinates (all-atom) as Nfrag x Natom x 3 Numpy array
- XXX-reduced.npy. Coordinates (reduced) as Nfrag x Nbeads x 3 Numpy array
- XXX-reduced-atomtypes.npy. The Nbeads atom types of the reduced fragment.
- XXX-template.pdb. The first (all-atom) PDB,
    (template PDB for visualization and atom selection)
- XXX-template-reduced.pdb. The first reduced PDB. 
    (template PDB for visualization and atom selection)
"""

from nefertiti.functions.parse_pdb import parse_pdb
from nefertiti.functions.write_pdb import write_pdb


import os
import sys
import numpy as np
from numpy.linalg import svd

motif = sys.argv[1]

os.chdir("../ORIGINAL")
list1 = f'{motif}-1.0-aa.list'
nfrag = len(open(list1).readlines())

first = f"{motif}/conf-aa-1.pdb"
firstr = f"{motif}/confr-1.pdb"
first_pdb = parse_pdb(open(first).read())
first_pdbr = parse_pdb(open(firstr).read())

coors = []
coorsr = []
fields = [field for field in first_pdb.dtype.names if field not in ("x", "y", "z")]
for n in range(nfrag):
    pdbf = f"{motif}/conf-aa-{n+1}.pdb"
    pdbrf = f"{motif}/confr-{n+1}.pdb"
    pdb = parse_pdb(open(pdbf).read())
    pdbr = parse_pdb(open(pdbrf).read())
    for field in fields:
        assert np.all(np.equal(pdb[field], first_pdb[field])), (n, field)
        assert np.all(np.equal(pdbr[field], first_pdbr[field])), (n, field)
    coor = np.stack((pdb["x"], pdb["y"], pdb["z"]),axis=1).astype(float)
    coorr = np.stack((pdbr["x"], pdbr["y"], pdbr["z"]),axis=1).astype(float)
    
    offset = -coor.mean()
    coor += offset
    coorr += offset
    
    for it in range(10000): # try to align on principal components, but numerical stability is bad...
        v, s, wt = np.linalg.svd(coor) 
        rotmat = wt.T
        if np.linalg.det(rotmat) < 0:
            rotmat[2] *= -1
        assert np.linalg.det(rotmat) > 0.99
        if rotmat[1,1] < -0.99 and rotmat[2,2] < -0.99:
            rotmat[1,1] *= -1
            rotmat[2,2] *= -1

        coor = coor.dot(rotmat)
        coorr = coorr.dot(rotmat)

        offset = -coor.mean()
        coor += offset
        coorr += offset

        if np.all(np.isclose(rotmat, np.eye(3), atol=0.01)):
            break

    assert np.all(np.isclose(rotmat, np.eye(3), atol=0.01)), (rotmat, s, n)

    coors.append(coor)
    coorsr.append(coorr)

coors = np.stack(coors)
coorsr = np.stack(coorsr)
atomtypes = first_pdbr["occupancy"].astype(int)

os.makedirs("../clean", exist_ok=True)
os.chdir("../clean")
open(f"{motif}-template.pdb", "w").write(write_pdb(first_pdb))
open(f"{motif}-template-reduced.pdb", "w").write(write_pdb(first_pdbr))
np.save(f"{motif}.npy", coors)
np.save(f"{motif}-reduced.npy", coorsr)
np.save(f"{motif}-reduced-atomtypes.npy", atomtypes)