
import os, sys
import numpy as np
from nefertiti.functions.parse_pdb import parse_pdb
from nefertiti.functions.parse_mmcif import parse_mmcif
from numpy.linalg import svd, norm


def load_file(f):
    data = open(f).read()
    if f.endswith(".pdb"):
        return parse_pdb(data)
    elif f.endswith(".cif"):
        return parse_mmcif(data)
    else:
        raise ValueError("Unknown file extension: {}".format(f))

# TODO: for other nucleic acids...
rings = {
    #nucleic acids
    "DT": ["C6", "C5", "C7", "N3", "O2", "O4"],
    
    # proteins
    "PHE": ["CB", "CG", "CD1", "CD2" , "CE1", "CE2", "CZ"],
    "TYR": ["CB", "CG", "CD1", "CD2" , "CE1", "CE2", "CZ"],
    "HIS": ["CB", "CG", "ND1", "CD2" , "CE1", "NE2"],
    "ARG": ["CD", "NE", "CZ" , "NH1", "NH2"],
    "TRP": ["CB", "CG", "CD1", "CD2" , "NE1", "CE2", "CE3", "CZ2", "CZ3", "CH2"],
}

prot = load_file(sys.argv[1])
prot_chain = sys.argv[2]
prot_resid = int(sys.argv[3])

nuc = load_file(sys.argv[4])
nuc_chain = sys.argv[5]
nuc_resid = int(sys.argv[6])

def select_ring(mol, chain, resid):
    if chain.upper() != "ALL":
        mask = (mol["chain"] == chain.encode())
        mol = mol[mask]
        assert len(mol)
    mask = (mol["resid"] == resid)
    mol = mol[mask]
    assert len(mol)

    mol_resname = mol[0]["resname"].decode()
    mol_ring = rings[mol_resname]
    mask = np.isin(mol["name"], [a.encode() for a in mol_ring])
    mol = mol[mask]
    assert len(mol)
    mol_coor = np.stack((mol["x"], mol["y"], mol["z"]), axis=1)
    return mol_coor

def calc_plane(coor):
    coor = coor - coor.mean(axis=0)
    covar = coor.T.dot(coor)
    v, s, wt = svd(covar)
    return wt[2] / norm(wt[2])

prot_coor = select_ring(prot, prot_chain, prot_resid)
nuc_coor = select_ring(nuc, nuc_chain, nuc_resid)

prot_plane = calc_plane(prot_coor)
nuc_plane = calc_plane(nuc_coor)

nrm = norm(np.cross(prot_plane, nuc_plane_rotamer))
if nrm > 1: 
    nrm = 1
angle = np.arcsin(nrm)
print(angle)
