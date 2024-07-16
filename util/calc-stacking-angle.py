
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

RINGS = {
    #nucleic acids
    "T": ["C6", "C5", "C7", "N3", "O2", "O4"],
    "U": ["N1", "C2", "N3", "C4", "C5", "C6"],
    "C": ["N1", "C2", "N3", "C4", "C5", "C6"],
    "G": ["N1", "C2", "N3", "C4", "C5", "C6", "N7", "C8", "N9"],
    "A": ["N1", "C2", "N3", "C4", "C5", "C6", "N7", "C8", "N9"], 
    
    # proteins
    "PHE": ["CG", "CD1", "CD2" , "CE1", "CE2", "CZ"],
    "TYR": ["CG", "CD1", "CD2" , "CE1", "CE2", "CZ"],
    "HIS": ["CG", "ND1", "CD2" , "CE1", "NE2"],
    "ARG": ["CD", "NE", "CZ" , "NH1", "NH2"],
    "TRP": ["CG", "CD1", "CD2" , "NE1", "CE2", "CE3", "CZ2", "CZ3", "CH2"],
}
for p1, p2 in (
    ("DT", "T"),
    ("RU", "U"),
    ("DC", "C"),
    ("RC", "C"),
    ("DG", "G"),
    ("RG", "G"),
    ("DA", "A"),
    ("RA", "A"),
):
    RINGS[p1] = RINGS[p2]

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
    mol_ring = RINGS[mol_resname]
    mask = np.isin(mol["name"], [a.encode() for a in mol_ring])
    mol = mol[mask]
    assert len(mol)
    mol_coor = np.stack((mol["x"], mol["y"], mol["z"]), axis=1)
    return mol_coor

def calc_plane(coor):
    coor = coor - coor.mean(axis=0)
    covar = coor.T.dot(coor)
    v, s, wt = svd(covar)
    plane_directionless = wt[2] / norm(wt[2])
    plane0_directed = np.cross((coor[1] - coor[0]), (coor[2] - coor[1]))
    flip = np.sign(np.dot(plane_directionless, plane0_directed))
    return plane_directionless * flip


prot_coor = select_ring(prot, prot_chain, prot_resid)
nuc_coor = select_ring(nuc, nuc_chain, nuc_resid)

prot_plane = calc_plane(prot_coor)
nuc_plane = calc_plane(nuc_coor)

nrm = norm(np.cross(prot_plane, nuc_plane))
if nrm > 1: 
    nrm = 1
angle = np.arcsin(nrm)
print("Plane angle: {:.3f} radians, {:.1f} degrees".format(angle, angle/np.pi*180) )
prot_center = prot_coor.mean(axis=0)
nuc_center = nuc_coor.mean(axis=0)
prot_x_vector = prot_coor[0] - prot_center
prot_x_vector /= norm(prot_x_vector)
prot_y_vector = np.cross(prot_plane, prot_x_vector)
prot_y_vector /= norm(prot_y_vector)
###assert abs(prot_x_vector.dot(prot_plane)) < 0.01, abs(prot_x_vector.dot(prot_plane)) # Not so for dual rings
nuc_x_vector = nuc_coor[0] - nuc_center
nuc_x_vector -= nuc_x_vector.dot(prot_plane) * prot_plane
if norm(nuc_x_vector) < 0.0001:
    dihed = np.pi / 2
else:
    nuc_x_vector /= norm(nuc_x_vector)
    assert abs(nuc_x_vector.dot(prot_plane)) < 0.01
    dihed_x = nuc_x_vector.dot(prot_x_vector)
    dihed_y = nuc_x_vector.dot(prot_y_vector)
    dihed = np.angle(dihed_x + 1.0j * dihed_y)

print("Dihedral angle: {:.3f} radians, {:.1f} degrees".format(dihed, dihed/np.pi*180) )
