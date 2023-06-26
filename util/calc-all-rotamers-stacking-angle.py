
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
    "U": ["N1", "C2", "N3", "C4", "C5", "C6"],
    "RU": ["N1", "C2", "N3", "C4", "C5", "C6"],
    "C": ["N1", "C2", "N3", "C4", "C5", "C6"],
    "RC": ["N1", "C2", "N3", "C4", "C5", "C6"],
    "G": ["N1", "C2", "N3", "C4", "C5", "C6", "N7", "C8", "N9"],
    "RG": ["N1", "C2", "N3", "C4", "C5", "C6", "N7", "C8", "N9"],
    
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

ligand_template = load_file(sys.argv[4])
ligand_template_chain = sys.argv[5]
ligand_template_resid = int(sys.argv[6])
ligand_conformers_file = sys.argv[7]
ligand_conformers = np.load(ligand_conformers_file)
assert ligand_conformers.ndim == 3 and ligand_conformers.shape[1] == len(ligand_template) and ligand_conformers.shape[2] == 3, ligand_conformers.shape

ligand_rotaconformers_file = sys.argv[8]
ligand_rotaconformers = np.load(ligand_rotaconformers_file)
assert ligand_rotaconformers["conformer"].min() == 0 and ligand_rotaconformers["conformer"].max() == len(ligand_conformers) - 1, (ligand_rotaconformers["conformer"].min(), ligand_rotaconformers["conformer"].max(), len(ligand_conformers) - 1)

outputfile = sys.argv[9]

def select_ring(mol, chain, resid):
    if chain.upper() != "ALL":
        mask0 = (mol["chain"] == chain.encode())
        assert mask0.sum()
    else:
        mask0 = np.ones(len(mol), bool)

    mask1 = (mol["resid"] == resid)
    assert mask1.sum()

    mol0 = mol[mask0 & mask1]
    mol_resname = mol0[0]["resname"].decode()
    mol_ring = rings[mol_resname]
    mask2 = np.isin(mol["name"], [a.encode() for a in mol_ring])
    assert mask2.sum()
    mask = mask0 & mask1 & mask2
    mol = mol[mask]
    assert len(mol)
    mol_coor = np.stack((mol["x"], mol["y"], mol["z"]), axis=1)
    return mol_coor, mask

def calc_plane(coor):
    coor = coor - coor.mean(axis=0)
    covar = coor.T.dot(coor)
    v, s, wt = svd(covar)
    return wt[2] / norm(wt[2])

def calc_planes(coor):
    coor = coor - coor.mean(axis=1)[:, None]
    covar = np.einsum("ijk,ijl->ikl", coor, coor) #coor[i].T.dot(coor[i])
    v, s, wt = svd(covar)
    return wt[:, 2, :] / norm(wt[:, 2, :], axis=-1)[..., None]

prot_coor, _ = select_ring(prot, prot_chain, prot_resid)
_, nuc_coor_mask = select_ring(ligand_template, ligand_template_chain, ligand_template_resid)
nuc_coor = ligand_conformers[:, nuc_coor_mask]

prot_plane = calc_plane(prot_coor)

angle = np.zeros(len(ligand_rotaconformers)).astype(np.float32)
CHUNKSIZE = 100000
for pos in range(0, len(ligand_rotaconformers), CHUNKSIZE):
    if pos % (10 * CHUNKSIZE) == 0:
        print(f"{pos}/{len(ligand_rotaconformers)}", file=sys.stderr)
    chunk = ligand_rotaconformers[pos:pos+CHUNKSIZE]
    
    all_nuc_coor = nuc_coor[chunk["conformer"]]
    nuc_coor_rotamers = np.einsum('kij,kjl->kil', all_nuc_coor, chunk["mat"]) #broadcasted all_nuc_coor[k].dot(chunk["mat"][k])
    nuc_plane_rotamers = calc_planes(nuc_coor_rotamers)
    nrm = norm(np.cross(prot_plane, nuc_plane_rotamers), axis=1)
    nrm = np.minimum(nrm, 1)
    chunk_angle = np.arcsin(nrm)
    angle[pos:pos+CHUNKSIZE] = chunk_angle
np.save(outputfile, angle)
