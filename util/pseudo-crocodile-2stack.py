import os, sys
import numpy as np
from numpy.linalg import norm, svd
from tqdm import tqdm
from nefertiti.functions.parse_pdb import parse_pdb
from nefertiti.functions.parse_mmcif import parse_mmcif
from math import sqrt, pi

MARGIN = 0.5

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
    mol_ring = RINGS[mol_resname]
    mask2 = np.isin(mol["name"], [a.encode() for a in mol_ring])
    assert mask2.sum()
    mask = mask0 & mask1 & mask2
    mol = mol[mask]
    assert len(mol)
    return mol, mask

def calc_plane(coor):
    coor = coor - coor.mean(axis=0)
    covar = coor.T.dot(coor)
    v, s, wt = svd(covar)
    plane_directionless = wt[2] / norm(wt[2])
    plane0_directed = np.cross((coor[1] - coor[0]), (coor[2] - coor[1]))
    flip = np.sign(np.dot(plane_directionless, plane0_directed))
    return plane_directionless * flip

def get_coor(mol):
    mol_coor = np.stack((mol["x"], mol["y"], mol["z"]), axis=-1)
    return mol_coor

prot = load_file(sys.argv[1])
prot_chain = sys.argv[2]
prot_resid1 = int(sys.argv[3])
prot_resid2 = int(sys.argv[4])

ligand_template = load_file(sys.argv[5])
ligand_template_chain = sys.argv[6]
ligand_template_resid1 = int(sys.argv[7])
ligand_template_resid2 = int(sys.argv[8])
ligand_conformers_file = sys.argv[9]
ligand_conformers = np.load(ligand_conformers_file)
ligand_conformers = ligand_conformers - ligand_conformers.mean(axis=1)[:, None, :]
assert ligand_conformers.ndim == 3 and ligand_conformers.shape[1] == len(ligand_template) and ligand_conformers.shape[2] == 3, ligand_conformers.shape

ligand_rotaconformers_file = sys.argv[10]
ligand_rotaconformers = np.load(ligand_rotaconformers_file)
assert ligand_rotaconformers["conformer"].min() >= 0 and ligand_rotaconformers["conformer"].max() <= len(ligand_conformers) - 1, (ligand_rotaconformers["conformer"].min(), ligand_rotaconformers["conformer"].max(), len(ligand_conformers) - 1)

outputfile = sys.argv[11]
offset_outputfile = sys.argv[12]

prot_coor1 = get_coor(select_ring(prot, prot_chain, prot_resid1)[0])
prot_center1 = prot_coor1.mean(axis=0)
prot_plane1 = calc_plane(prot_coor1)
prot_coor2 = get_coor(select_ring(prot, prot_chain, prot_resid2)[0])
prot_center2 = prot_coor2.mean(axis=0)
prot_plane2 = calc_plane(prot_coor2)
_, nuc_coor_mask1 = select_ring(ligand_template, ligand_template_chain, ligand_template_resid1)
nuc_coor1 = ligand_conformers[:, nuc_coor_mask1]
nuc_center1 = nuc_coor1.mean(axis=1)

_, nuc_coor_mask2 = select_ring(ligand_template, ligand_template_chain, ligand_template_resid2)
nuc_coor2 = ligand_conformers[:, nuc_coor_mask2]
nuc_center2 = nuc_coor2.mean(axis=1)

all_nuc_center1 = nuc_center1[ligand_rotaconformers["conformer"]]
roco_nuc_center1 = np.einsum('kj,kjl->kl', all_nuc_center1, ligand_rotaconformers["rotmat"]) #broadcasted all_nuc_coor1[k].dot(chunk["rotmat"][k])
del all_nuc_center1

all_nuc_center2 = nuc_center2[ligand_rotaconformers["conformer"]]
roco_nuc_center2 = np.einsum('kj,kjl->kl', all_nuc_center2, ligand_rotaconformers["rotmat"]) #broadcasted all_nuc_coor1[k].dot(chunk["rotmat"][k])
del all_nuc_center2

gridspacing = sqrt(3)/3

roco_offset_bb1 = np.min(prot_center1-roco_nuc_center1, axis=0), np.max(prot_center1-roco_nuc_center1, axis=0)
roco_offset_bb2 = np.min(prot_center2-roco_nuc_center2, axis=0), np.max(prot_center2-roco_nuc_center2, axis=0)
print(roco_offset_bb1)
print(roco_offset_bb2)
roco_offset_bb = np.max(np.stack((roco_offset_bb1[0], roco_offset_bb2[0])),axis=0), np.min(np.stack((roco_offset_bb1[1], roco_offset_bb2[1])),axis=0)
print(roco_offset_bb)
box_origin = np.floor((roco_offset_bb[0] - 5)/gridspacing).astype(int)
box_upper = np.ceil((roco_offset_bb[1] + 5)/gridspacing).astype(int)
box_dim = box_upper - box_origin + 1
print(box_origin, box_upper, box_dim)
box = np.stack(np.meshgrid(
    np.arange(box_origin[0], box_upper[0] + 1),
    np.arange(box_origin[1], box_upper[1] + 1),
    np.arange(box_origin[2], box_upper[2] + 1),
),axis=-1) * gridspacing
print(box[0,0,0], roco_offset_bb[0] - 5 - MARGIN, gridspacing)
print(box[-1,-1,-1], roco_offset_bb[1] + 5 + MARGIN, gridspacing)

lateral_slope = 1.78966

chunksize = 20
offsets = box.reshape(-1, 3)

print(len(offsets))
def analyze_chunk(n):
    offsets_chunk = offsets[n:n+chunksize]
    
    chunk_center1 = roco_nuc_center1[None, :, :] + offsets_chunk[:, None, :]
    center_vec1 = chunk_center1 - prot_center1
    center_z1 = np.abs(center_vec1[:, :, 2])
    center_dis1 = norm(center_vec1, axis=2)
    if center_dis1.min() > 5.0 + MARGIN:
        return []
    axial1 = np.abs(center_vec1.dot(prot_plane1))
    axial1 = np.maximum(0, np.minimum(center_dis1-0.001, axial1))
    lateral1 = np.sqrt(center_dis1**2 - axial1**2)
    mask1a = (center_z1 >= 2.3 - MARGIN) & (center_z1 <= 4.5 + MARGIN)
    center_dis_corrected1 = center_dis1 - lateral1 / lateral_slope
    mask1b = (center_dis_corrected1 >= 2.3 - MARGIN) & (center_dis_corrected1 <= 3.8 + MARGIN)
    mask1 = mask1a & mask1b & (center_dis1 <= 5.0 + MARGIN)
    if mask1.sum() == 0:
        return []

    chunk_center2 = roco_nuc_center2[None, :, :] + offsets_chunk[:, None, :]
    center_vec2 = chunk_center2 - prot_center2
    center_z2 = np.abs(center_vec2[:, :, 2])
    center_dis2 = norm(center_vec2, axis=2)
    if center_dis2.min() > 5.0 + MARGIN:
        return []
    axial2 = np.abs(center_vec2.dot(prot_plane2))
    axial2 = np.maximum(0, np.minimum(center_dis2-0.001, axial2))
    lateral2 = np.sqrt(center_dis2**2 - axial2**2)
    mask2a = (center_z2 >= 2.3 - MARGIN) & (center_z2 <= 4.5 + MARGIN)
    center_dis_corrected2 = center_dis2 - lateral2 / lateral_slope
    mask2b = (center_dis_corrected2 >= 2.3 - MARGIN) & (center_dis_corrected2 <= 3.8 + MARGIN)
    mask2 = mask2a & mask2b & (center_dis2 <= 5.0 + MARGIN)
    if mask2.sum() == 0:
        return []
    
    mask = (mask1 & mask2)

    result = np.nonzero(mask)
    return np.stack(result, axis=1)

###for n in tqdm(range(0, len(offsets), chunksize)):
results = []

from tqdm.contrib.concurrent import process_map 
results0 = process_map(analyze_chunk, range(0, len(offsets), chunksize), chunksize=1, max_workers=6)
results = [(ind, r) for ind,r in enumerate(results0) if len(r)]
for ind, r in results:
    r[:, 0] += ind * chunksize
results = np.concatenate([r[1] for r in results], axis=0)


print(f"{len(results)} total solutions", file=sys.stderr)
np.save(outputfile, results)

np.save(offset_outputfile, offsets)