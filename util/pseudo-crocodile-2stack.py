import os, sys
import numpy as np
from nefertiti.functions.parse_pdb import parse_pdb
from nefertiti.functions.parse_mmcif import parse_mmcif


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
prot_resid1 = int(sys.argv[3])
prot_resid2 = int(sys.argv[4])

ligand_template = load_file(sys.argv[5])
ligand_template_chain = sys.argv[6]
ligand_template_resid1 = int(sys.argv[7])
ligand_template_resid2 = int(sys.argv[8])
ligand_conformers_file = sys.argv[9]
ligand_conformers = np.load(ligand_conformers_file)
assert ligand_conformers.ndim == 3 and ligand_conformers.shape[1] == len(ligand_template) and ligand_conformers.shape[2] == 3, ligand_conformers.shape

ligand_rotaconformers_file = sys.argv[10]
ligand_rotaconformers = np.load(ligand_rotaconformers_file)
assert ligand_rotaconformers["conformer"].min() >= 0 and ligand_rotaconformers["conformer"].max() <= len(ligand_conformers) - 1, (ligand_rotaconformers["conformer"].min(), ligand_rotaconformers["conformer"].max(), len(ligand_conformers) - 1)

outputfile = sys.argv[11]

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

prot_coor1, _ = select_ring(prot, prot_chain, prot_resid1)
prot_coor2, _ = select_ring(prot, prot_chain, prot_resid2)
_, nuc_coor_mask1 = select_ring(ligand_template, ligand_template_chain, ligand_template_resid1)
nuc_coor1 = ligand_conformers[:, nuc_coor_mask1]
_, nuc_coor_mask2 = select_ring(ligand_template, ligand_template_chain, ligand_template_resid2)
nuc_coor2 = ligand_conformers[:, nuc_coor_mask2]

all_nuc_coor1 = nuc_coor1[ligand_rotaconformers["conformer"]]
nuc_coor_rotamers1 = np.einsum('kij,kjl->kil', all_nuc_coor1, ligand_rotaconformers["mat"]) #broadcasted all_nuc_coor1[k].dot(chunk["mat"][k])

all_nuc_coor2 = nuc_coor2[ligand_rotaconformers["conformer"]]
nuc_coor_rotamers2 = np.einsum('kij,kjl->kil', all_nuc_coor2, ligand_rotaconformers["mat"]) #broadcasted all_nuc_coor2[k].dot(chunk["mat"][k])

odis=7
d1 = prot_coor1[None, None, :, :] - nuc_coor_rotamers1[:, :, None, :]
d1low = d1 - (odis,odis,odis)
d1high = d1 + (odis,odis,odis)
d2 = prot_coor2[None, None, :, :] - nuc_coor_rotamers2[:, :, None, :]
d2low = d2 - (odis,odis,odis)
d2high = d2 + (odis,odis,odis)

solution_dtype = np.dtype([
    ("conformer", np.uint16),
    ("rotamer", np.uint32),
    ("mat", float, (4,4)),        
])
solutions = []
nsolutions = 0
for n, roco in enumerate(ligand_rotaconformers):
    if n % 1000 == 0:
        print(f"{n+1}/{len(ligand_rotaconformers)}, {nsolutions} solutions", file=sys.stderr)
    d1l, d1h = d1low[n], d1high[n]
    d2l, d2h = d2low[n], d2high[n]
    dd1l, dd1h = d1l.reshape(-1, 3), d1h.reshape(-1, 3)
    dd2l, dd2h = d2l.reshape(-1, 3), d2h.reshape(-1, 3)
    bmin = np.maximum(dd1l.max(axis=0), dd2l.max(axis=0))
    bmax = np.minimum(dd1h.min(axis=0), dd2h.min(axis=0))
    if (bmax - bmin).min() < 0:
        continue
    bmin = np.round(bmin/0.5) * 0.5
    bmax = np.round(bmax/0.5) * 0.5
    x = np.arange(bmin[0], bmax[0], 0.5)
    y = np.arange(bmin[1], bmax[1], 0.5)
    z = np.arange(bmin[2], bmax[2], 0.5)
    offset = np.empty((len(x), len(y), len(z), 3), float)
    offset[..., 0] = x[:, None, None]
    offset[..., 1] = y[None, :, None]
    offset[..., 2] = z[None, None, :]
    # offset is to be *added* to ligand coordinates
    # but *subtracted* from d1 and d2, since d1 = receptor - ligand
    od1 = d1[n].reshape(-1, 3)[None, None, None, :, :] - offset[:, :, :, None, None, :]
    dd1 = np.sqrt((od1*od1).sum(axis=-1))
    od2 = d2[n].reshape(-1, 3)[None, None, None, :, :] - offset[:, :, :, None, None, :]
    dd2 = np.sqrt((od2*od2).sum(axis=-1))
    mask1a = (dd1.max(axis=-1) < 7)
    mask1b = (dd1.min(axis=-1) < 5)
    mask2a = (dd2.max(axis=-1) < 7)
    mask2b = (dd2.min(axis=-1) < 5)
    mask = (mask1a & mask1b & mask2a & mask2b)
    mask = mask.reshape(mask.shape[:3])
    ntrans = mask.sum()
    if not ntrans:
        continue
    curr_solutions = np.zeros(ntrans, solution_dtype)
    curr_solutions["conformer"] = roco["conformer"]
    curr_solutions["rotamer"] = roco["rotamer"]
    mat4 = curr_solutions["mat"]
    mat4[:, 3,3] = 1
    mat4[:, :3,:3] = roco["mat"]
    mat4[:, 3,:3] = offset[mask]

    solutions.append(curr_solutions)
    nsolutions += ntrans

solutions = np.concatenate(solutions)
assert len(solutions) == nsolutions
print(f"{nsolutions} total solutions", file=sys.stderr)
np.save(outputfile, solutions)