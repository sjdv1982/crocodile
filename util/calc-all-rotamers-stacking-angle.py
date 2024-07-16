
import json
import os, sys
import numpy as np
from nefertiti.functions.parse_pdb import parse_pdb
from nefertiti.functions.parse_mmcif import parse_mmcif
from numpy.linalg import svd, norm
from tqdm import tqdm

import seamless
seamless.delegate(level=1)

from seamless import transformer, Checksum

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

rotamer_template = load_file(sys.argv[4])
rotamer_template_chain = sys.argv[5]
rotamer_template_resid = int(sys.argv[6])

def select_ring(mol, chain, resid):
    if chain.upper() != "ALL":
        mask = (mol["chain"] == chain.encode())
        mol = mol[mask]
        assert len(mol)
    mask1 = (mol["resid"] == resid)
    assert mask1.sum()

    mol_resname = mol[mask1][0]["resname"].decode()
    mol_ring = RINGS[mol_resname]
    mask2 = np.isin(mol["name"], [a.encode() for a in mol_ring])
    assert mask2.sum()

    mask = (mask1 & mask2)
    mol = mol[mask]
    mol_coor = np.stack((mol["x"], mol["y"], mol["z"]), axis=1)
    return mol_coor, mask

def calc_plane(coor):
    coor = coor - coor.mean(axis=0)
    covar = coor.T.dot(coor)
    v, s, wt = svd(covar)
    plane_directionless = wt[2] / norm(wt[2])
    plane0_directed = np.cross((coor[1] - coor[0]), (coor[2] - coor[1]))
    flip = np.sign(np.dot(plane_directionless, plane0_directed))
    return plane_directionless * flip

def calc_planes(coor):
    coor = coor - coor.mean(axis=1)[:, None]
    covar = np.einsum("ijk,ijl->ikl", coor, coor) #coor[i].T.dot(coor[i])
    v, s, wt = svd(covar)
    planes_directionless = wt[:, 2, :] / norm(wt[:, 2, :], axis=-1)[..., None]
    planes0_directed = np.cross((coor[:, 1, :] - coor[:, 0, :]), (coor[:, 2, :] - coor[:, 1, :]), axis=1)
    flip = np.sign(np.einsum("ij,ij->i", planes_directionless, planes0_directed))
    return planes_directionless * flip[:, None]

prot_coor, _ = select_ring(prot, prot_chain, prot_resid)
_, nuc_coor_mask = select_ring(rotamer_template, rotamer_template_chain, rotamer_template_resid)

prot_plane = calc_plane(prot_coor)
prot_center = prot_coor.mean(axis=0)
prot_x_vector = prot_coor[0] - prot_center
prot_x_vector /= norm(prot_x_vector)
prot_y_vector = np.cross(prot_plane, prot_x_vector)
prot_y_vector /= norm(prot_y_vector)

@transformer(return_transformation=True)
def calc_stacking_angle(prot_plane, prot_x_vector, prot_y_vector, nuc_conf_coor, rotamer_matrices):
    import numpy as np
    from numpy.linalg import svd, norm

    def calc_planes(coor):
        coor = coor - coor.mean(axis=1)[:, None]
        covar = np.einsum("ijk,ijl->ikl", coor, coor) #coor[i].T.dot(coor[i])
        v, s, wt = svd(covar)
        return wt[:, 2, :] / norm(wt[:, 2, :], axis=-1)[..., None]

    nuc_coor_rotamers = np.einsum('ij,kjl->kil', nuc_conf_coor, rotamer_matrices) #broadcasted nuc_coor.dot(rotamer_matrices[k])
    nuc_plane_rotamers = calc_planes(nuc_coor_rotamers)
    nrm = norm(np.cross(prot_plane, nuc_plane_rotamers), axis=1)
    nrm = np.minimum(nrm, 1)
    angles = np.arcsin(nrm)
                       
    nuc_centers = nuc_coor_rotamers.mean(axis=1)
    nuc_x_vectors0 = nuc_coor_rotamers[:, 0, :] - nuc_centers
    nuc_x_vectors0 -= np.einsum("ij,j->i", nuc_x_vectors0, prot_plane)[:, None] * prot_plane[None, :]
    
    dihed = np.zeros(len(rotamer_matrices))
    mask = (norm(nuc_x_vectors0, axis=1) < 0.0001)    
    nuc_x_vectors = nuc_x_vectors0 / norm(nuc_x_vectors0, axis=1)[:, None]
    dihed_x = np.einsum("ij,j->i", nuc_x_vectors, prot_x_vector)
    dihed_y = np.einsum("ij,j->i", nuc_x_vectors, prot_y_vector)  
    dihed0 = np.angle(dihed_x + 1.0j * dihed_y)
    dihed[~mask] = dihed0[~mask]
    dihed[mask] = np.pi / 2
    
    return np.stack((angles, dihed), axis=-1)

conformers_file = sys.argv[7]
conformers = np.load(conformers_file)
assert conformers.ndim == 3 and conformers.shape[1] == len(rotamer_template) and conformers.shape[2] == 3

nuc_coor = conformers[:, nuc_coor_mask]

rotamer_matrices_index_file = sys.argv[8]
with open(rotamer_matrices_index_file) as f:
    rotamer_matrices_index = json.load(f)

assert len(rotamer_matrices_index) == len(conformers)

output_file = sys.argv[9]

def func(n):
    rotamer_matrices = Checksum(rotamer_matrices_index[n])
    return calc_stacking_angle(prot_plane, prot_x_vector, prot_y_vector, nuc_coor[n], rotamer_matrices)

results = [None] * len(rotamer_matrices_index)

with tqdm(total=len(rotamer_matrices_index)) as progress_bar:
    with seamless.multi.TransformationPool(10) as pool:
        def callback(n, tfm):
            progress_bar.update(1)
            value = tfm.value
            if value is None:
                print(
                    f"""Failure for conformer {n}:
        status: {tfm.status}
        exception: {tfm.exception}
        logs: {tfm.logs}"""
                )
            results[n] = value

        transformations = pool.apply(func, len(rotamer_matrices_index), callback=callback)

if not any([r is None for r in results]):    
    results = np.concatenate(results)
    np.save(output_file, results)
