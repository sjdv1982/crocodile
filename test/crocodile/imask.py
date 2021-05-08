# build interface mask
from scipy.spatial import cKDTree
import numpy as np

def imask(pdbs, cutoff):
    result = []
    coors = []
    for pdb in pdbs:
        coor = np.zeros((len(pdb), 3))
        coor[:, 0] = pdb["x"]
        coor[:, 1] = pdb["y"]
        coor[:, 2] = pdb["z"]
        coors.append(coor)
    for pdb, coor in zip(pdbs, coors):
        pdb_tree = cKDTree(coor)
        other_pdbs = np.concatenate([c for c in coors if c is not coor])
        other_pdb_tree = cKDTree(other_pdbs)
        all_pairs = pdb_tree.query_ball_tree(other_pdb_tree, r=cutoff)
        contact_res = set()
        for chain, resid, pairs in zip(pdb["chain"], pdb["resid"], all_pairs):
            if len(pairs):
                contact_res.add((chain, resid))
        curr_result = np.array(
            [((chain, resid) in contact_res) for chain, resid in zip(pdb["chain"], pdb["resid"]) ]
        )
        curr_result = curr_result & np.isin(pdb["name"], [b"CA", b"C", b"O", b"N"])
        result.append(curr_result)
    return result
    