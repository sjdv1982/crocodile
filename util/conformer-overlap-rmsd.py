import numpy as np
from nefertiti.functions.parse_pdb import parse_pdb
from nefertiti.functions.superimpose import superimpose_array

import sys

def load_selected_coors(template_pdb, conformers_file, residues):
    template = parse_pdb(open(template_pdb).read())
    conformers = np.load(conformers_file)
    assert conformers.ndim == 3 and conformers.shape[1] == len(template), (template.shape, conformers.shape)
    res1, res2 = residues
    mask1 = (template["resid"] == res1)
    mask2 = (template["resid"] == res2)
    mask = (mask1 | mask2)
    selected_coors = conformers[:, mask]
    return selected_coors

motif1_template_pdb = sys.argv[1]
motif1_conformers_file = sys.argv[2]
motif1_coors = load_selected_coors(motif1_template_pdb, motif1_conformers_file, (2,3))

motif2_template_pdb = sys.argv[3]
motif2_conformers_file = sys.argv[4]
motif2_coors = load_selected_coors(motif2_template_pdb, motif2_conformers_file, (1,2))

outfile = sys.argv[5]

assert motif1_coors.shape[1] == motif2_coors.shape[1], (motif1_coors.shape, motif2_coors.shape)

motif1_coors -= motif1_coors.mean(axis=1)[:, None, :]
motif2_coors -= motif2_coors.mean(axis=1)[:, None, :]

overlap_rmsd = np.empty((len(motif1_coors), len(motif2_coors)))
for n, curr_motif1_coors in enumerate(motif1_coors):
    #if n % 100 == 0:
    #    print(f"{n+1}/{len(motif1_coors)}", file=sys.stderr)
    _, rmsd = superimpose_array(motif2_coors, curr_motif1_coors)
    overlap_rmsd[n] = rmsd

np.save(outfile, overlap_rmsd)