import numpy as np
from nefertiti.functions.parse_pdb import parse_pdb

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
motif1_rotaconformers_file = sys.argv[3]
motif1_rotaconformers = np.load(motif1_rotaconformers_file)


motif2_template_pdb = sys.argv[4]
motif2_conformers_file = sys.argv[5]
motif2_coors = load_selected_coors(motif2_template_pdb, motif2_conformers_file, (1,2))
motif2_rotaconformers_file = sys.argv[6]
motif2_rotaconformers = np.load(motif2_rotaconformers_file)

overlap_rmsd_file = sys.argv[7]
overlap_rmsd = np.load(overlap_rmsd_file)

outfile = sys.argv[8]

assert motif1_coors.shape[1] == motif2_coors.shape[1], (motif1_coors.shape, motif2_coors.shape)
natoms = motif1_coors.shape[1]

motif1_coors -= motif1_coors.mean(axis=1)[:, None, :]
motif2_coors -= motif2_coors.mean(axis=1)[:, None, :]

MAX_OVERLAPS = 100000000
overlaps_dtype = np.dtype([
    ("conformer1", np.int16),
    ("rotamer1", np.int32),
    ("conformer2", np.int16),
    ("rotamer2", np.int32),
    ("rmsd", np.float32),
])
overlaps = np.empty(MAX_OVERLAPS, overlaps_dtype)
pos = 0
t = 1.0
tsq = t ** 2

CHUNKSIZE = 100
for n, curr_motif1_coors in enumerate(motif1_coors):
    print(f"{n+1}/{len(motif1_coors)}", file=sys.stderr)
    curr_motif1_rotaconformers = motif1_rotaconformers["mat"][motif1_rotaconformers["conformer"] == n]
    curr_motif1_rocoors = np.einsum("ij,kjl->kil", curr_motif1_coors, curr_motif1_rotaconformers)
    for nn, curr_motif2_coors in enumerate(motif2_coors):
        if overlap_rmsd[n, nn] > t:
            continue
        print(f"{n+1}/{len(motif1_coors)} {nn+1}/{len(motif2_coors)}", file=sys.stderr)
        curr_motif2_coors = motif2_coors[nn]
        curr_motif2_rotaconformers = motif2_rotaconformers["mat"][motif2_rotaconformers["conformer"] == nn]
        for nnn in range(0, len(curr_motif2_rotaconformers), CHUNKSIZE):
            ccurr_motif2_rotaconformers = curr_motif2_rotaconformers[nnn:nnn+CHUNKSIZE]
            print(f"{n+1}/{len(motif1_coors)} {nn+1}/{len(motif2_coors)} {nnn+1}/{len(curr_motif2_rotaconformers)}, {pos}", file=sys.stderr)
            ccurr_motif2_rocoors = np.einsum("ij,kjl->kil", curr_motif2_coors, ccurr_motif2_rotaconformers)
            d = curr_motif1_rocoors[:, None] - ccurr_motif2_rocoors[None, :]
            msd = np.einsum("abjk,abjk->ab",d,d)/natoms
            ind1, ind2 = np.where(msd<tsq)
            nind = len(ind1)
            if nind:
                assert pos+nind < MAX_OVERLAPS
                chunk = overlaps[pos:pos+nind]
                chunk["conformer1"] = n
                chunk["rotamer1"] = ind1
                chunk["conformer2"] = nn
                chunk["rotamer2"] = ind2
                chunk["rmsd"] = np.sqrt(msd[ind1, ind2])
                pos += nind        

np.save(outfile, overlaps)