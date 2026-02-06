import sys
import json
import numpy as np
from crocodile.nuc.reference import Reference
from nefertiti.functions.parse_pdb import parse_pdb
from nefertiti.functions.write_pdb import write_pdb
from library_config import (
    mononucleotide_templates,
    dinucleotide_templates,
    dinucleotide_libraries,
)

refe_ppdb = parse_pdb(open(sys.argv[1]).read())
refe = Reference(
    ppdb=refe_ppdb,
    mononucleotide_templates=mononucleotide_templates,
    rna=True,
    ignore_unknown=False,
    ignore_missing=False,
    ignore_reordered=False,
)


positions = [0] + np.cumsum([len(c) for c in refe._coordinates]).tolist()
print(positions)
for fragpos in refe.get_fragment_positions(2):
    pos = refe._get_pos(fragpos, 2)
    start, end = positions[pos], positions[pos + 2]
    frag_ppdb = refe_ppdb[start:end]
    pdbtxt = write_pdb(frag_ppdb)
    with open(f"fragments/frag-{fragpos}.pdb", "w") as f:
        f.write(pdbtxt)
