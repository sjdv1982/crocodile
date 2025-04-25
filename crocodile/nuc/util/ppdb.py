# Converts a parsed PDB into a list of nucleotides

import numpy as np
from typing import Tuple
from nefertiti.functions.parse_pdb import atomic_dtype


def ppdb2nucseq(
    ppdb: np.ndarray,
    *,
    rna: bool,
    ignore_unknown: bool = False,
    return_index: bool = False,
) -> str | Tuple[str, list]:
    """Converts a nucleic acid parsed PDB to a trinucleotide sequence.
    Returns mono-nucleotide sequence.
    If rna, the parsed PDB is interpreted as RNA, else as DNA
    If ignore_unknown, unknown residue names (i.e. non-canonical bases) are ignored, else raise an exception
    If return_index, return a list of (first, last+1) indices,
      corresponding to the first and last atom indices of each nucleotide
    """
    if ppdb.dtype != atomic_dtype:
        raise TypeError(ppdb.dtype)
    if len(np.unique(ppdb["chain"])) > 1:
        raise ValueError("Multiple chains in parsed PDB")

    indices0 = np.nonzero(np.diff(ppdb["resid"], prepend=-1, append=-1))[0]
    indices = []
    nucs = []
    for n in range(len(indices0) - 1):
        ind1, ind2 = indices0[n], indices0[n + 1]
        try:
            nuc = map_resname(ppdb[ind1]["resname"], rna=rna)
        except KeyError as exc:
            if ignore_unknown:
                continue
            else:
                raise exc from None
        nucs.append(nuc)
        indices.append((ind1, ind2))

    seq = "".join(nucs)
    if return_index:
        return seq, indices
    else:
        return seq


from crocodile.nuc import map_resname
