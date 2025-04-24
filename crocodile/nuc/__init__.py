import numpy as np
from typing import Tuple
from nefertiti.functions.parse_pdb import atomic_dtype


_basic_mapping_rna = {k: k for k in ("A", "C", "G", "U")}
_basic_mapping_dna = {k: k for k in ("A", "C", "G", "T")}
_basic_mapping_rna.update({"R" + k: k for k in ("A", "C", "G", "U")})
_basic_mapping_dna.update({"D" + k: k for k in ("A", "C", "G", "T")})


def map_resname(
    resname: bytes | str, *, rna: bool = True, extra_mapping: dict | None = None
):
    """Maps a parsed PDB resname to a mononucleotide sequence.
    An unknown mapping gives rise to a KeyError.
    Extra resnames (i.e mutated bases) can be provided via extra_mapping"""

    if isinstance(resname, bytes):
        resname = resname.decode()
    mapping = _basic_mapping_rna if rna else _basic_mapping_dna
    if extra_mapping:
        mapping = mapping.copy().update(extra_mapping)
    try:
        return mapping[resname]
    except KeyError as exc:
        if not rna and resname in _basic_mapping_rna:
            raise KeyError("Mode is DNA (rna=False) but resname is RNA") from None
        elif rna and resname in _basic_mapping_dna:
            raise KeyError("Mode is RNA but resname is DNA") from None
        else:
            raise exc from None


def ppdb2nucseq(
    ppdb: np.ndarray,
    *,
    rna: bool,
    ignore_unknown: bool = False,
    return_index: bool = False,
) -> str | Tuple[str, list]:
    """Converts a nucleic acid parsed PDB to a mono-nucleotide sequence.
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
