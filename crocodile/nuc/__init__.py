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
