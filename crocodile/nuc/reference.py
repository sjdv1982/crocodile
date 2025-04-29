import numpy as np
from .util.ppdb import ppdb2nucseq


class Reference:
    """PDB reference for fitting. The reference object can be queried for fragment coordinates.

    ppdb: Parsed PDB in numpy format
    mononucleotide_templates: dictionary of template parsed PDBs for each mononucleotide

    rna: True for RNA, else DNA

    If ignore_unknown, unknown residue names (i.e. non-canonical bases) are ignored, else raise an exception
    If ignore_missing, ignore nucleotides with missing template atoms, else raise an exception
    If ignore_reordered, ignore nucleotides with extra atoms or a different atom order than the template, else reorder them

    Discontinuous residue numberings are allowed, but are considered as chain breaks; no fragments with discontinuity are returned
    """

    def __init__(
        self,
        ppdb: np.ndarray,
        *,
        mononucleotide_templates: dict[str, np.ndarray],
        rna,
        ignore_unknown: bool,
        ignore_missing: bool,
        ignore_reordered: bool,
    ):

        template_inds = {}

        for base in mononucleotide_templates:
            seq, inds = ppdb2nucseq(
                mononucleotide_templates[base],
                rna=rna,
                ignore_unknown=False,
                return_index=True,
            )
            if seq != base:
                raise ValueError(f"Invalid template for {base} ({seq})")
            template_inds[base] = inds

        monoseq, indices = ppdb2nucseq(
            ppdb, rna=rna, ignore_unknown=ignore_unknown, return_index=True
        )

        sequence = []
        resids = []
        coordinates = []
        for n in range(len(monoseq)):
            first, last = indices[n]
            first_atom = ppdb[first]
            resid0 = first_atom["resid"]

            base = monoseq[n]
            if base not in mononucleotide_templates:
                if ignore_unknown:
                    continue
                raise ValueError(f"Unknown base '{base}'")

            template = mononucleotide_templates[base]

            if last - first < len(template):
                if ignore_missing:
                    continue
                raise ValueError(
                    f"""Nucleotide '{base}', resid {resid0}: 
Incorrect number of atoms {last - first}, while template has {len(template)} atoms"""
                )

            atom_indices = []
            has_missing_atoms = False
            for t_atom in template:
                atom_name = t_atom["name"]
                matches = []
                for atom_nr in range(first, last):
                    atom = ppdb[atom_nr]
                    if atom["name"] == atom_name:
                        matches.append(atom_nr)
                if len(matches) == 0:
                    if ignore_missing:
                        has_missing_atoms = True
                        break
                    raise ValueError(
                        f"""Nucleotide '{base}', resid {resid0}: 
Missing atom {atom_name}"""
                    )
                elif len(matches) > 1:
                    raise ValueError(
                        f"""Nucleotide '{base}', resid {resid0}: 
Duplicate atom {atom_name}"""
                    )
                atom_indices.append(matches[0])

            if has_missing_atoms:
                continue

            if ignore_reordered:
                if atom_indices != list(range(first, last)):
                    continue

            resids.append(resid0)
            ppdb_dinuc = ppdb[atom_indices]
            refe_coor = np.stack(
                (ppdb_dinuc["x"], ppdb_dinuc["y"], ppdb_dinuc["z"]), axis=-1
            ).astype(float)
            sequence.append(base)
            coordinates.append(refe_coor)

        self.ppdb = ppdb
        self.rna = rna
        self.sequence = "".join(sequence)
        self._coordinates = coordinates
        self._resids = resids

    def get_fragment_positions(self, fraglen: int):
        """
        Get the positions of all valid fragments

        Fragment 1 (with position 1) is the first fragment.
        It has position 1 regardless of the first resid.
        Fragment 2 is the fragment starting at the first resid + 1, etc.

        Valid fragments are those where no nucleotides are missing.
        Nucleotides may be considered missing because of:
        - Non-contiguous numbering
        - Unknown bases
        - Missing atoms in a nucleotide
        - Out-of-order atoms in a nucleotide

        """
        result = []
        for pos0 in range(len(self._resids) - fraglen + 1):
            resids = self._resids[pos0 : pos0 + fraglen]
            first = resids[0]
            if resids != list(range(first, first + fraglen)):
                continue
            pos = first - self._resids[0] + 1
            result.append(pos)
        return result

    def get_resids(self, fragment_position: int, fraglen: int):
        """Get the residues belonging to fragment X, where X is the fragment position.

        If the fragment is not a valid fragment, raise ValueError"""
        offset = self._resids[0]
        first_resid = offset + fragment_position - 1
        try:
            pos = self._resids.index(first_resid)
        except IndexError:
            raise ValueError from None
        resids = self._resids[pos : pos + fraglen]
        if resids != list(range(first_resid, first_resid + fraglen)):
            raise ValueError("Not a valid fragment")
        return resids

    def _get_pos(self, fragment_position, fraglen):
        offset = self._resids[0]
        first_resid = offset + fragment_position - 1
        try:
            pos = self._resids.index(first_resid)
        except IndexError:
            raise ValueError from None
        if self._resids[pos : pos + fraglen] != list(
            range(first_resid, first_resid + fraglen)
        ):
            raise ValueError("Not a valid fragment")
        return pos

    def get_coordinates(self, fragment_position: int, fraglen: int):
        """Get the coordinates belonging to fragment X, where X is the fragment position.

        If the fragment is not a valid fragment, raise ValueError"""
        pos = self._get_pos(fragment_position, fraglen)
        coordinates = self._coordinates[pos : pos + fraglen]
        if fraglen == 1:
            return coordinates[0].copy()
        else:
            return np.concatenate(coordinates)

    def get_sequence(self, fragment_position: int, fraglen: int):
        """Get the sequence belonging to fragment X, where X is the fragment position.

        If the fragment is not a valid fragment, raise ValueError"""
        pos = self._get_pos(fragment_position, fraglen)
        return self.sequence[pos : pos + fraglen]
