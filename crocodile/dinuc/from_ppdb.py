# Converts a parsed PDB  into a list of superimposed dinucleotides

import itertools
import sys
import numpy as np
from nefertiti.functions.parse_pdb import atomic_dtype
from nefertiti.functions.superimpose import superimpose_array
from crocodile.nuc.from_ppdb import ppdb2nucseq
from crocodile.dinuc import dinuc_dtype


def ppdb2coor(
    ppdb: np.ndarray,
    *,
    template_pdbs: dict[str, np.ndarray],
    rna: bool,
    ignore_unknown: bool,
    ignore_missing: bool,
    ignore_reordered: bool,
):
    """Converts a parsed nucleic acid PDB chain
    into a dinucleotide sequence and a dict of dinucleotide coordinate arrays.

    template_pdbs: dictionary of template parsed PDBs for each dinucleotide sequence

    Dinuncleotides that are non-contiguous in residue numbering are skipped.
    If ignore_unknown, unknown residue names (i.e. non-canonical bases) are ignored, else raise an exception
    If ignore_missing, ignore nucleotides with missing template atoms, else raise an exception
    If ignore_reordered, ignore nucleotides with extra atoms or a different atom order than the template, else reorder them

    Returns: sequence, first_resids, coordinates
        sequence: list of dinucleotide sequences
            example: ["AA", "AC", "CA", "AA", "AA", "AC", "CC", ...]
        first_resids: list of the first resid of each dinucleotide.
        coordinates: list of Numpy arrays,
            where each array has the shape (a, 3),
                where a is the number of atoms in the template of the corresponding dinucleotide
    """

    template_inds = {}

    for diseq in template_pdbs:
        seq, inds = ppdb2nucseq(
            template_pdbs[diseq], rna=rna, ignore_unknown=False, return_index=True
        )
        if seq != diseq:
            raise ValueError(f"Invalid template for {diseq} ({seq})")
        template_inds[diseq] = inds

    monoseq, indices = ppdb2nucseq(
        ppdb, rna=rna, ignore_unknown=ignore_unknown, return_index=True
    )

    sequence = []
    first_resids = []
    coordinates = []
    for n in range(len(monoseq) - 1):
        resids = [ppdb["resid"][indices[n + i][0]] for i in range(2)]
        if resids[1] != resids[0] + 1:  # non-contiguous numbering
            continue
        first_resids.append(resids[0])
        di_inds = indices[n : n + 2]
        discontinuous = False
        if di_inds[0][1] != di_inds[1][0]:
            discontinuous = True
        if discontinuous:
            continue

        diseq = monoseq[n : n + 2]
        if diseq not in template_pdbs:
            raise ValueError(f"Unknown dinucleotide '{diseq}'")
        template = template_pdbs[diseq]
        first = di_inds[0][0]
        first_atom = ppdb[first]
        di_inds_offset = [(v[0] - first, v[1] - first) for v in di_inds]

        di_natoms = di_inds[1][1] - di_inds[0][0]
        if di_natoms < len(template):
            if ignore_missing:
                continue
            raise ValueError(
                f"""dinucleotide '{diseq}' starting at residue {first_atom['resid']}: 
Incorrect number of atoms {di_natoms}, while template has {len(template)} atoms"""
            )
        elif any(
            [
                (x[1] - x[0] < y[1] - y[0])
                for x, y in zip(di_inds_offset, template_inds[diseq])
            ]
        ):
            if ignore_missing:
                continue
            raise ValueError(
                f"""dinucleotide '{diseq}' starting at residue {first_atom['resid']}: 
Dinucleotide has layout {di_inds_offset}, while template has {template_inds[diseq]}"""
            )

        # Allow for out-of-order atoms
        atom_indices = []
        baseline_template = template_inds[diseq][0][0]
        baseline_ppdb = di_inds[0][0]
        for inuc in range(2):
            t_inds = template_inds[diseq][inuc]
            baseline_inuc = t_inds[0]
            natoms = t_inds[1] - t_inds[0]
            for i in range(natoms):
                atom_ind_template = baseline_template + baseline_inuc + i
                atom_template = template[atom_ind_template]
                atom_name = atom_template["name"]
                for ii in range(natoms):
                    atom_ind_ppdb = baseline_ppdb + baseline_inuc + ii
                    atom_ppdb = ppdb[atom_ind_ppdb]
                    if atom_ppdb["name"] == atom_name:
                        atom_indices.append(atom_ind_ppdb)
                        break
                else:
                    break
        if len(atom_indices) < di_natoms:
            if ignore_missing:
                continue
            raise ValueError(
                f"""Dinucleotide '{diseq}' starting at residue {first_atom['resid']}: 
Dinucleotide has at least the same number of atoms as the template, but they are differently named"""
            )
        elif ignore_reordered:
            if atom_indices != list(range(baseline_ppdb, baseline_ppdb + di_natoms)):
                continue
        ppdb_dinuc = ppdb[atom_indices]
        refe = np.stack(
            (ppdb_dinuc["x"], ppdb_dinuc["y"], ppdb_dinuc["z"]), axis=-1
        ).astype(float)

        sequence.append(diseq)
        coordinates.append(refe)

    return sequence, first_resids, coordinates


def from_ppdb(
    ppdb: np.ndarray,
    *,
    template_pdbs: dict[str, np.ndarray],
    dinuc_conformer_library: dict[str, np.ndarray],
    rna: bool,
    ignore_unknown: bool,
    ignore_missing: bool,
    ignore_reordered: bool,
    rmsd_margin: float,
    rmsd_soft_max: float = None,
):
    """Converts a parsed nucleic acid PDB chain
    into a list of superimposed dinucleotides.

    template_pdbs: dictionary of template parsed PDBs for each dinucleotide sequence
    dinuc_conformer_library: dictionary of numerical arrays of shape (nconf, natoms, 3),
      one for each dinucleotide sequence,
      where "natoms" is the same as the length of the template PDB for that sequence.

    Trinuncleotides that are non-contiguous in residue numbering are skipped.
    If ignore_unknown, unknown residue names (i.e. non-canonical bases) are ignored, else raise an exception
    If ignore_missing, ignore nucleotides with missing template atoms, else raise an exception
    If ignore_reordered, ignore nucleotides with extra atoms or a different atom order than the template, else reorder them

    All conformers are fitted, a conformer is kept if its fitting RMSD is smaller than sqrt(best-fitting-RMSD**2 + margin**2)

    For conformers with fitting RMSDs beyond rmsd_soft_max, they are only kept if they are the best fitting

    NOTE: validation of the data must have been done before!
    """
    sequence, first_resids, coordinates = ppdb2coor(
        ppdb,
        template_pdbs=template_pdbs,
        rna=rna,
        ignore_unknown=ignore_unknown,
        ignore_missing=ignore_missing,
        ignore_reordered=ignore_reordered,
    )

    results0 = []
    for diseq, first_resid, refe in zip(sequence, first_resids, coordinates):

        if diseq not in dinuc_conformer_library:
            raise ValueError(f"Unknown dinucleotide '{diseq}'")

        conformers = dinuc_conformer_library[diseq]
        rotmatrices, rmsds = superimpose_array(conformers, refe)

        sorting = np.argsort(rmsds)
        sorted_indices = np.arange(len(rmsds), dtype=np.uint16)[sorting]
        rmsds = rmsds[sorting]
        rotmatrices = rotmatrices[sorting]

        best_rmsd = rmsds.min()
        max_rmsd = np.sqrt(best_rmsd**2 + rmsd_margin**2)
        if rmsd_soft_max:
            max_rmsd = min(max_rmsd, rmsd_soft_max)
            max_rmsd = max(max_rmsd, best_rmsd)
        best_conformers0 = np.where(rmsds <= max_rmsd)[0]
        best_rmsds = rmsds[best_conformers0]
        best_rotmatrices = rotmatrices[best_conformers0]
        best_conformers = sorted_indices[best_conformers0]
        best_confs = conformers[best_conformers]

        best_matrices = np.zeros((len(best_conformers), 4, 4), float)
        best_matrices[:, :3, :3] = best_rotmatrices
        best_matrices[:, 3, :3] = refe.mean(axis=0) - np.einsum(
            "ik,ikl->il", best_confs.mean(axis=1), best_rotmatrices
        )
        best_matrices[:, 3, 3] = 1

        results0.append(
            (first_resid, diseq, best_conformers, best_matrices, best_rmsds)
        )

    nresults = sum([len(result0[2]) for result0 in results0])
    results = np.empty(nresults, dtype=dinuc_dtype)
    pos = 0
    for result0 in results0:
        first_resid, diseq, best_conformers, best_matrices, best_rmsds = result0
        for n in range(len(best_conformers)):
            result = results[pos]
            result["first_resid"] = first_resid
            result["sequence"] = diseq
            result["conformer"] = best_conformers[n]
            result["matrix"] = best_matrices[n]
            result["rmsd"] = best_rmsds[n]
            pos += 1

    return results


def err(msg):
    print(msg, file=sys.stderr)
    exit(1)


def main():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "ppdb",
        help="A numpy array of a parsed PDB, containing a single nucleotide chain",
    )
    parser.add_argument(
        "--templates",
        help="""Template file pattern.
                        Pattern must contain XXX, which will be replaced for each dinucleotide sequence.
                        Each file must contain a numpy array of a parsed PDB, corresponding to that dinucleotide
                        """,
        required=True,
    )
    parser.add_argument(
        "--conformers",
        help="""Conformer file pattern.
                        Pattern must contain XXX, which will be replaced for each dinucleotide sequence.
                        Each file must contain a numpy array of (nconf, natoms, 3) numbers.
                        """,
        required=True,
    )

    parser.add_argument(
        "--output",
        help="Output file for resulting array of dinucleotide fitted conformers",
        required=True,
    )

    parser.add_argument(
        "--rna", action="store_true", help="Interpret the parsed PDB as RNA"
    )
    parser.add_argument(
        "--dna", action="store_true", help="Interpret the parsed PDB as DNA"
    )
    parser.add_argument(
        "--ignore-unknown",
        action="store_true",
        help="Ignore unknown resnames, i.e. non-canonical bases",
    )

    parser.add_argument(
        "--ignore-missing",
        action="store_true",
        help="Ignore nucleotides with missing atoms",
    )

    parser.add_argument(
        "--ignore-reordered",
        action="store_true",
        help="Ignore nucleotides with additional or reordered atoms",
    )

    parser.add_argument(
        "--margin",
        default=1.8,
        help="""Store not only the best-fitting conformer, but all conformers up to 'margin' beyond that.
        To be precise, a conformer is kept if its fitting RMSD is smaller than sqrt(best-fitting-RMSD**2 + margin**2)""",
    )

    args = parser.parse_args()
    if args.dna == args.rna:
        err("Specify --dna OR --rna")

    bases = ("A", "C", "G", "T") if args.dna else ("A", "C", "G", "U")
    dinuc_sequences = ["".join(s) for s in itertools.product(bases, repeat=3)]

    ppdb_file = args.ppdb
    ppdb = np.load(ppdb_file)
    if ppdb.dtype != atomic_dtype:
        err(f"'{ppdb_file }' does not contain a parsed PDB")

    if len(np.unique(ppdb["chain"])) > 1:
        err(f"'{ppdb_file }' contains multiple chains")

    templates = {}
    for seq in dinuc_sequences:
        filename = args.templates.replace("XXX", seq)
        template = np.load(filename)
        if template.dtype != atomic_dtype:
            err(f"Template '{filename}' does not contain a parsed PDB")
        templates[seq] = template

    conformers = {}
    for seq in dinuc_sequences:
        filename = args.conformers.replace("XXX", seq)
        conformer = np.load(filename)
        if conformer.dtype not in (np.float32, np.float64):
            err(f"Conformer file '{filename}' does not contain an array of numbers")
        if conformer.ndim != 3:
            err(f"Conformer file '{filename}' does not contain a 3D coordinate array")
        if conformer.shape[1] != len(templates[seq]):
            err(
                f"Sequence {seq}: conformer '{filename}' doesn't have the same number of atoms as the template"
            )
        conformers[seq] = conformer.astype(float)

    result = from_ppdb(
        ppdb,
        template_pdbs=templates,
        dinuc_conformer_library=conformers,
        rna=args.rna,
        ignore_unknown=args.ignore_unknown,
        ignore_missing=args.ignore_missing,
        ignore_reordered=args.ignore_reordered,
        rmsd_margin=args.margin,
    )

    np.save(args.output, result)


if __name__ == "__main__":
    main()
