# Converts a parsed PDB  into a list of superimposed trinucleotides

import itertools
import sys
import numpy as np
from typing import Tuple
from nefertiti.functions.parse_pdb import atomic_dtype
from nefertiti.functions.superimpose import superimpose_array
from crocodile.trinuc import map_resname, trinuc_dtype


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


def from_ppdb(
    ppdb: np.ndarray,
    *,
    template_pdbs: dict[str, np.ndarray],
    trinuc_conformer_library: dict[str, np.ndarray],
    rna: bool,
    ignore_unknown: bool,
    ignore_missing: bool,
    rmsd_margin: float,
):
    """Converts a parsed nucleic acid PDB chain
    into a list of superimposed trinucleotides.

    template_pdbs: dictionary of template parsed PDBs for each trinucleotide sequence
    trinuc_conformer_library: dictionary of numerical arrays of shape (nconf, natoms, 3),
      one for each trinucleotide sequence,
      where "natoms" is the same as the length of the template PDB for that sequence.

    If ignore_unknown, unknown residue names (i.e. non-canonical bases) are ignored, else raise an exception
    If ignore_missing, ignore nucleotides with missing heavy atoms

    All conformers are fitted, a conformer is kept if its fitting RMSD is smaller than sqrt(best-fitting-RMSD**2 + margin**2)

    NOTE: validation of the data must have been done before!
    """
    template_inds = {}
    for triseq in template_pdbs:
        seq, inds = ppdb2nucseq(
            template_pdbs[triseq], rna=rna, ignore_unknown=False, return_index=True
        )
        if seq != triseq:
            raise ValueError(f"Invalid template for {triseq} ({seq})")
        template_inds[triseq] = inds

    seq, indices = ppdb2nucseq(
        ppdb, rna=rna, ignore_unknown=ignore_unknown, return_index=True
    )
    results0 = []

    for n in range(len(seq) - 2):
        resids = [ppdb["resid"][indices[n + i][0]] for i in range(3)]
        if (
            resids[1] != resids[0] + 1 or resids[2] != resids[1] + 1
        ):  # non-contiguous numbering
            continue
        tri_inds = indices[n : n + 3]
        discontinuous = False
        for nn in range(2):
            if tri_inds[nn][1] != tri_inds[nn + 1][0]:
                discontinuous = True
        if discontinuous:
            continue

        triseq = seq[n : n + 3]
        if triseq not in template_pdbs or triseq not in trinuc_conformer_library:
            raise ValueError(f"Unknown trinucleotide '{triseq}'")
        template = template_pdbs[triseq]
        first = tri_inds[0][0]
        first_atom = ppdb[first]
        tri_inds_offset = [(v[0] - first, v[1] - first) for v in tri_inds]

        tri_natoms = tri_inds[2][1] - tri_inds[0][0]
        if tri_natoms != len(template):
            if ignore_missing:
                continue
            raise ValueError(
                f"""Trinucleotide '{triseq}' starting at residue {first_atom['resid']}: 
Incorrect number of atoms {tri_natoms}, while template has {len(template)} atoms"""
            )
        elif tri_inds_offset != template_inds[triseq]:
            if ignore_missing:
                continue
            raise ValueError(
                f"""Trinucleotide '{triseq}' starting at residue {first_atom['resid']}: 
Trinucleotide has layout {tri_inds_offset}, while template has {template_inds[triseq]}"""
            )

        # Allow for out-of-order atoms
        atom_indices = []
        baseline_template = template_inds[triseq][0][0]
        baseline_ppdb = tri_inds[0][0]
        for inuc in range(3):
            t_inds = template_inds[triseq][inuc]
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
        if len(atom_indices) != tri_natoms:
            if ignore_missing:
                continue
            raise ValueError(
                f"""Trinucleotide '{triseq}' starting at residue {first_atom['resid']}: 
Trinucleotide has the same number of atoms as the template, but they are differently named"""
            )
        ppdb_trinuc = ppdb[atom_indices]
        refe = np.stack(
            (ppdb_trinuc["x"], ppdb_trinuc["y"], ppdb_trinuc["z"]), axis=-1
        ).astype(float)
        conformers = trinuc_conformer_library[triseq]
        rotmatrices, rmsds = superimpose_array(conformers, refe)

        sorting = np.argsort(rmsds)
        sorted_indices = np.arange(len(rmsds), dtype=np.uint16)[sorting]
        rmsds = rmsds[sorting]
        rotmatrices = rotmatrices[sorting]

        best_rmsd = rmsds.min()
        max_rmsd = np.sqrt(best_rmsd**2 + rmsd_margin**2)

        best_conformers0 = np.where(rmsds < max_rmsd)[0]
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

        first_resid = first_atom["resid"]
        results0.append(
            (first_resid, triseq, best_conformers, best_matrices, best_rmsds)
        )

    nresults = sum([len(result0[2]) for result0 in results0])
    results = np.empty(nresults, dtype=trinuc_dtype)
    pos = 0
    for result0 in results0:
        first_resid, triseq, best_conformers, best_matrices, best_rmsds = result0
        for n in range(len(best_conformers)):
            result = results[pos]
            result["first_resid"] = first_resid
            result["sequence"] = triseq
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
                        Pattern must contain XXX, which will be replaced for each trinucleotide sequence.
                        Each file must contain a numpy array of a parsed PDB, corresponding to that trinucleotide
                        """,
        required=True,
    )
    parser.add_argument(
        "--conformers",
        help="""Conformer file pattern.
                        Pattern must contain XXX, which will be replaced for each trinucleotide sequence.
                        Each file must contain a numpy array of (nconf, natoms, 3) numbers.
                        """,
        required=True,
    )

    parser.add_argument(
        "--output",
        help="Output file for resulting array of trinucleotide fitted conformers",
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
        help="Ignore nucleotides with missing heavy atoms",
    )

    parser.add_argument(
        "--margin",
        default=1.6,
        help="""Store not only the best-fitting conformer, but all conformers up to 'margin' beyond that.
        To be precise, a conformer is kept if its fitting RMSD is smaller than sqrt(best-fitting-RMSD**2 + margin**2)""",
    )

    args = parser.parse_args()
    if args.dna == args.rna:
        err("Specify --dna OR --rna")

    bases = ("A", "C", "G", "T") if args.dna else ("A", "C", "G", "U")
    trinuc_sequences = ["".join(s) for s in itertools.product(bases, repeat=3)]

    ppdb_file = args.ppdb
    ppdb = np.load(ppdb_file)
    if ppdb.dtype != atomic_dtype:
        err(f"'{ppdb_file }' does not contain a parsed PDB")

    if len(np.unique(ppdb["chain"])) > 1:
        err(f"'{ppdb_file }' contains multiple chains")

    templates = {}
    for seq in trinuc_sequences:
        filename = args.templates.replace("XXX", seq)
        template = np.load(filename)
        if template.dtype != atomic_dtype:
            err(f"Template '{filename}' does not contain a parsed PDB")
        templates[seq] = template

    conformers = {}
    for seq in trinuc_sequences:
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
        trinuc_conformer_library=conformers,
        rna=args.rna,
        ignore_unknown=args.ignore_unknown,
        ignore_missing=args.ignore_missing,
        rmsd_margin=args.margin,
    )

    np.save(args.output, result)


if __name__ == "__main__":
    main()
