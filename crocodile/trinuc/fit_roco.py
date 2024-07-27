# Calculate the RMSD of rotaconformers with respect to a reference
# The rotamers and the reference are each centered, but not superimposed
#
# Takes as input a fitted trinucleotide conformer array as generated by from_ppdb

# Converts a parsed PDB  into a list of superimposed trinucleotides

import itertools
import os
import sys
import numpy as np
from tqdm import tqdm
from nefertiti.functions.parse_pdb import atomic_dtype
from crocodile.trinuc import trinuc_dtype, trinuc_roco_dtype
import json


def fit_roco(
    trinuc_arrays: np.ndarray,
    refe_ppdbs: np.ndarray,
    *,
    template_pdbs: dict[str, np.ndarray],
    trinuc_conformer_library: dict[str, np.ndarray],
    trinuc_rotaconformer_library: dict[(str, int), np.ndarray],
    rna: bool,
    prefilter_rmsd_min: float | None,
    prefilter_rmsd_max: float | None,
    rmsd_margin: float,
):
    """Converts an array of superimposed trinucleotides
    from conformers-with-explicit-matrices into rotaconformers-with-translations
    Returns the best-fitting rotaconformer(s) for each trinucleotide.

    trinuc_arrays: list of arrays of superimposed trinucleotides
    refe_ppdbs: list of reference parsed PDBs where the structures have been fitted against
    The two lists must be of equal length

    template_pdbs: dictionary of template parsed PDBs for each trinucleotide sequence

    trinuc_conformer_library: dictionary of numerical arrays of shape (nconf, natoms, 3),
      one for each trinucleotide sequence,
      where "natoms" is the same as the length of the template PDB for that sequence.

    trinuc_rotaconformer_library: dictionary of lists of filenames of 3x3 rotaconformer matrix arrays
      one for each trinucleotide sequence

    rna: If true, the trinucleotides are interpreted as RNA, else as DNA.

    rmsd_margin: the RMSD margin
    All rotaconformers are fitted, a rotaconformer is kept if its fitting RMSD is smaller than sqrt(best-fitting-RMSD**2 + margin**2)

    prefilter_rmsd_min: Skip all conformers that have been pre-fitted at an RMSD better than this.
        These conformers are considered artificially accurate (i.e. derived from the bound form

    prefilter_rmsd_max: Skip all fragments that have been pre-fitted at an RMSD higher than this.
        These fragments are considered to be missing from the conformer library

    NOTE: The trinuc array must have been produced with ignore_reordered. This is validated.
    NOTE: Other validation of the data must have been done before!
    """
    from crocodile.trinuc.from_ppdb import ppdb2nucseq

    if len(refe_ppdbs) != len(trinuc_arrays):
        raise ValueError(
            "The lists pfreference PDBs and the trinucleotide arrays must be equal"
        )

    template_pdbs2 = {}
    for triseq in template_pdbs:
        template = template_pdbs[triseq].copy()
        template["resid"] -= template[0]["resid"]
        template_pdbs2[triseq] = template

    all_refe_indices = []
    all_refe_coordinates = []
    for n, refe_ppdb in enumerate(refe_ppdbs):
        _, refe_indices = ppdb2nucseq(
            refe_ppdb, rna=rna, ignore_unknown=True, return_index=True
        )
        refe_indices = {refe_ppdb[ind[0]]["resid"]: ind[0] for ind in refe_indices}
        refe_coordinates = np.stack(
            (refe_ppdb["x"], refe_ppdb["y"], refe_ppdb["z"]), axis=-1
        )
        all_refe_indices.append(refe_indices)
        all_refe_coordinates.append(refe_coordinates)

    roco = {}
    for n, trinuc_array in enumerate(trinuc_arrays):
        for pos, fitted_trinuc in enumerate(trinuc_array):
            triseq = fitted_trinuc["sequence"].decode()
            triseq2 = triseq.replace("U", "C").replace("G", "A")
            conformer = int(fitted_trinuc["conformer"])
            key = (triseq2, conformer)
            if key not in roco:
                roco[key] = []
            roco[key].append((n, pos))

    results = []
    nresults = []
    for n in range(len(trinuc_arrays)):
        results.append(np.empty(100, dtype=trinuc_roco_dtype))
        nresults.append(0)

    for key in tqdm(roco):  # pylint: disable=consider-using-dict-items
        triseq2, conformer = key
        matrix_file = trinuc_rotaconformer_library[triseq2][conformer]
        matrices = np.load(matrix_file)
        for n, pos in roco[key]:
            trinuc = trinuc_arrays[n][pos]
            triseq = trinuc["sequence"].decode()
            template = template_pdbs2[triseq]
            conformer_coordinates = trinuc_conformer_library[triseq][conformer]
            if prefilter_rmsd_min is not None and trinuc["rmsd"] < prefilter_rmsd_min:
                continue
            if prefilter_rmsd_max is not None and trinuc["rmsd"] > prefilter_rmsd_max:
                continue
            refe_index = all_refe_indices[n][trinuc["first_resid"]]
            curr_refe = refe_ppdbs[n][refe_index : refe_index + len(template)]
            ok = True
            if not np.all(
                np.equal(curr_refe["resid"] - curr_refe[0]["resid"], template["resid"])
            ):
                ok = False
            elif not np.all(np.equal(curr_refe["name"], template["name"])):
                ok = False
            if not ok:
                raise ValueError(
                    f"""Trinucleotide starting at {trinuc["first_resid"]}:
                                 Wrong layout in reference PDB, did you fit with --ignore-reordered?"""
                )

            curr_refe_coordinates = all_refe_coordinates[n][
                refe_index : refe_index + len(template)
            ]
            curr_refe_com = curr_refe_coordinates.mean(axis=0)
            superpositions = np.einsum("kj,ijl->ikl", conformer_coordinates, matrices)
            offsets = curr_refe_com - superpositions.mean(axis=1)
            dif = superpositions + offsets[:, None] - curr_refe_coordinates
            rmsds = np.sqrt((dif * dif).sum(axis=2).mean(axis=1))
            best_rmsd = rmsds.min()

            max_rmsd = np.sqrt(best_rmsd**2 + rmsd_margin**2)
            mask = rmsds < max_rmsd
            nresult_new = mask.sum()
            if not nresult_new:
                continue
            result = results[n]
            nresult = nresults[n]
            if nresult + nresult_new > len(result):
                old_result = result
                newsize = max(int(1.2 * len(old_result)), nresult + nresult_new)
                results[n] = np.empty(newsize, dtype=result.dtype)
                result = results[n]
                result[:nresult] = old_result[:nresult]
            result_new = result[nresult : nresult + nresult_new]
            result_new["first_resid"] = trinuc["first_resid"]
            result_new["sequence"] = triseq
            result_new["conformer"] = conformer
            result_new["rotation_matrix"] = matrices[mask]
            result_new["rmsd"] = rmsds[mask]
            result_new["offset"] = offsets[mask]
            nresults[n] = nresult + nresult_new

    results = [result[:nresult] for result, nresult in zip(results, nresults)]
    for result in results:
        inds = result["rmsd"].argsort()
        result[:] = result[inds]
        inds = result["first_resid"].argsort(kind="stable")
        result[:] = result[inds]
    return results


def err(msg):
    print(msg, file=sys.stderr)
    exit(1)


def main():
    import argparse

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "trinuc",
        help="A numpy array of trinucleotide fitted conformers",
    )
    parser.add_argument(
        "ppdb",
        help="A numpy array of a parsed PDB, previously used to fit the trinucleotides on",
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
        "--rotaconformers",
        help="""Rotaconformer index file pattern.
                        Pattern must contain XXX, which will be replaced for each trinucleotide sequence.
                        Each file must contain a list of filenames (or checksums), one per conformer. 
                        Each filename or checksum corresponds to the array of (3, 3) rotation matrices for that conformer
                        """,
        required=True,
    )

    parser.add_argument(
        "--rotaconformer-directory",
        help="Directory to prepend to the filenames/checksums in the rotaconformer indices",
    )

    parser.add_argument(
        "--output",
        help="Output file for resulting array of trinucleotide fitted rotaconformers",
        required=True,
    )

    parser.add_argument(
        "--rna", action="store_true", help="Interpret the parsed PDB as RNA"
    )
    parser.add_argument(
        "--dna", action="store_true", help="Interpret the parsed PDB as DNA"
    )

    parser.add_argument(
        "--prefilter-min",
        type=float,
        help="""Skip all conformers that have been pre-fitted at an RMSD better than this.
        These conformers are considered artificially accurate (i.e. derived from the bound form)""",
    )

    parser.add_argument(
        "--prefilter-max",
        type=float,
        help="""Skip all fragments that have been pre-fitted at an RMSD higher than this.
        These fragments are considered to be missing from the conformer library.""",
    )

    parser.add_argument(
        "--margin",
        default=1.0,
        type=float,
        help="""Store not only the best-fitting rotaconformer, but all rotaconformers up to 'margin' beyond that.
        To be precise, a rotaconformer is kept if its fitting RMSD is smaller than sqrt(best-fitting-RMSD**2 + margin**2)""",
    )

    args = parser.parse_args()
    if args.dna == args.rna:
        err("Specify --dna OR --rna")

    bases = ("A", "C", "G", "T") if args.dna else ("A", "C", "G", "U")
    trinuc_sequences = ["".join(s) for s in itertools.product(bases, repeat=3)]

    trinuc_array_file = args.trinuc
    trinuc_array = np.load(trinuc_array_file)
    if trinuc_array.dtype != trinuc_dtype:
        err(f"'{trinuc_array_file}' does not contain a trinucleotide fitted array")

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

    rotaconformer_indices = {}
    for seq in trinuc_sequences:
        seq2 = seq.replace("U", "C").replace("G", "A")
        if seq == seq2:
            filename = args.rotaconformers.replace("XXX", seq)
            with open(filename) as fp:
                rotaconformer_index = json.load(fp)
            if not isinstance(rotaconformer_index, list):
                err(
                    f"Sequence {seq}: '{filename}' is not a list of filenames/checksums"
                )
            if len(rotaconformer_index) != len(conformers[seq]):
                err(
                    f"Sequence {seq}: There are {len(conformers[seq])} conformers but {len(rotaconformer_index)} rotaconformers"
                )
            if args.rotaconformer_directory:
                for fnr, f in list(enumerate(rotaconformer_index)):
                    rotaconformer_index[fnr] = os.path.join(
                        args.rotaconformer_directory, f
                    )
            rotaconformer_indices[seq] = rotaconformer_index
    for seq in trinuc_sequences:
        seq2 = seq.replace("U", "C").replace("G", "A")
        rotaconformer_indices[seq] = rotaconformer_indices[seq2]

    result = fit_roco(
        [trinuc_array],
        [ppdb],
        template_pdbs=templates,
        trinuc_conformer_library=conformers,
        trinuc_rotaconformer_library=rotaconformer_indices,
        rna=args.rna,
        prefilter_rmsd_min=args.prefilter_min,
        prefilter_rmsd_max=args.prefilter_max,
        rmsd_margin=args.margin,
    )[0]

    np.save(args.output, result)


if __name__ == "__main__":
    main()
