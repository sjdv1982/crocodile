import itertools
import sys
import numpy as np
from nefertiti.functions.parse_pdb import atomic_dtype
from crocodile.trinuc import trinuc_dtype, trinuc_roco_dtype


def err(msg):
    print(msg, file=sys.stderr)
    exit(1)


def get_overlap_rmsd(
    trinuc_array: np.ndarray,
    *,
    template_pdbs: dict[str, np.ndarray],
    trinuc_conformer_library: dict[str, np.ndarray],
    rna: bool,
):
    from crocodile.trinuc.from_ppdb import ppdb2nucseq

    if len(np.unique(trinuc_array["first_resid"])) > len(trinuc_array):
        raise ValueError(
            """Trinucleotide array contains more than one fitting per fragment.
                         Running discretize_grid will also select the best fitting."""
        )

    preatoms = {}
    postatoms = {}
    for seq, tmpl in template_pdbs.items():
        _, resind = ppdb2nucseq(tmpl, rna=rna, ignore_unknown=False, return_index=True)
        assert (
            len(resind) == 3
            and resind[1][1] == resind[2][0]
            and resind[0][1] == resind[1][0]
        )
        preatoms[seq] = len(tmpl) - resind[0][1]
        postatoms[seq] = resind[1][1]

    coordinates = {}
    for trinuc in trinuc_array:
        seq = trinuc["sequence"].decode()
        conf = trinuc_conformer_library[seq][trinuc["conformer"]]
        coors = conf.dot(trinuc["rotation_matrix"]) + trinuc["offset"]
        coordinates[trinuc["first_resid"]] = coors

    trinucs = {trinuc["first_resid"]: trinuc for trinuc in trinuc_array}
    result_dtype = np.dtype(
        [
            ("first_resid", np.uint16),
            ("fit1", np.float32),
            ("fit2", np.float32),
            ("overlap_rmsd", np.float32),
        ]
    )
    results = np.empty(len(trinucs), result_dtype)
    nresults = 0
    for resid, trinuc in trinucs.items():
        seq = trinuc["sequence"].decode()
        next_trinuc = trinucs.get(resid + 1)
        if next_trinuc is None:
            continue
        next_seq = next_trinuc["sequence"].decode()

        coors, next_coors = coordinates[resid], coordinates[resid + 1]
        pre_coors = coors[-preatoms[seq] :]
        post_coors = next_coors[: postatoms[next_seq]]

        assert len(pre_coors) == len(post_coors), resid
        overlap_rmsd = np.sqrt(((pre_coors - post_coors) ** 2).sum(axis=1).mean(axis=0))
        result = results[nresults]
        result["first_resid"] = resid
        result["fit1"] = trinuc["rmsd"]
        result["fit2"] = next_trinuc["rmsd"]
        result["overlap_rmsd"] = overlap_rmsd
        nresults += 1
    return results[:nresults]


def main():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "trinuc",
        help="A trinucleotide fitted array of rotaconformers on a grid, in numpy format",
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
        "--rna", action="store_true", help="Interpret the trinucleotides as RNA"
    )
    parser.add_argument(
        "--dna", action="store_true", help="Interpret the trinucleotides as DNA"
    )

    args = parser.parse_args()
    if args.dna == args.rna:
        err("Specify --dna OR --rna")

    bases = ("A", "C", "G", "T") if args.dna else ("A", "C", "G", "U")
    trinuc_sequences = ["".join(s) for s in itertools.product(bases, repeat=3)]

    trinuc_array_file = args.trinuc
    trinuc_array = np.load(trinuc_array_file)
    if trinuc_array.dtype != trinuc_roco_dtype:
        if trinuc_array.dtype == trinuc_dtype:
            err(
                f"'{trinuc_array_file}' does not contain rotaconformers, first use fit_roco and discretize_grid"
            )
        else:
            err(
                f"'{trinuc_array_file}' does not contain a trinucleotide rotaconformer fitted array"
            )

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

    results = get_overlap_rmsd(
        trinuc_array,
        template_pdbs=templates,
        trinuc_conformer_library=conformers,
        rna=args.rna,
    )

    print("#first_resid #fit1 #fit2 #overlap_rmsd")
    for result in results:
        print(
            result["first_resid"],
            result["fit1"],
            result["fit2"],
            result["overlap_rmsd"],
        )


if __name__ == "__main__":
    main()
