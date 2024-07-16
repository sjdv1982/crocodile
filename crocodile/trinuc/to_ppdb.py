import itertools
import sys
import numpy as np
from nefertiti.functions.parse_pdb import atomic_dtype
from crocodile.trinuc import trinuc_dtype

def err(msg):
    print(msg, file=sys.stderr)
    exit(1)

def to_ppdb(
    trinuc_array: np.ndarray,
    *,
    template_pdbs: dict[str, np.ndarray],
    trinuc_conformer_library: dict[str, np.ndarray],
    rna: bool
):
    from crocodile.trinuc.from_ppdb import ppdb2nucseq

    resinds = {}
    for seq, tmpl in template_pdbs.items():
        _, resind = ppdb2nucseq(tmpl, rna=rna, ignore_unknown=False, return_index=True)
        resinds[seq] = resind

    result0 = {}
    counts = {}

    for trinuc in trinuc_array:
        first_resid = trinuc["first_resid"]
        if first_resid not in counts:
            counts[first_resid] = 0
        counts[first_resid] += 1
        baseline_resid = 10 * (first_resid - 1) + 1
        if counts[first_resid] <= 26:
            chain = chr(ord('A') + counts[first_resid] - 1)
        else:
            chain = chr(ord('a') + counts[first_resid] - 1 - 26)
        seq = trinuc["sequence"].decode()
        tmpl = template_pdbs[seq].copy()
        tmpl["chain"] = chain
        for nuc in range(3):
            nuc_start, nuc_end = resinds[seq][nuc]
            tmpl[nuc_start:nuc_end]["resid"] = baseline_resid + nuc
        conf = trinuc_conformer_library[seq][trinuc["conformer"]]
        conf4 = np.concatenate((conf, np.ones(len(conf))[:, None]), axis=1)
        coors = conf4.dot(trinuc["matrix"])[:, :3]
        tmpl["x"] = coors[:, 0]
        tmpl["y"] = coors[:, 1]
        tmpl["z"] = coors[:, 2]
        if chain not in result0:
            result0[chain] = []
        result0[chain].append(tmpl)

    subresults = []
    for chain in sorted(result0.keys()):
        subresult = np.concatenate(result0[chain], dtype=atomic_dtype)
        subresults.append(subresult)
    result = np.concatenate(subresults, dtype=atomic_dtype)
    result["index"] = np.arange(len(result)) + 1
    return result

def main():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "trinuc",
        help="A trinucleotide fitted array of conformers, translated rotaconformers or rotaconformers on a grid, in numpy format",
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

    # TODO: rotaconformers (optional), grid spacing (optional)

    parser.add_argument(
        "--output",
        help="Output file for the result, in parsed PDB numpy format",
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
    if trinuc_array.dtype != trinuc_dtype:
        # TODO: rotaconf+trans, rotaconf+grid
        err(f"'{trinuc_array_file}' does not contain a trinucleotide fitted array")

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

    result = to_ppdb(
        trinuc_array,
        template_pdbs=templates,
        trinuc_conformer_library=conformers,
        rna=args.rna
    )

    np.save(args.output, result)


if __name__ == "__main__":
    main()
