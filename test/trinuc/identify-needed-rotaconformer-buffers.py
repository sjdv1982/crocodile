import itertools
import os
import sys

import numpy as np

from nefertiti.functions.parse_pdb import atomic_dtype
from crocodile.trinuc import trinuc_dtype

from seamless import Buffer, Checksum


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
        "--rotaconformers",
        help="""Rotaconformer file pattern.
                        Pattern must contain XXX, which will be replaced for each trinucleotide sequence.
                        Each file must contain a list of checksums, one checksum per conformer. 
                        Each checksum corresponds to the array of (3, 3) rotation matrices for that conformer
                        """,
        required=True,
    )

    parser.add_argument(
        "--output",
        help="Output directory for .INDEX files",
        required=True,
    )

    parser.add_argument(
        "--rna", action="store_true", help="Interpret the parsed PDB as RNA"
    )
    parser.add_argument(
        "--dna", action="store_true", help="Interpret the parsed PDB as DNA"
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

    rotaconformer_indices = {}
    for seq in trinuc_sequences:
        seq2 = seq.replace("U", "C").replace("G", "A")
        if seq == seq2:
            filename = args.rotaconformers.replace("XXX", seq)
            rotaconformer_index = Buffer.load(filename).deserialize("plain")
            rotaconformer_indices[seq] = rotaconformer_index
    for seq in trinuc_sequences:
        seq2 = seq.replace("U", "C").replace("G", "A")
        rotaconformer_indices[seq] = rotaconformer_indices[seq2]

    needed_rotamers = set()
    for fitted_trinuc in trinuc_array:
        seq = fitted_trinuc["sequence"].decode()
        conformer = int(fitted_trinuc["conformer"])
        needed_rotamers.add((seq, conformer))
    needed_buffers = {
        rotaconformer_indices[seq][conformer] for seq, conformer in needed_rotamers
    }

    for cs in needed_buffers:
        fname = os.path.join(args.output, cs)
        cs = Checksum(cs)
        cs.save(fname)


if __name__ == "__main__":
    main()
