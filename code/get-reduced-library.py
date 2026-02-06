from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from library import config


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Load a dinucleotide library, optionally exclude one or more PDB codes, "
            "reduce coordinates, and save the reduced coordinate array."
        )
    )
    parser.add_argument(
        "sequence",
        help="Dinucleotide sequence (e.g. AA, AC, GU, ...).",
    )
    parser.add_argument(
        "output",
        type=Path,
        help="Output .npy file for reduced coordinates.",
    )
    parser.add_argument(
        "-x",
        "--exclude",
        action="append",
        default=[],
        metavar="PDB",
        help="PDB code to exclude. Can be repeated, e.g. -x 1b7f -x 3sxl.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    sequence = args.sequence.upper()
    if len(sequence) != 2:
        raise ValueError("Sequence must be a dinucleotide (length 2)")

    libraries, _ = config()
    if sequence not in libraries:
        valid = ", ".join(sorted(libraries.keys()))
        raise ValueError(f"Unknown dinucleotide sequence '{sequence}'. Valid: {valid}")

    excluded = [code.lower() for code in args.exclude]
    pdb_code: str | list[str] | None
    if not excluded:
        pdb_code = None
    elif len(excluded) == 1:
        pdb_code = excluded[0]
    else:
        pdb_code = excluded

    library = libraries[sequence].create(pdb_code)
    reduced = library.reduce()

    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.save(args.output, reduced)


if __name__ == "__main__":
    main()
