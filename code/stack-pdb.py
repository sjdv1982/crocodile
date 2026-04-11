from __future__ import annotations

import argparse
from math import pi
from pathlib import Path
import sys
from typing import Sequence

import numpy as np

from parse_pdb import parse_pdb
from stack_geometry import (
    protein_ring_coordinates,
    rna_ring_coordinates,
    stacking_angle_dihedral,
)


def _existing_file(path: str) -> str:
    p = Path(path)
    if not p.exists():
        raise argparse.ArgumentTypeError(f"file does not exist: {path}")
    if not p.is_file():
        raise argparse.ArgumentTypeError(f"not a file: {path}")
    return path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="stack-pdb",
        description="Calculate stacking angle and dihedral from a protein PDB and an RNA PDB.",
    )
    parser.add_argument(
        "--protein",
        required=True,
        type=_existing_file,
        help="PDB file of the protein.",
    )
    parser.add_argument(
        "--rna",
        required=True,
        type=_existing_file,
        help="PDB file of the RNA.",
    )
    parser.add_argument(
        "--resid",
        required=True,
        type=int,
        help="Residue ID (numeric) of the protein aromatic residue that stacks.",
    )
    parser.add_argument(
        "--rna-resid",
        required=True,
        type=int,
        help="Residue ID (numeric) of the RNA base that stacks.",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Show a full traceback on errors.",
    )
    return parser


def _select_model_1_if_present(atoms: np.ndarray) -> np.ndarray:
    if np.any(atoms["model"] == 1):
        return atoms[atoms["model"] == 1]
    return atoms


def _run(args: argparse.Namespace) -> int:
    protein_atoms = parse_pdb(Path(args.protein).read_text())
    protein_atoms = _select_model_1_if_present(protein_atoms)

    rna_atoms = parse_pdb(Path(args.rna).read_text())
    rna_atoms = _select_model_1_if_present(rna_atoms)

    protein_ring = protein_ring_coordinates(protein_atoms, args.resid)
    rna_ring = rna_ring_coordinates(rna_atoms, args.rna_resid)
    angle, dihedral = stacking_angle_dihedral(protein_ring, rna_ring)
    print(f"Angle: {angle / pi * 180:.3f}  Dihedral: {dihedral / pi * 180:.3f}")
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    try:
        return int(_run(args))
    except SystemExit:
        raise
    except Exception as exc:
        if getattr(args, "debug", False):
            raise
        print(f"error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
