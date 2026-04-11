#!/usr/bin/env python3
from __future__ import annotations

import argparse
from math import sqrt
from pathlib import Path

import numpy as np

from poses import pack_expanded_poses, write_pose_files
from rmsd_poses import (
    _dinucleotide_sequence,
    _existing_pdb_file,
    _load_library,
    _pdb_code,
    _load_reference,
    _rotamers_to_matrices,
)
from superimpose import superimpose_array

GRID_SPACING = sqrt(3) / 3


def _nonexistent_path(path: str) -> str:
    p = Path(path)
    if p.exists():
        raise argparse.ArgumentTypeError(f"path already exists: {path}")
    return path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="write_best_fit_pose",
        description=(
            "Write poses-1.npy and offsets-1.dat for the single best-fit pose "
            "against a reference PDB."
        ),
    )
    parser.add_argument(
        "--sequence",
        required=True,
        type=_dinucleotide_sequence,
        help="Dinucleotide sequence (AA/AC/.../UU)",
    )
    parser.add_argument(
        "--reference",
        required=True,
        type=_existing_pdb_file,
        help="Reference PDB file whose topology must match the library template",
    )
    parser.add_argument(
        "--output-dir",
        required=True,
        type=_nonexistent_path,
        help="Output directory to create; will contain poses-1.npy and offsets-1.dat",
    )
    parser.add_argument(
        "--verify-checksums",
        action="store_true",
        help="Enable fraglib checksum verification when loading the library config",
    )
    parser.add_argument(
        "--pdb-exclude",
        nargs="+",
        default=[],
        type=_pdb_code,
        metavar="PDB",
        help="One or more PDB codes to exclude from the fragment library.",
    )
    return parser


def _grid_round_translation(translation_world: np.ndarray) -> np.ndarray:
    translation_grid = np.round(translation_world / GRID_SPACING)
    if translation_grid.min() < np.iinfo(np.int16).min or translation_grid.max() > np.iinfo(
        np.int16
    ).max:
        raise ValueError("Best-fit translation exceeds int16 grid range")
    return translation_grid.astype(np.int16)


def _rmsd_to_reference(
    transformed: np.ndarray,
    reference_coords: np.ndarray,
) -> np.ndarray:
    dif = transformed - reference_coords
    return np.sqrt(np.einsum("...ij,...ij->...", dif, dif) / len(reference_coords))


def select_best_pose(
    sequence: str,
    reference_path: Path,
    verify_checksums: bool,
    excluded_pdb_codes: set[str],
) -> tuple[tuple[np.ndarray, np.ndarray, np.ndarray], np.ndarray, float, float, float]:
    lib = _load_library(
        sequence,
        verify_checksums=verify_checksums,
        excluded_pdb_codes=excluded_pdb_codes,
    )
    coordinates = lib.coordinates.astype(np.float32, copy=False)
    reference_coords = _load_reference(reference_path, lib.template)

    _, conformer_rmsd = superimpose_array(coordinates, reference_coords)
    best_conformer = int(np.argmin(conformer_rmsd))
    best_kabsch_rmsd = float(conformer_rmsd[best_conformer])
    if best_conformer > np.iinfo(np.uint16).max:
        raise ValueError("Best conformer index exceeds uint16 range")

    conformer_coords = coordinates[best_conformer]
    rotamers = _rotamers_to_matrices(lib.get_rotamers(best_conformer))
    if len(rotamers) == 0:
        raise ValueError(f"Conformer {best_conformer} has no rotamers")

    superpositions = np.einsum("aj,njk->nak", conformer_coords, rotamers)
    reference_com = reference_coords.mean(axis=0)
    centered_offsets = reference_com - superpositions.mean(axis=1)
    centered = superpositions + centered_offsets[:, None, :]
    centered_rmsd = _rmsd_to_reference(centered, reference_coords)

    best_rotamer = int(np.argmin(centered_rmsd))
    best_centered_rmsd = float(centered_rmsd[best_rotamer])
    if best_rotamer > np.iinfo(np.uint16).max:
        raise ValueError("Best rotamer index exceeds uint16 range")

    best_translation_grid = _grid_round_translation(centered_offsets[best_rotamer])
    best_translation_world = best_translation_grid.astype(np.float32) * GRID_SPACING
    final_pose = superpositions[best_rotamer] + best_translation_world[None, :]
    final_rmsd = float(_rmsd_to_reference(final_pose, reference_coords))

    pose = np.array(
        [[best_conformer, best_rotamer, 0]],
        dtype=np.uint16,
    )
    packed = pack_expanded_poses(
        pose[:, 0],
        pose[:, 1],
        best_translation_grid[None, :],
    )
    if len(packed) != 1 or len(packed[0][0]) != 1:
        raise RuntimeError("Internal error: expected a single packed pose")
    return (
        packed[0],
        best_translation_grid,
        best_kabsch_rmsd,
        best_centered_rmsd,
        final_rmsd,
    )


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    packed, translation_grid, kabsch_rmsd, centered_rmsd, final_rmsd = select_best_pose(
        sequence=args.sequence,
        reference_path=Path(args.reference),
        verify_checksums=args.verify_checksums,
        excluded_pdb_codes=set(args.pdb_exclude),
    )
    written = write_pose_files(output_dir, [packed])
    poses_path, offsets_path = written[0]
    poses, _, _ = packed
    print(f"Wrote {poses_path} and {offsets_path}")
    print(f"conformer={int(poses[0, 0])}")
    print(f"rotamer={int(poses[0, 1])}")
    print(
        "translation_grid="
        f"{int(translation_grid[0])} {int(translation_grid[1])} {int(translation_grid[2])}"
    )
    print(f"kabsch_rmsd={kabsch_rmsd:.6f}")
    print(f"centered_rmsd={centered_rmsd:.6f}")
    print(f"final_rmsd={final_rmsd:.6f}")


if __name__ == "__main__":
    main()
