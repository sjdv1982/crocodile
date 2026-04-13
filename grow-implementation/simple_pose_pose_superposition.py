#!/usr/bin/env python3
"""Simple pose-pose superposition implementation.

The simple implementation is based on the reference implementation, but with some very simple optimizations.
The basic principle is to use "SD math". Knowing the number of atoms, the RMSD can be converted to an SD.
The SD then has the following sources:
- Conformational SD (the cRMSD converted to SD). Kabsch superposition of the conformers.
- Rotational SD. Based on the difference between the rotamer and the Kabsch-perfect rotation.
These two sources can be captured together in a "Rotaconformer SD".
This is computed as the rotamer matrix applied to the centered X-Y pose minus the centered D-E pose.
- Grid discretization SD. This is the squared sum of the M-MM vector snapped to gridspacing.
  This is the translational SD of the optimal translation.
- Grid-translational SD . This is additional pose translation SD caused by non-optimal translation.

First, a precomputation step. Choose a series of thresholds T, starting from 0, incrementing by one gridspacing,
and continuing one value beyond 1.3 .
For each T, pre-build a grid of translations, where xyz each range from -T/gridspacing to T/gridspacing,
capture the set of all elements that are under T, and extrude this set by one grid unit in all directions.
These are the translation sets, each set stored under T.

Also, calculate the maximum possible grid discretization SD, which is the case of half a grid unit in all directions.

Then, the simple algorithm does the following steps:
- Load the fragment library for AB (or BC if backward).
  Obtain the pose coordinates for the pose D-E, selecting only nucleotide B.
  Compute the mean coordinate M for the pose, *and subtract this from the pose*.
- Load the cRMSD matrix ABC and choose the row (or column if backward) for D.
  Apply "cRMSD < F" as a mask, and do np.where.
  This is the list of potential conformers for BC (or AB if backward).
  Load that conformer library after unloading the old one (if AB != BC).
- Iterate over each conformer X in the list:
    - Compute the "remaining" SD: the total overlap SD threshold minus the conformer SD
    - Take the coordinates of he unrotated conformer, selecting only nucleotide B.
      Calculate the mean coordinate MM0 *and subtract this from the conformer*
    - Load its rotamers. Apply the rotation matrices to MM0 to obtain MM.
      Calculate the grid discretization SD.
      Reject rotamers where this is larger than the "remaining" SD.
      This rejection test is only useful if the "remaining" SD is smaller than the maximum possible grid discretization SD.
    - Calculate a "remaining2" SD as the total overlap SD trheshold minus the grid discretization SD

    - Obtain the (centered!) pose coordinates for the pose X-Y for each (remaining) rotamer Y.
    - For each pose, calculate the rotaconformer SD.
      Reject rotamers where this is larger than the "remaining2" SD.
    - Calculate a "remaining3" SD as "remaining2" SD minus the rotaconformer SD
    - Also calculate a "remaining4" SD as the total overlap SD minus the rotaconformer SD
    - Select the appropriate translation set where T goes just beyond sqrt("remaining3"/natoms).
    - Apply all translations in the set (adding the optimal superposition), but apply it only to MM.
      Use this to to calculate the actual translation SD, and keep all points lower than "remaining4"
For each test case, store the poses in a pose dir.
"""

from __future__ import annotations

import argparse
import csv
import gc
import shutil
import sys
from dataclasses import dataclass
from math import ceil, sqrt
from pathlib import Path
from typing import Any

import numpy as np
from scipy.spatial.transform import Rotation


GRID_SPACING = sqrt(3.0) / 3.0
MAX_OVERLAP_RMSD = 1.3
SAMPLES_CSV = Path(__file__).with_name("pose_pose_samples.csv")
DEFAULT_OUTPUT_ROOT = Path(__file__).with_name("simple_pose_dirs")
DEFAULT_SUMMARY_CSV = Path(__file__).with_name("simple_pose_pose_summary.csv")
ROTAMER_BATCH_SIZE = 4096
RMSD_BOUNDARY_TOLERANCE = 2e-6


@dataclass
class Case:
    sample_id: int
    ab: str
    bc: str
    direction: str
    source_sequence: str
    target_sequence: str
    source_mask: tuple[bool, bool]
    target_mask: tuple[bool, bool]
    conformer: int
    rotaconformer: int
    crmsd_threshold: float
    overlap_rmsd: float
    reference_centered: np.ndarray | None = None
    reference_mean: np.ndarray | None = None
    candidate_conformers: np.ndarray | None = None
    candidate_crmsds: np.ndarray | None = None
    pose_count: int = 0


@dataclass(frozen=True)
class TranslationSets:
    thresholds: np.ndarray
    offsets: tuple[np.ndarray, ...]
    max_grid_discretization_sd_per_atom: float


def _bootstrap_code_path() -> Path:
    repo_root = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(repo_root / "code"))
    return repo_root


def _library_directory():
    repo_root = _bootstrap_code_path()
    from library import (
        LibraryDirectory,
        _fraglib_config,
        _require_int,
        _require_str,
        _resolve_templates_dir,
    )

    config_data = _fraglib_config()
    fraglib_root = repo_root / "fraglib"
    templates_dir = _resolve_templates_dir(config_data, fraglib_root)
    directory = LibraryDirectory(
        fraglen=_require_int(config_data, "fraglen"),
        filepattern=_require_str(config_data, "conformers"),
        replacement_filepattern=_require_str(config_data, "conformer_replacements"),
        replacement_origin_filepattern=_require_str(
            config_data, "conformer_replacement_origins"
        ),
        extension_filepattern=_require_str(config_data, "conformer_extensions"),
        extension_origin_filepattern=_require_str(
            config_data, "conformer_extension_origins"
        ),
        rotaconformers_filepattern=_require_str(config_data, "rotamers"),
        rotaconformers_index_filepattern=_require_str(config_data, "rotamers_indices"),
        rotaconformers_extension_filepattern=_require_str(
            config_data, "rotamer_extensions"
        ),
        rotaconformers_extension_index_filepattern=_require_str(
            config_data, "rotamer_extension_indices"
        ),
    )
    return directory, templates_dir


def _load_library(
    directory: Any,
    templates_dir: Path,
    sequence: str,
    nucleotide_mask: tuple[bool, bool],
):
    template = np.load(str(templates_dir / f"{sequence}-ppdb.npy"))
    factory = directory.load(
        sequence=sequence,
        template=template,
        with_extension=True,
        with_replacement=True,
        with_rotaconformers=True,
    )
    factory.load_rotaconformers()
    library = factory.create(
        None,
        nucleotide_mask=np.array(nucleotide_mask, dtype=bool),
        with_rotaconformers=True,
    )
    return factory, library


def _rotamers_to_matrices(rotamers: np.ndarray) -> np.ndarray:
    if rotamers.ndim == 3 and rotamers.shape[1:] == (3, 3):
        return rotamers.astype(np.float32, copy=False)
    if rotamers.ndim == 2 and rotamers.shape[1] == 3:
        return Rotation.from_rotvec(rotamers).as_matrix().astype(np.float32, copy=False)
    raise ValueError(
        "Unsupported rotamer representation; expected shape [N,3] rotvec or [N,3,3] matrix"
    )


def _read_cases(limit: int | None, sample_id: int | None) -> list[Case]:
    cases: list[Case] = []
    with SAMPLES_CSV.open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    if sample_id is not None:
        rows = [row for row in rows if int(row["sample_id"]) == sample_id]
        if not rows:
            raise ValueError(f"No sample row found with sample_id {sample_id}")
    if limit is not None:
        rows = rows[:limit]
    for row in rows:
        ab = row["ab"]
        bc = row["bc"]
        direction = row["direction"]
        if direction == "forward":
            source_sequence = ab
            target_sequence = bc
            source_mask = (False, True)
            target_mask = (True, False)
        elif direction == "backward":
            source_sequence = bc
            target_sequence = ab
            source_mask = (True, False)
            target_mask = (False, True)
        else:
            raise ValueError(f"Unknown direction {direction!r}")
        cases.append(
            Case(
                sample_id=int(row["sample_id"]),
                ab=ab,
                bc=bc,
                direction=direction,
                source_sequence=source_sequence,
                target_sequence=target_sequence,
                source_mask=source_mask,
                target_mask=target_mask,
                conformer=int(row["conformer"]),
                rotaconformer=int(row["rotaconformer"]),
                crmsd_threshold=float(row["crmsd_threshold"]),
                overlap_rmsd=float(row["overlap_rmsd"]),
            )
        )
    return cases


def _precompute_translation_sets() -> TranslationSets:
    thresholds = []
    offsets = []
    threshold = 0.0
    while threshold <= MAX_OVERLAP_RMSD:
        thresholds.append(threshold)
        threshold += GRID_SPACING
    thresholds.append(threshold)

    extrusion = np.stack(
        np.meshgrid(
            np.arange(-1, 2, dtype=np.int16),
            np.arange(-1, 2, dtype=np.int16),
            np.arange(-1, 2, dtype=np.int16),
            indexing="ij",
        ),
        axis=-1,
    ).reshape(-1, 3)

    for threshold in thresholds:
        n = int(ceil(threshold / GRID_SPACING))
        axis = np.arange(-n, n + 1, dtype=np.int16)
        grid = np.stack(np.meshgrid(axis, axis, axis, indexing="ij"), axis=-1).reshape(
            -1, 3
        )
        if threshold == 0.0:
            base = grid[np.zeros(len(grid), dtype=bool)]
        else:
            distances = np.linalg.norm(grid.astype(np.float32) * GRID_SPACING, axis=1)
            base = grid[distances < threshold]
        if len(base) == 0:
            extruded = np.empty((0, 3), dtype=np.int16)
        else:
            candidates = (base[:, None, :] + extrusion[None, :, :]).reshape(-1, 3)
            extruded = np.unique(candidates, axis=0).astype(np.int16, copy=False)
        offsets.append(extruded)

    max_grid_discretization_sd_per_atom = 3 * (0.5 * GRID_SPACING) ** 2
    return TranslationSets(
        thresholds=np.asarray(thresholds, dtype=np.float32),
        offsets=tuple(offsets),
        max_grid_discretization_sd_per_atom=max_grid_discretization_sd_per_atom,
    )


def _select_translation_set_index(
    translation_sets: TranslationSets,
    remaining_sd: np.ndarray,
    natoms: int,
) -> np.ndarray:
    radii = np.sqrt(np.maximum(remaining_sd, 0.0) / natoms)
    indices = np.searchsorted(translation_sets.thresholds, radii, side="right")
    return np.minimum(indices, len(translation_sets.offsets) - 1)


def _compute_reference_poses(
    cases: list[Case],
    directory: Any,
    templates_dir: Path,
) -> None:
    for sequence in sorted({case.source_sequence for case in cases}):
        sequence_cases = [case for case in cases if case.source_sequence == sequence]
        for mask in sorted({case.source_mask for case in sequence_cases}):
            factory, library = _load_library(directory, templates_dir, sequence, mask)
            try:
                for case in sequence_cases:
                    if case.source_mask != mask:
                        continue
                    if case.conformer >= len(library.coordinates):
                        raise ValueError(
                            f"Case {case.sample_id}: conformer {case.conformer} "
                            f"out of range for {sequence}"
                        )
                    coords = library.coordinates[case.conformer].astype(
                        np.float32, copy=False
                    )
                    rotamers = _rotamers_to_matrices(
                        library.get_rotamers(case.conformer)
                    )
                    if case.rotaconformer >= len(rotamers):
                        raise ValueError(
                            f"Case {case.sample_id}: rotaconformer {case.rotaconformer} "
                            f"out of range for conformer {case.conformer}"
                        )
                    reference_coords = coords @ rotamers[case.rotaconformer]
                    reference_mean = reference_coords.mean(axis=0)
                    case.reference_mean = reference_mean
                    case.reference_centered = reference_coords - reference_mean
            finally:
                del library
                factory.unload_rotaconformers()
                del factory
                gc.collect()


def _compute_candidate_conformers(cases: list[Case]) -> None:
    from library import load_crmsds

    for case in cases:
        crmsds = load_crmsds(case.ab, case.bc)
        if case.direction == "forward":
            values = crmsds[case.conformer, :]
        else:
            values = crmsds[:, case.conformer]
        candidate_mask = values < case.crmsd_threshold
        case.candidate_conformers = np.where(candidate_mask)[0].astype(
            np.int64,
            copy=False,
        )
        case.candidate_crmsds = values[candidate_mask].astype(np.float32, copy=True)
        del crmsds
        gc.collect()


def _write_case_poses(
    case: Case,
    library: Any,
    outdir: Path,
    translation_sets: TranslationSets,
) -> None:
    from poses import PoseStreamAccumulator

    if case.reference_centered is None or case.reference_mean is None:
        raise RuntimeError(f"Case {case.sample_id}: reference pose was not computed")
    if case.candidate_conformers is None or case.candidate_crmsds is None:
        raise RuntimeError(f"Case {case.sample_id}: candidate conformers not computed")

    reference_centered = case.reference_centered.astype(np.float32, copy=False)
    reference_mean = case.reference_mean.astype(np.float32, copy=False)
    reference_coords = reference_centered + reference_mean
    natoms = reference_centered.shape[0]
    total_overlap_sd = natoms * case.overlap_rmsd * case.overlap_rmsd
    max_grid_discretization_sd = (
        natoms * translation_sets.max_grid_discretization_sd_per_atom
    )

    writer = PoseStreamAccumulator(outdir)
    try:
        for conformer, crmsd in zip(case.candidate_conformers, case.candidate_crmsds):
            conformer = int(conformer)
            conformer_sd = natoms * float(crmsd) * float(crmsd)
            remaining = total_overlap_sd - conformer_sd
            if remaining <= 0.0:
                continue

            coords = library.coordinates[conformer].astype(np.float32, copy=False)
            mean0 = coords.mean(axis=0)
            centered = coords - mean0
            rotamers = _rotamers_to_matrices(library.get_rotamers(conformer))
            if len(rotamers) == 0:
                continue

            for rotamer_start in range(0, len(rotamers), ROTAMER_BATCH_SIZE):
                rotamer_stop = min(rotamer_start + ROTAMER_BATCH_SIZE, len(rotamers))
                rotamer_block = rotamers[rotamer_start:rotamer_stop]
                rotamer_indices = np.arange(rotamer_start, rotamer_stop, dtype=np.int64)

                means = mean0 @ rotamer_block
                continuous_translation = reference_mean[None, :] - means
                best_grid = np.rint(continuous_translation / GRID_SPACING).astype(
                    np.int32
                )
                best_world = best_grid.astype(np.float32) * GRID_SPACING
                delta = best_world - continuous_translation
                grid_discretization_sd = natoms * np.einsum("ij,ij->i", delta, delta)

                if remaining < max_grid_discretization_sd:
                    grid_keep = np.nonzero(grid_discretization_sd <= remaining)[0]
                else:
                    grid_keep = np.arange(len(rotamer_block), dtype=np.int64)
                if grid_keep.size == 0:
                    continue

                remaining2 = total_overlap_sd - grid_discretization_sd[grid_keep]
                centered_poses = np.einsum(
                    "aj,njk->nak",
                    centered,
                    rotamer_block[grid_keep],
                )
                centered_dif = centered_poses - reference_centered[None, :, :]
                rotaconformer_sd = np.einsum("nij,nij->n", centered_dif, centered_dif)
                rot_keep = np.nonzero(rotaconformer_sd < remaining2)[0]
                if rot_keep.size == 0:
                    continue

                kept_grid = best_grid[grid_keep][rot_keep]
                kept_delta = delta[grid_keep][rot_keep]
                kept_rotamer_indices = rotamer_indices[grid_keep][rot_keep]
                kept_rotamers = rotamer_block[grid_keep][rot_keep]
                kept_rotaconformer_sd = rotaconformer_sd[rot_keep]
                kept_remaining2 = remaining2[rot_keep]
                remaining3 = kept_remaining2 - kept_rotaconformer_sd
                remaining4 = total_overlap_sd - kept_rotaconformer_sd

                set_indices = _select_translation_set_index(
                    translation_sets,
                    remaining3,
                    natoms,
                )
                set_order = np.argsort(set_indices, kind="stable")
                ordered_set_indices = set_indices[set_order]
                set_starts = np.concatenate(
                    (
                        np.array([0], dtype=np.int64),
                        np.flatnonzero(
                            ordered_set_indices[1:] != ordered_set_indices[:-1]
                        )
                        + 1,
                    )
                )
                set_stops = np.concatenate(
                    (set_starts[1:], np.array([len(set_order)], dtype=np.int64))
                )
                for set_start, set_stop in zip(set_starts, set_stops):
                    set_index = ordered_set_indices[set_start]
                    translation_offsets = translation_sets.offsets[int(set_index)]
                    if len(translation_offsets) == 0:
                        continue
                    local_rows = set_order[set_start:set_stop]
                    translation_offsets32 = translation_offsets.astype(
                        np.float32,
                        copy=False,
                    )
                    shifted_delta = (
                        kept_delta[local_rows, None, :]
                        + translation_offsets32[None, :, :] * GRID_SPACING
                    )
                    translation_sd = natoms * np.einsum(
                        "rgj,rgj->rg", shifted_delta, shifted_delta
                    )
                    keep = translation_sd < remaining4[local_rows, None]
                    combined_sd = (
                        translation_sd + kept_rotaconformer_sd[local_rows, None]
                    )
                    boundary = (
                        np.abs(np.sqrt(combined_sd / natoms) - case.overlap_rmsd)
                        <= RMSD_BOUNDARY_TOLERANCE
                    )
                    if np.any(boundary):
                        boundary_rotamer_rows, boundary_translation_rows = np.nonzero(
                            boundary
                        )
                        boundary_local_rows = local_rows[boundary_rotamer_rows]
                        boundary_translations = (
                            kept_grid[boundary_local_rows]
                            + translation_offsets[
                                boundary_translation_rows
                            ].astype(np.int32)
                        )
                        boundary_poses = np.einsum(
                            "aj,njk->nak",
                            coords,
                            kept_rotamers[boundary_local_rows],
                        )
                        boundary_world = (
                            boundary_translations.astype(np.float32) * GRID_SPACING
                        )
                        boundary_shifted = (
                            boundary_poses + boundary_world[:, None, :]
                        )
                        boundary_dif = boundary_shifted - reference_coords[None, :, :]
                        boundary_rmsd = np.sqrt(
                            np.einsum("nij,nij->n", boundary_dif, boundary_dif)
                            / natoms
                        )
                        keep[boundary] = False
                        keep[boundary_rotamer_rows, boundary_translation_rows] = (
                            boundary_rmsd < case.overlap_rmsd
                        )
                    rotamer_subrows, translation_rows = np.nonzero(keep)
                    if rotamer_subrows.size == 0:
                        continue

                    translations = (
                        kept_grid[local_rows[rotamer_subrows]]
                        + translation_offsets[translation_rows].astype(np.int32)
                    )
                    if translations.size and (
                        translations.min() < np.iinfo(np.int16).min
                        or translations.max() > np.iinfo(np.int16).max
                    ):
                        raise ValueError(
                            f"Case {case.sample_id}: translation exceeds int16 range"
                        )
                    conformers = np.full(
                        len(translations),
                        conformer,
                        dtype=np.uint16,
                    )
                    rotamer_indices_chunk = kept_rotamer_indices[
                        local_rows[rotamer_subrows]
                    ].astype(
                        np.uint16,
                        copy=False,
                    )
                    if len(rotamer_indices_chunk) != len(translations):
                        raise RuntimeError(
                            f"Case {case.sample_id}: mismatched rotamer/translation chunk"
                        )
                    writer.add_chunk(
                        conformers,
                        rotamer_indices_chunk,
                        translations.astype(np.int16, copy=False),
                    )
                    del shifted_delta, translation_sd, combined_sd, keep, boundary

        writer.finish()
        case.pose_count = writer.total_poses
    finally:
        writer.cleanup()


def _write_target_poses(
    cases: list[Case],
    directory: Any,
    templates_dir: Path,
    translation_sets: TranslationSets,
    output_root: Path,
) -> None:
    for sequence in sorted({case.target_sequence for case in cases}):
        sequence_cases = [case for case in cases if case.target_sequence == sequence]
        for mask in sorted({case.target_mask for case in sequence_cases}):
            factory, library = _load_library(directory, templates_dir, sequence, mask)
            try:
                for case in sequence_cases:
                    if case.target_mask != mask:
                        continue
                    outdir = output_root / f"case-{case.sample_id:03d}"
                    _write_case_poses(case, library, outdir, translation_sets)
            finally:
                del library
                factory.unload_rotaconformers()
                del factory
                gc.collect()


def _write_summary(cases: list[Case], summary_csv: Path, output_root: Path) -> None:
    with summary_csv.open("w", newline="") as handle:
        fieldnames = [
            "sample_id",
            "ab",
            "bc",
            "direction",
            "source_sequence",
            "target_sequence",
            "conformer",
            "rotaconformer",
            "crmsd_threshold",
            "overlap_rmsd",
            "candidate_conformers",
            "pose_count",
            "pose_dir",
        ]
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for case in cases:
            candidates = case.candidate_conformers
            writer.writerow(
                {
                    "sample_id": case.sample_id,
                    "ab": case.ab,
                    "bc": case.bc,
                    "direction": case.direction,
                    "source_sequence": case.source_sequence,
                    "target_sequence": case.target_sequence,
                    "conformer": case.conformer,
                    "rotaconformer": case.rotaconformer,
                    "crmsd_threshold": f"{case.crmsd_threshold:.4f}",
                    "overlap_rmsd": f"{case.overlap_rmsd:.4f}",
                    "candidate_conformers": 0
                    if candidates is None
                    else len(candidates),
                    "pose_count": case.pose_count,
                    "pose_dir": str((output_root / f"case-{case.sample_id:03d}").resolve()),
                }
            )


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the simple SD-math pose-pose superposition implementation."
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Only process the first N sample rows.",
    )
    parser.add_argument(
        "--sample-id",
        type=int,
        default=None,
        help="Only process the sample row with this sample_id.",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=DEFAULT_OUTPUT_ROOT,
        help="Directory where case pose dirs are written.",
    )
    parser.add_argument(
        "--summary",
        type=Path,
        default=DEFAULT_SUMMARY_CSV,
        help="Summary CSV path.",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    if args.limit is not None and args.sample_id is not None:
        raise ValueError("--limit and --sample-id cannot be used together")
    cases = _read_cases(args.limit, args.sample_id)
    if args.sample_id is None:
        if args.output_root.exists():
            shutil.rmtree(args.output_root)
        args.output_root.mkdir(parents=True)
    else:
        args.output_root.mkdir(parents=True, exist_ok=True)
        case_outdir = args.output_root / f"case-{args.sample_id:03d}"
        if case_outdir.exists():
            shutil.rmtree(case_outdir)

    translation_sets = _precompute_translation_sets()
    directory, templates_dir = _library_directory()
    _compute_reference_poses(cases, directory, templates_dir)
    _compute_candidate_conformers(cases)
    _write_target_poses(cases, directory, templates_dir, translation_sets, args.output_root)
    _write_summary(cases, args.summary, args.output_root)


if __name__ == "__main__":
    main()
