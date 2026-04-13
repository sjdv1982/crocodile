#!/usr/bin/env python3
from __future__ import annotations

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


N_CASES = 20
GRID_SPACING = sqrt(3.0) / 3.0
SAMPLES_CSV = Path(__file__).with_name("pose_pose_samples.csv")
OUTPUT_ROOT = Path(__file__).with_name("reference_pose_dirs")
SUMMARY_CSV = Path(__file__).with_name("reference_pose_pose_summary.csv")
ROTAMER_BATCH_SIZE = 256


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
    reference_coords: np.ndarray | None = None
    reference_mean: np.ndarray | None = None
    candidate_conformers: np.ndarray | None = None
    pose_count: int = 0


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


def _read_cases() -> list[Case]:
    cases: list[Case] = []
    with SAMPLES_CSV.open(newline="") as handle:
        for row in list(csv.DictReader(handle))[:N_CASES]:
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
                    case.reference_coords = reference_coords
                    case.reference_mean = reference_coords.mean(axis=0)
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
        case.candidate_conformers = np.where(values < case.crmsd_threshold)[0].astype(
            np.int64,
            copy=False,
        )
        del crmsds
        gc.collect()


def _translation_grid(overlap_rmsd: float) -> np.ndarray:
    n = int(ceil(overlap_rmsd / GRID_SPACING))
    axis = np.arange(-n, n + 1, dtype=np.int16)
    return np.stack(np.meshgrid(axis, axis, axis, indexing="ij"), axis=-1).reshape(-1, 3)


def _write_case_poses(case: Case, library: Any, outdir: Path) -> None:
    from poses import PoseStreamAccumulator

    if case.reference_coords is None or case.reference_mean is None:
        raise RuntimeError(f"Case {case.sample_id}: reference pose was not computed")
    if case.candidate_conformers is None:
        raise RuntimeError(f"Case {case.sample_id}: candidate conformers not computed")

    grid_offsets = _translation_grid(case.overlap_rmsd)
    grid_offsets32 = grid_offsets.astype(np.int32, copy=False)
    reference_coords = case.reference_coords.astype(np.float32, copy=False)
    reference_mean = case.reference_mean.astype(np.float32, copy=False)
    natoms = reference_coords.shape[0]

    writer = PoseStreamAccumulator(outdir)
    try:
        for conformer in case.candidate_conformers:
            conformer = int(conformer)
            coords = library.coordinates[conformer].astype(np.float32, copy=False)
            rotamers = _rotamers_to_matrices(library.get_rotamers(conformer))
            if len(rotamers) == 0:
                continue
            poses = np.einsum("aj,njk->nak", coords, rotamers)
            means = poses.mean(axis=1)
            best_grid = np.rint((reference_mean[None, :] - means) / GRID_SPACING)
            best_grid = best_grid.astype(np.int32)

            best_world = best_grid.astype(np.float32) * GRID_SPACING
            best_poses = poses + best_world[:, None, :]
            dif = best_poses - reference_coords[None, :, :]
            best_rmsd = np.sqrt(np.einsum("nij,nij->n", dif, dif) / natoms)
            base_keep = np.nonzero(best_rmsd < case.overlap_rmsd)[0]
            if base_keep.size == 0:
                continue

            for start in range(0, len(base_keep), ROTAMER_BATCH_SIZE):
                rotamer_rows = base_keep[start : start + ROTAMER_BATCH_SIZE]
                translation_grid = best_grid[rotamer_rows, None, :] + grid_offsets32
                translation_world = translation_grid.astype(np.float32) * GRID_SPACING
                shifted = (
                    poses[rotamer_rows, None, :, :]
                    + translation_world[:, :, None, :]
                )
                dif = shifted - reference_coords[None, None, :, :]
                rmsd = np.sqrt(np.einsum("rgaj,rgaj->rg", dif, dif) / natoms)
                rotamer_subrows, grid_rows = np.nonzero(rmsd < case.overlap_rmsd)
                if rotamer_subrows.size == 0:
                    continue

                translations = translation_grid[rotamer_subrows, grid_rows]
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
                rotamer_indices = rotamer_rows[rotamer_subrows].astype(
                    np.uint16,
                    copy=False,
                )
                writer.add_chunk(
                    conformers,
                    rotamer_indices,
                    translations.astype(np.int16, copy=False),
                )
        writer.finish()
        case.pose_count = writer.total_poses
    finally:
        writer.cleanup()


def _write_target_poses(cases: list[Case], directory: Any, templates_dir: Path) -> None:
    for sequence in sorted({case.target_sequence for case in cases}):
        sequence_cases = [case for case in cases if case.target_sequence == sequence]
        for mask in sorted({case.target_mask for case in sequence_cases}):
            factory, library = _load_library(directory, templates_dir, sequence, mask)
            try:
                for case in sequence_cases:
                    if case.target_mask != mask:
                        continue
                    outdir = OUTPUT_ROOT / f"case-{case.sample_id:03d}"
                    _write_case_poses(case, library, outdir)
            finally:
                del library
                factory.unload_rotaconformers()
                del factory
                gc.collect()


def _write_summary(cases: list[Case]) -> None:
    with SUMMARY_CSV.open("w", newline="") as handle:
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
                    "pose_dir": str(OUTPUT_ROOT / f"case-{case.sample_id:03d}"),
                }
            )


def main() -> None:
    cases = _read_cases()
    if OUTPUT_ROOT.exists():
        shutil.rmtree(OUTPUT_ROOT)
    OUTPUT_ROOT.mkdir(parents=True)

    directory, templates_dir = _library_directory()
    _compute_reference_poses(cases, directory, templates_dir)
    _compute_candidate_conformers(cases)
    _write_target_poses(cases, directory, templates_dir)
    _write_summary(cases)


if __name__ == "__main__":
    main()
