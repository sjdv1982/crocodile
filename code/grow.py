from __future__ import annotations

import argparse
from dataclasses import dataclass
from math import sqrt
from pathlib import Path
import sys
import time
from typing import Sequence

import numpy as np
from scipy.spatial.transform import Rotation

_CODE_DIR = Path(__file__).resolve().parent
if str(_CODE_DIR) not in sys.path:
    sys.path.insert(0, str(_CODE_DIR))

from library import config, load_crmsds
from poses import (
    PoseStreamAccumulator,
    discover_pose_pairs_with_counts,
    load_offset_table,
    open_pose_array,
)


GRID_SPACING = sqrt(3.0) / 3.0
MAX_OVERLAP_RMSD = 1.3
ROTAMER_BATCH_SIZE = 4096
RMSD_BOUNDARY_TOLERANCE = 2e-6
TRACE_RMSD_BOUNDARY_TOLERANCE = 1e-5


@dataclass(frozen=True)
class TranslationSets:
    thresholds: np.ndarray
    offsets: tuple[np.ndarray, ...]
    max_grid_discretization_sd_per_atom: float


@dataclass(frozen=True)
class SourcePool:
    conformers: np.ndarray
    rotamers: np.ndarray
    translations: np.ndarray
    unique_conformers: np.ndarray
    conformer_starts: np.ndarray
    conformer_counts: np.ndarray


@dataclass(frozen=True)
class SourceConformerCache:
    conformer: int
    coords: np.ndarray
    centered: np.ndarray
    mean0: np.ndarray
    pose_trace: float
    rotamer_indices: np.ndarray
    rotamer_matrices: np.ndarray
    rotamer_flat: np.ndarray
    mean_rotated: np.ndarray
    instance_translations: np.ndarray
    instance_starts: np.ndarray
    instance_counts: np.ndarray


@dataclass(frozen=True)
class GrowthLayout:
    source_mask: tuple[bool, bool]
    target_mask: tuple[bool, bool]
    crmsd_ab: str
    crmsd_bc: str
    source_on_rows: bool


def _existing_dir(path: str) -> str:
    p = Path(path)
    if not p.exists():
        raise argparse.ArgumentTypeError(f"directory does not exist: {path}")
    if not p.is_dir():
        raise argparse.ArgumentTypeError(f"not a directory: {path}")
    return path


def _dinucleotide_sequence(seq: str) -> str:
    seq = seq.strip().upper()
    if len(seq) != 2:
        raise argparse.ArgumentTypeError("dinucleotide sequence must have length 2")
    allowed = set("ACGU")
    if any(ch not in allowed for ch in seq):
        raise argparse.ArgumentTypeError(
            "dinucleotide sequence must contain only A/C/G/U"
        )
    return seq


def _pdb_code(code: str) -> str:
    code = code.strip()
    if len(code) != 4 or not code[0].isdigit() or not code[1:].isalnum():
        raise argparse.ArgumentTypeError(
            "PDB codes must be 4 chars: one digit + 3 alphanumeric characters"
        )
    return code.upper()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="grow",
        description="Grow a source pose pool into a target pose pool using pooled trace-SD matching.",
    )
    parser.add_argument(
        "--debug", action="store_true", help="Show a full traceback on errors."
    )
    parser.add_argument(
        "--source-poses",
        required=True,
        type=_existing_dir,
        help="Source pose directory containing poses-*.npy(.zst) and offsets-*.dat.",
    )
    parser.add_argument(
        "--source-sequence",
        required=True,
        type=_dinucleotide_sequence,
        help="Source dinucleotide sequence.",
    )
    parser.add_argument(
        "--target-sequence",
        required=True,
        type=_dinucleotide_sequence,
        help="Target dinucleotide sequence.",
    )
    parser.add_argument(
        "--direction",
        required=True,
        choices=("forward", "backward"),
        help="Growth direction relative to the source fragment.",
    )
    parser.add_argument(
        "--crmsd",
        required=True,
        type=float,
        help="cRMSD threshold for conformer pre-filtering.",
    )
    parser.add_argument(
        "--ov-rmsd",
        required=True,
        type=float,
        help="Whole-pose RMSD threshold.",
    )
    parser.add_argument(
        "--output",
        required=True,
        help="Output directory where poses-*.npy.zst and offsets-*.dat are written.",
    )
    parser.add_argument(
        "--max-poses-per-chunk",
        type=int,
        default=30_000_000,
        metavar="N",
        help="Maximum number of poses per output file pair (default: 30000000).",
    )
    parser.add_argument(
        "--test-seed",
        type=int,
        default=0,
        help="Random seed for test options (default: 0).",
    )
    parser.add_argument(
        "--test-conformers",
        type=int,
        default=None,
        metavar="N",
        help="If set, reduce the target conformer library to N elements selected at random.",
    )
    parser.add_argument(
        "--test-rotamers",
        type=int,
        default=None,
        metavar="M",
        help="If set, reduce target conformer rotamer lists to M shared positions.",
    )
    parser.add_argument(
        "--pdb-exclude",
        nargs="+",
        default=[],
        type=_pdb_code,
        metavar="PDB",
        help="One or more PDB codes to exclude from the target fragment library and cRMSD matrix.",
    )
    return parser


def _rotamers_to_matrices(rotamers: np.ndarray) -> np.ndarray:
    if rotamers.ndim == 3 and rotamers.shape[1:] == (3, 3):
        return rotamers.astype(np.float32, copy=False)
    if rotamers.ndim == 2 and rotamers.shape[1] == 3:
        return Rotation.from_rotvec(rotamers).as_matrix().astype(np.float32, copy=False)
    raise ValueError(
        "Unsupported rotamer representation; expected shape [N,3] rotvec or [N,3,3] matrix"
    )


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
        n = int(np.ceil(threshold / GRID_SPACING))
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


def _rmsd_tolerance_to_sd_tolerance(
    natoms: int,
    rmsd: np.ndarray | float,
    tolerance: float,
) -> np.ndarray | float:
    return natoms * (2.0 * np.asarray(rmsd) * tolerance + tolerance * tolerance)


def _sort_source_rows(
    conformers: np.ndarray,
    rotamers: np.ndarray,
    translations: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    order = np.argsort(rotamers, kind="stable")
    order = order[np.argsort(conformers[order], kind="stable")]
    return conformers[order], rotamers[order], translations[order]


def _load_source_pool(source_poses: str | Path) -> SourcePool:
    pairs = discover_pose_pairs_with_counts(source_poses)
    total = sum(count for _, _, count in pairs)
    conformers = np.empty(total, dtype=np.uint16)
    rotamers = np.empty(total, dtype=np.uint16)
    translations = np.empty((total, 3), dtype=np.int16)

    offset = 0
    for pose_path, offsets_path, count in pairs:
        poses = np.asarray(open_pose_array(pose_path), dtype=np.uint16)
        offset_table = load_offset_table(offsets_path)
        count = int(count)
        stop = offset + count
        conformers[offset:stop] = poses[:, 0]
        rotamers[offset:stop] = poses[:, 1]
        translations[offset:stop] = offset_table[poses[:, 2].astype(np.int64)]
        offset = stop

    conformers, rotamers, translations = _sort_source_rows(
        conformers,
        rotamers,
        translations,
    )
    unique_conformers, conformer_starts, conformer_counts = np.unique(
        conformers,
        return_index=True,
        return_counts=True,
    )
    return SourcePool(
        conformers=conformers,
        rotamers=rotamers,
        translations=translations,
        unique_conformers=unique_conformers.astype(np.int64, copy=False),
        conformer_starts=conformer_starts.astype(np.int64, copy=False),
        conformer_counts=conformer_counts.astype(np.int64, copy=False),
    )


def _resolve_growth_layout(
    source_sequence: str,
    target_sequence: str,
    direction: str,
) -> GrowthLayout:
    if direction == "forward":
        return GrowthLayout(
            source_mask=(False, True),
            target_mask=(True, False),
            crmsd_ab=source_sequence,
            crmsd_bc=target_sequence,
            source_on_rows=True,
        )
    return GrowthLayout(
        source_mask=(True, False),
        target_mask=(False, True),
        crmsd_ab=target_sequence,
        crmsd_bc=source_sequence,
        source_on_rows=False,
    )


def _stable_pose_order(rotamers: np.ndarray, translations: np.ndarray) -> np.ndarray:
    if len(rotamers) != len(translations):
        raise ValueError("rotamers and translations must have the same length")
    if len(rotamers) == 0:
        return np.empty((0,), dtype=np.int64)
    return np.lexsort(
        (
            translations[:, 2],
            translations[:, 1],
            translations[:, 0],
            rotamers,
        )
    ).astype(np.int64, copy=False)


def _build_target_to_sources(
    source_conformers: np.ndarray,
    crmsds: np.ndarray,
    threshold: float,
    *,
    source_on_rows: bool,
) -> dict[int, np.ndarray]:
    if source_on_rows:
        candidate_matrix = crmsds[source_conformers.astype(np.int64)] < threshold
    else:
        candidate_matrix = (crmsds[:, source_conformers.astype(np.int64)].T) < threshold
    source_rows, target_cols = np.nonzero(candidate_matrix)
    target_to_sources: dict[int, list[int]] = {}
    for source_row, target_col in zip(source_rows.tolist(), target_cols.tolist()):
        target_to_sources.setdefault(int(target_col), []).append(
            int(source_conformers[source_row])
        )
    return {
        target: np.asarray(sources, dtype=np.int64)
        for target, sources in target_to_sources.items()
    }


def _select_target_conformers(
    target_conformers: np.ndarray,
    count: int | None,
    seed: int,
) -> np.ndarray:
    if count is None or count >= len(target_conformers):
        return target_conformers.copy()
    if count <= 0:
        raise ValueError("--test-conformers must be positive")
    rng = np.random.default_rng(seed)
    chosen = rng.choice(target_conformers, size=count, replace=False)
    return np.sort(chosen.astype(target_conformers.dtype, copy=False))


def _select_target_rotamer_positions(
    target_library,
    target_conformers: np.ndarray,
    count: int | None,
    seed: int,
) -> np.ndarray | None:
    if count is None:
        return None
    if count <= 0:
        raise ValueError("--test-rotamers must be positive")
    smallest = None
    for conformer in target_conformers.tolist():
        nrot = len(target_library.get_rotamers(int(conformer)))
        smallest = nrot if smallest is None else min(smallest, nrot)
    assert smallest is not None
    if count > smallest:
        raise ValueError(
            f"--test-rotamers={count} is larger than the smallest selected rotamer list ({smallest})"
        )
    rng = np.random.default_rng(seed)
    return np.sort(rng.choice(smallest, size=count, replace=False).astype(np.int64))


def _build_source_caches(
    source_pool: SourcePool,
    source_library,
) -> dict[int, SourceConformerCache]:
    caches: dict[int, SourceConformerCache] = {}
    for conformer, start, count in zip(
        source_pool.unique_conformers.tolist(),
        source_pool.conformer_starts.tolist(),
        source_pool.conformer_counts.tolist(),
    ):
        stop = start + count
        if conformer >= len(source_library.coordinates):
            raise ValueError(
                f"Source conformer {conformer} out of range for source library of size {len(source_library.coordinates)}"
            )
        coords = source_library.coordinates[conformer].astype(np.float32, copy=False)
        mean0 = coords.mean(axis=0).astype(np.float32, copy=False)
        centered = (coords - mean0).astype(np.float32, copy=False)
        pose_trace = float(np.einsum("ij,ij->", centered, centered))
        rotamer_slice = source_pool.rotamers[start:stop]
        unique_rotamers, instance_starts, instance_counts = np.unique(
            rotamer_slice,
            return_index=True,
            return_counts=True,
        )
        all_rotamers = _rotamers_to_matrices(source_library.get_rotamers(conformer))
        if len(unique_rotamers) and int(unique_rotamers.max()) >= len(all_rotamers):
            raise ValueError(
                f"Source rotamer index out of range for conformer {conformer}"
            )
        rotamer_matrices = all_rotamers[unique_rotamers.astype(np.int64)]
        rotamer_flat = np.ascontiguousarray(
            rotamer_matrices.reshape(len(rotamer_matrices), 9)
        )
        mean_rotated = np.einsum("j,njk->nk", mean0, rotamer_matrices).astype(
            np.float32,
            copy=False,
        )
        caches[int(conformer)] = SourceConformerCache(
            conformer=int(conformer),
            coords=coords,
            centered=centered,
            mean0=mean0,
            pose_trace=pose_trace,
            rotamer_indices=unique_rotamers.astype(np.int64, copy=False),
            rotamer_matrices=rotamer_matrices,
            rotamer_flat=rotamer_flat,
            mean_rotated=mean_rotated,
            instance_translations=source_pool.translations[start:stop],
            instance_starts=instance_starts.astype(np.int64, copy=False),
            instance_counts=instance_counts.astype(np.int64, copy=False),
        )
    return caches


def _expand_source_instances(
    cache: SourceConformerCache,
    pp_rows: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    counts = cache.instance_counts[pp_rows]
    total = int(counts.sum())
    if total == 0:
        return np.empty((0,), dtype=np.int64), np.empty((0, 3), dtype=np.int16)
    repeat_idx = np.repeat(np.arange(len(pp_rows), dtype=np.int64), counts)
    repeated_starts = cache.instance_starts[pp_rows][repeat_idx]
    group_offsets = np.cumsum(counts, dtype=np.int64) - counts
    within = np.arange(total, dtype=np.int64) - np.repeat(group_offsets, counts)
    translation_indices = repeated_starts + within
    return repeat_idx, cache.instance_translations[translation_indices]


def _pooled_trace_grow(
    source_caches: dict[int, SourceConformerCache],
    target_library,
    target_to_sources: dict[int, np.ndarray],
    target_conformers: np.ndarray,
    *,
    ov_rmsd: float,
    max_poses_per_chunk: int,
    output_dir: Path,
    translation_sets: TranslationSets,
    target_rotamer_positions: np.ndarray | None,
) -> int:
    writer = PoseStreamAccumulator(
        output_dir,
        zstd=True,
        max_poses_per_chunk=max_poses_per_chunk,
    )
    total_overlap_sd = None
    try:
        for target_conformer in target_conformers.tolist():
            if target_conformer not in target_to_sources:
                continue
            coords_t = target_library.coordinates[target_conformer].astype(
                np.float32, copy=False
            )
            natoms = coords_t.shape[0]
            if total_overlap_sd is None:
                total_overlap_sd = natoms * ov_rmsd * ov_rmsd
            mean_t0 = coords_t.mean(axis=0).astype(np.float32, copy=False)
            centered_t = (coords_t - mean_t0).astype(np.float32, copy=False)
            target_trace = float(np.einsum("ij,ij->", centered_t, centered_t))
            rot_qq_all = _rotamers_to_matrices(
                target_library.get_rotamers(target_conformer)
            )
            if target_rotamer_positions is None:
                rot_qq = rot_qq_all
                rotamer_indices = np.arange(len(rot_qq_all), dtype=np.int64)
            else:
                rot_qq = rot_qq_all[target_rotamer_positions]
                rotamer_indices = target_rotamer_positions
            if len(rot_qq) == 0:
                continue
            rot_qq_3x3n = np.ascontiguousarray(np.transpose(rot_qq, (1, 2, 0)))
            means_q = np.einsum("j,njk->nk", mean_t0, rot_qq).astype(
                np.float32, copy=False
            )

            for source_conformer in target_to_sources[target_conformer].tolist():
                cache = source_caches.get(int(source_conformer))
                if cache is None:
                    continue
                cross_second_moment = cache.centered.T.dot(centered_t).astype(
                    np.float32,
                    copy=False,
                )
                sqq_t = np.einsum(
                    "ij,jkn->ikn",
                    cross_second_moment,
                    rot_qq_3x3n,
                    optimize=True,
                ).reshape(9, len(rot_qq))
                trace_block = cache.rotamer_flat @ sqq_t
                rc_sd = cache.pose_trace + target_trace - 2.0 * trace_block
                rc_sd = np.maximum(rc_sd, 0.0).astype(np.float32, copy=False)

                pp_rows, qq_cols = np.nonzero(rc_sd < total_overlap_sd)
                if pp_rows.size == 0:
                    continue
                rc_kept = rc_sd[pp_rows, qq_cols]
                repeat_idx, instance_translations = _expand_source_instances(
                    cache, pp_rows
                )
                if instance_translations.size == 0:
                    continue
                pp_rows_exp = pp_rows[repeat_idx]
                qq_cols_exp = qq_cols[repeat_idx]
                rc_exp = rc_kept[repeat_idx]

                source_means = (
                    cache.mean_rotated[pp_rows_exp]
                    + instance_translations.astype(np.float32) * GRID_SPACING
                )
                continuous_translation = source_means - means_q[qq_cols_exp]
                best_grid = np.rint(continuous_translation / GRID_SPACING).astype(
                    np.int32
                )
                best_world = best_grid.astype(np.float32) * GRID_SPACING
                delta = best_world - continuous_translation
                grid_discretization_sd = (
                    natoms * np.einsum("ij,ij->i", delta, delta)
                ).astype(np.float32, copy=False)
                remaining2 = total_overlap_sd - grid_discretization_sd

                rot_boundary_rmsd = np.sqrt(np.maximum(remaining2, 0.0) / natoms)
                rot_boundary_tolerance = _rmsd_tolerance_to_sd_tolerance(
                    natoms,
                    rot_boundary_rmsd,
                    TRACE_RMSD_BOUNDARY_TOLERANCE,
                )
                rot_boundary = np.abs(rc_exp - remaining2) <= rot_boundary_tolerance
                if np.any(rot_boundary):
                    boundary_rows = np.nonzero(rot_boundary)[0]
                    source_centered = np.einsum(
                        "aj,njk->nak",
                        cache.centered,
                        cache.rotamer_matrices[pp_rows_exp[boundary_rows]],
                        optimize=True,
                    )
                    target_centered = np.einsum(
                        "aj,njk->nak",
                        centered_t,
                        rot_qq[qq_cols_exp[boundary_rows]],
                        optimize=True,
                    )
                    dif = source_centered - target_centered
                    rc_exp[boundary_rows] = np.einsum("nij,nij->n", dif, dif)

                keep_rot = np.nonzero(rc_exp < remaining2)[0]
                if keep_rot.size == 0:
                    continue

                kept_pp_rows = pp_rows_exp[keep_rot]
                kept_qq_cols = qq_cols_exp[keep_rot]
                kept_rc = rc_exp[keep_rot]
                kept_best_grid = best_grid[keep_rot]
                kept_delta = delta[keep_rot]
                kept_instance_translations = instance_translations[keep_rot]
                remaining3 = (
                    total_overlap_sd - grid_discretization_sd[keep_rot] - kept_rc
                )
                remaining4 = total_overlap_sd - kept_rc

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
                    set_index = int(ordered_set_indices[set_start])
                    translation_offsets = translation_sets.offsets[set_index]
                    if len(translation_offsets) == 0:
                        continue
                    local_rows = set_order[set_start:set_stop]
                    translation_offsets32 = translation_offsets.astype(
                        np.float32, copy=False
                    )
                    shifted_delta = (
                        kept_delta[local_rows, None, :]
                        + translation_offsets32[None, :, :] * GRID_SPACING
                    )
                    translation_sd = natoms * np.einsum(
                        "rgj,rgj->rg",
                        shifted_delta,
                        shifted_delta,
                    )
                    keep = translation_sd < remaining4[local_rows, None]
                    combined_sd = translation_sd + kept_rc[local_rows, None]
                    boundary = np.abs(np.sqrt(combined_sd / natoms) - ov_rmsd) <= max(
                        RMSD_BOUNDARY_TOLERANCE,
                        TRACE_RMSD_BOUNDARY_TOLERANCE,
                    )
                    if np.any(boundary):
                        boundary_rows, boundary_offsets = np.nonzero(boundary)
                        exact_rows = local_rows[boundary_rows]
                        source_pose = np.einsum(
                            "aj,njk->nak",
                            cache.coords,
                            cache.rotamer_matrices[kept_pp_rows[exact_rows]],
                            optimize=True,
                        )
                        target_pose = np.einsum(
                            "aj,njk->nak",
                            coords_t,
                            rot_qq[kept_qq_cols[exact_rows]],
                            optimize=True,
                        )
                        source_world = (
                            kept_instance_translations[exact_rows].astype(np.float32)
                            * GRID_SPACING
                        )
                        target_world = (
                            kept_best_grid[exact_rows]
                            + translation_offsets[boundary_offsets].astype(np.int32)
                        ).astype(np.float32) * GRID_SPACING
                        dif = (
                            target_pose
                            + target_world[:, None, :]
                            - source_pose
                            - source_world[:, None, :]
                        )
                        boundary_rmsd = np.sqrt(
                            np.einsum("nij,nij->n", dif, dif) / natoms
                        )
                        keep[boundary] = False
                        keep[boundary_rows, boundary_offsets] = boundary_rmsd < ov_rmsd
                    kept_rows, kept_offsets = np.nonzero(keep)
                    if kept_rows.size == 0:
                        continue

                    translations = kept_best_grid[
                        local_rows[kept_rows]
                    ] + translation_offsets[kept_offsets].astype(np.int32)
                    if translations.size and (
                        translations.min() < np.iinfo(np.int16).min
                        or translations.max() > np.iinfo(np.int16).max
                    ):
                        raise ValueError("translation exceeds int16 range")
                    if target_conformer > np.iinfo(np.uint16).max:
                        raise ValueError("target conformer exceeds uint16 range")
                    out_conformers = np.full(
                        len(translations),
                        target_conformer,
                        dtype=np.uint16,
                    )
                    out_rotamers = rotamer_indices[kept_qq_cols[local_rows[kept_rows]]]
                    if (
                        len(out_rotamers)
                        and int(out_rotamers.max()) > np.iinfo(np.uint16).max
                    ):
                        raise ValueError("target rotamer exceeds uint16 range")
                    emit_order = _stable_pose_order(out_rotamers, translations)
                    writer.add_chunk(
                        out_conformers[emit_order],
                        out_rotamers[emit_order].astype(np.uint16, copy=False),
                        translations[emit_order].astype(np.int16, copy=False),
                    )

        writer.finish()
        return writer.total_poses
    finally:
        writer.cleanup()


def _run(args: argparse.Namespace) -> int:
    if args.max_poses_per_chunk <= 0:
        raise ValueError("--max-poses-per-chunk must be positive")
    if args.crmsd < 0.0:
        raise ValueError("--crmsd must be non-negative")
    if args.ov_rmsd <= 0.0:
        raise ValueError("--ov-rmsd must be positive")

    output_dir = Path(args.output)
    if output_dir.exists():
        raise ValueError(f"--output directory already exists: {output_dir}")

    layout = _resolve_growth_layout(
        args.source_sequence,
        args.target_sequence,
        args.direction,
    )
    if layout.crmsd_ab[1] != layout.crmsd_bc[0]:
        raise ValueError(
            f"Source and target sequences do not overlap for {args.direction} growth: {args.source_sequence}/{args.target_sequence}"
        )

    t0 = time.perf_counter()
    print("[1/6] Loading source pose pool...", file=sys.stderr)
    source_pool = _load_source_pool(args.source_poses)

    print(
        "[2/6] Loading source library and precomputing source caches...",
        file=sys.stderr,
    )
    factories, _templates = config(verify_checksums=False)
    source_factory = factories[args.source_sequence]
    source_factory.load_rotaconformers()
    source_library = source_factory.create(
        pdb_code=None,
        nucleotide_mask=np.array(layout.source_mask, dtype=bool),
        with_rotaconformers=True,
    )
    source_caches = _build_source_caches(source_pool, source_library)
    del source_library
    source_factory.unload_rotaconformers()

    print("[3/6] Building cRMSD pivot...", file=sys.stderr)
    excluded_pdb_codes = sorted(args.pdb_exclude) or None
    crmsds = load_crmsds(
        layout.crmsd_ab,
        layout.crmsd_bc,
        pdb_code=excluded_pdb_codes,
    )
    target_to_sources = _build_target_to_sources(
        source_pool.unique_conformers,
        crmsds,
        args.crmsd,
        source_on_rows=layout.source_on_rows,
    )
    target_conformers = np.array(sorted(target_to_sources.keys()), dtype=np.int64)

    print("[4/6] Loading target library...", file=sys.stderr)
    target_factory = factories[args.target_sequence]
    target_factory.load_rotaconformers()
    target_library = target_factory.create(
        pdb_code=excluded_pdb_codes,
        nucleotide_mask=np.array(layout.target_mask, dtype=bool),
        with_rotaconformers=True,
    )
    if target_library.conformer_mask is not None:
        valid = target_library.conformer_mask.astype(bool)
        target_conformers = target_conformers[valid[target_conformers]]
        target_to_sources = {
            int(target): sources
            for target, sources in target_to_sources.items()
            if target < len(valid) and bool(valid[target])
        }
    target_conformers = _select_target_conformers(
        target_conformers,
        args.test_conformers,
        args.test_seed,
    )
    target_rotamer_positions = _select_target_rotamer_positions(
        target_library,
        target_conformers,
        args.test_rotamers,
        args.test_seed,
    )

    print("[5/6] Running pooled trace grow...", file=sys.stderr)
    translation_sets = _precompute_translation_sets()
    total_poses = _pooled_trace_grow(
        source_caches,
        target_library,
        target_to_sources,
        target_conformers,
        ov_rmsd=args.ov_rmsd,
        max_poses_per_chunk=args.max_poses_per_chunk,
        output_dir=output_dir,
        translation_sets=translation_sets,
        target_rotamer_positions=target_rotamer_positions,
    )

    print("[6/6] Finalizing...", file=sys.stderr)
    elapsed = time.perf_counter() - t0
    print(
        (
            f"source_poses={len(source_pool.conformers)} "
            f"source_conformers={len(source_pool.unique_conformers)} "
            f"target_conformers={len(target_conformers)} "
            f"accepted_poses={total_poses} elapsed={elapsed:.2f}s"
        ),
        file=sys.stderr,
    )
    target_factory.unload_rotaconformers()
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    try:
        return int(_run(args))
    except (
        ValueError,
        FileNotFoundError,
        RuntimeError,
        OSError,
        KeyError,
        TypeError,
    ) as exc:
        if getattr(args, "debug", False):
            raise
        print(f"error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
