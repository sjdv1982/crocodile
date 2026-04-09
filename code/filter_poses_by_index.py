#!/usr/bin/env python3
"""Filter a pose pool by global 0-based pose indices."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Sequence

import numpy as np

from poses import (
    PoseStreamAccumulator,
    canonical_center_codes,
    discover_pose_pairs_with_counts,
    load_offset_table,
    open_pose_array,
)


def _existing_dir(path: str) -> str:
    p = Path(path)
    if not p.exists():
        raise argparse.ArgumentTypeError(f"directory does not exist: {path}")
    if not p.is_dir():
        raise argparse.ArgumentTypeError(f"not a directory: {path}")
    return path


def _existing_file(path: str) -> str:
    p = Path(path)
    if not p.exists():
        raise argparse.ArgumentTypeError(f"file does not exist: {path}")
    if not p.is_file():
        raise argparse.ArgumentTypeError(f"not a file: {path}")
    return path


def _positive_int(value: str) -> int:
    try:
        result = int(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(f"invalid integer: {value}") from exc
    if result <= 0:
        raise argparse.ArgumentTypeError("value must be positive")
    return result


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="filter_poses_by_index",
        description=(
            "Filter a pose directory by a NumPy array of global 0-based pose indices."
        ),
    )
    parser.add_argument(
        "--input-dir",
        required=True,
        type=_existing_dir,
        help="Directory containing poses-*.npy(.zst) and offsets-*.dat files.",
    )
    parser.add_argument(
        "--indices",
        required=True,
        type=_existing_file,
        help="Path to a .npy or .npz file containing a 1D integer index array.",
    )
    parser.add_argument(
        "--output-dir",
        required=True,
        help="Output pose directory; must not already exist.",
    )
    parser.add_argument(
        "--indices-key",
        default=None,
        help="Array name to load from a .npz archive. Required when the archive contains multiple arrays.",
    )
    parser.add_argument(
        "--max-poses-per-chunk",
        type=_positive_int,
        default=2**32,
        metavar="N",
        help="Maximum poses per output chunk (default: 4294967296).",
    )
    parser.add_argument(
        "--zstd",
        action="store_true",
        help="Write compressed poses-*.npy.zst output chunks.",
    )
    return parser


def _load_index_array(path: Path, *, indices_key: str | None) -> np.ndarray:
    loaded = np.load(path, allow_pickle=False)
    if isinstance(loaded, np.ndarray):
        if indices_key is not None:
            raise ValueError("--indices-key can only be used with .npz input")
        return loaded

    try:
        names = list(loaded.files)
        if indices_key is None:
            if len(names) != 1:
                raise ValueError(
                    f"{path} contains multiple arrays; use --indices-key to select one"
                )
            indices_key = names[0]
        elif indices_key not in loaded:
            raise ValueError(f"{path} does not contain array {indices_key!r}")
        return loaded[indices_key]
    finally:
        loaded.close()


def _normalize_indices(indices: np.ndarray, *, total_poses: int) -> np.ndarray:
    indices = np.asarray(indices)
    if indices.ndim != 1:
        raise ValueError("indices array must be 1D")
    if not np.issubdtype(indices.dtype, np.integer):
        raise ValueError("indices array must have an integer dtype")
    if len(indices) == 0:
        return indices.astype(np.int64, copy=False)

    min_index = int(indices.min())
    max_index = int(indices.max())
    if min_index < 0:
        raise ValueError("indices array must contain only non-negative values")
    if max_index >= total_poses:
        raise ValueError(
            f"indices array contains out-of-range value {max_index}; total poses: {total_poses}"
        )
    return indices.astype(np.int64, copy=False)


def _validate_output_dir(output_dir: Path) -> None:
    if output_dir.exists():
        raise ValueError(f"--output-dir already exists: {output_dir}")


def filter_poses_by_index(
    input_dir: str | Path,
    indices_path: str | Path,
    output_dir: str | Path,
    *,
    indices_key: str | None = None,
    max_poses_per_chunk: int = 2**32,
    zstd: bool = False,
) -> list[tuple[Path, Path]]:
    """Filter a pose directory by a NumPy array of global 0-based pose indices.

    The global pose index space is defined by numeric shard order
    (`poses-1`, `poses-2`, ...) and row order within each shard. The selected
    poses are written into a fresh output pose pool with compact shard numbering
    starting at 1. Input indices may be unsorted and may contain duplicates;
    output pose order matches the order of the provided indices exactly.

    Parameters
    ----------
    input_dir
        Directory containing `poses-*.npy` or `poses-*.npy.zst` files and their
        matching `offsets-*.dat` files.
    indices_path
        Path to a `.npy` or `.npz` file containing a 1D integer array of
        global 0-based pose indices.
    output_dir
        Directory to create for the filtered output pose pool. It must not
        already exist.
    indices_key
        Optional array name to load from a `.npz` archive.
    max_poses_per_chunk
        Maximum number of poses to write into each output shard.
    zstd
        If true, write compressed `poses-*.npy.zst` output shards.

    Returns
    -------
    list[tuple[Path, Path]]
        The written `(poses_path, offsets_path)` pairs in output shard order.
    """

    input_dir = Path(input_dir)
    indices_path = Path(indices_path)
    output_dir = Path(output_dir)

    if max_poses_per_chunk <= 0:
        raise ValueError("max_poses_per_chunk must be positive")
    _validate_output_dir(output_dir)

    pairs = discover_pose_pairs_with_counts(input_dir)
    counts = np.array([count for _, _, count in pairs], dtype=np.int64)
    cumulative = np.cumsum(counts, dtype=np.int64)
    total_poses = int(cumulative[-1]) if len(cumulative) else 0

    indices = _normalize_indices(
        _load_index_array(indices_path, indices_key=indices_key),
        total_poses=total_poses,
    )

    writer = PoseStreamAccumulator(
        output_dir,
        max_poses_per_chunk=max_poses_per_chunk,
        zstd=zstd,
    )
    if len(indices) == 0:
        return writer.finish()

    order = np.argsort(indices, kind="stable")
    sorted_indices = indices[order]
    chunk_ids = np.searchsorted(cumulative, sorted_indices, side="right")
    chunk_boundaries = np.flatnonzero(chunk_ids[1:] != chunk_ids[:-1]) + 1
    group_starts = np.concatenate(([0], chunk_boundaries))
    group_stops = np.concatenate((chunk_boundaries, [len(sorted_indices)]))
    chunk_starts = np.concatenate(([0], cumulative[:-1]))

    conformers = np.empty((len(indices),), dtype=np.uint16)
    rotamers = np.empty((len(indices),), dtype=np.uint16)
    translations = np.empty((len(indices), 3), dtype=np.int16)

    for start, stop in zip(group_starts, group_stops):
        chunk_id = int(chunk_ids[start])
        pose_path, offsets_path, _ = pairs[chunk_id]
        local_rows = (sorted_indices[start:stop] - chunk_starts[chunk_id]).astype(
            np.intp, copy=False
        )
        output_rows = order[start:stop].astype(np.intp, copy=False)

        poses = open_pose_array(pose_path)
        if poses.ndim != 2 or poses.shape[1] != 3:
            raise ValueError(f"Invalid pose array shape in {pose_path}: {poses.shape}")
        selected = np.asarray(poses[local_rows], dtype=np.uint16)

        offset_table = load_offset_table(offsets_path)
        offset_indices = selected[:, 2].astype(np.intp, copy=False)
        if offset_indices.size and int(offset_indices.max()) >= len(offset_table):
            raise ValueError(f"Pose offset index out of range in {pose_path}")

        conformers[output_rows] = selected[:, 0]
        rotamers[output_rows] = selected[:, 1]
        translations[output_rows] = offset_table[offset_indices]

    center_codes = canonical_center_codes(translations)
    run_starts = np.concatenate(
        ([0], np.flatnonzero(center_codes[1:] != center_codes[:-1]) + 1)
    )
    run_stops = np.concatenate((run_starts[1:], [len(translations)]))
    for start, stop in zip(run_starts, run_stops):
        writer.add_chunk(
            conformers[start:stop],
            rotamers[start:stop],
            translations[start:stop],
        )
    return writer.finish()


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    filter_poses_by_index(
        args.input_dir,
        args.indices,
        args.output_dir,
        indices_key=args.indices_key,
        max_poses_per_chunk=args.max_poses_per_chunk,
        zstd=args.zstd,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
