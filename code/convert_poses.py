#!/usr/bin/env python3
"""Convert poses-*.npy + offsets-*.dat + fragment library → rotvec DOFs.

Output:
  <output-prefix>.rotvec.npy   -- (N, 6) float64: columns 0-2 = Rodrigues rotvec,
                                   columns 3-5 = world-frame translation in Angstrom
  <output-prefix>.conformers.npy -- (N,) int32: 0-based conformer index per pose
"""
from __future__ import annotations

import argparse
from math import sqrt
from pathlib import Path

import numpy as np

GRID_SPACING = sqrt(3) / 3


def _existing_file(path: str) -> str:
    p = Path(path)
    if not p.exists():
        raise argparse.ArgumentTypeError(f"file does not exist: {path}")
    if not p.is_file():
        raise argparse.ArgumentTypeError(f"not a file: {path}")
    return path


def _existing_dir(path: str) -> str:
    p = Path(path)
    if not p.exists():
        raise argparse.ArgumentTypeError(f"directory does not exist: {path}")
    if not p.is_dir():
        raise argparse.ArgumentTypeError(f"not a directory: {path}")
    return path


def _positive_int(value: str) -> int:
    try:
        result = int(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(f"invalid integer: {value}") from exc
    if result <= 0:
        raise argparse.ArgumentTypeError("index must be positive")
    return result


def _dinucleotide_sequence(seq: str) -> str:
    seq = seq.strip().upper()
    if len(seq) != 2:
        raise argparse.ArgumentTypeError("--sequence must be a dinucleotide (length 2)")
    if any(ch not in {"A", "C", "G", "U"} for ch in seq):
        raise argparse.ArgumentTypeError("--sequence must contain only A/C/G/U")
    return seq


def _rotamers_to_rotvecs(rotamers: np.ndarray) -> np.ndarray:
    """Convert rotamers (N,3) rotvec or (N,3,3) matrix → (N,3) float64 rotvec."""
    if rotamers.ndim == 2 and rotamers.shape[1] == 3:
        return rotamers.astype(np.float64, copy=False)
    if rotamers.ndim == 3 and rotamers.shape[1:] == (3, 3):
        from scipy.spatial.transform import Rotation

        return Rotation.from_matrix(rotamers).as_rotvec().astype(np.float64)
    raise ValueError(
        f"Unsupported rotamer shape {rotamers.shape}; expected (N,3) or (N,3,3)"
    )


def _load_rotamer_library(sequence: str, verify_checksums: bool):
    from library import config

    libraries, _ = config(verify_checksums=verify_checksums)
    if sequence not in libraries:
        raise ValueError(f"Sequence {sequence} not available in fragment library")

    libf = libraries[sequence]
    libf.load_rotaconformers()
    return libf.create(None, with_rotaconformers=True)


def _convert_loaded_poses(
    poses: np.ndarray,
    offset_table: np.ndarray,
    lib,
    rotvec_cache: dict[int, np.ndarray],
    source_name: str,
) -> tuple[np.ndarray, np.ndarray]:
    poses = np.asarray(poses, dtype=np.uint16)
    if poses.ndim != 2 or poses.shape[1] != 3:
        raise ValueError(f"Invalid pose array shape in {source_name}: {poses.shape}")

    n = len(poses)
    conformer_ids = poses[:, 0]
    rotamer_ids = poses[:, 1].astype(np.intp, copy=False)
    offset_indices = poses[:, 2].astype(np.intp, copy=False)
    if offset_indices.size and (offset_indices.max() >= len(offset_table)):
        raise ValueError(f"Pose offset index out of range for offsets table in {source_name}")
    rotvec_dofs = np.empty((n, 6), dtype=np.float64)
    rotvec_dofs[:, 3:] = offset_table[offset_indices].astype(np.float64, copy=False)
    rotvec_dofs[:, 3:] *= GRID_SPACING

    conformers = conformer_ids.astype(np.int32, copy=False)
    if n == 0:
        return rotvec_dofs, conformers

    conf_indices_sorted = np.argsort(conformer_ids, kind="stable")
    sorted_conf = conformer_ids[conf_indices_sorted]
    boundaries = np.flatnonzero(np.diff(sorted_conf)) + 1
    group_starts = np.empty((len(boundaries) + 2,), dtype=np.intp)
    group_starts[0] = 0
    group_starts[-1] = n
    group_starts[1:-1] = boundaries

    for start, end in zip(group_starts[:-1], group_starts[1:]):
        rows = conf_indices_sorted[start:end]
        conf = int(sorted_conf[start])
        rv = rotvec_cache.get(conf)
        if rv is None:
            rv = _rotamers_to_rotvecs(lib.get_rotamers(conf))
            rotvec_cache[conf] = rv
        group_rotamers = rotamer_ids[rows]
        bad = group_rotamers >= len(rv)
        if np.any(bad):
            bad_pos = int(np.flatnonzero(bad)[0])
            pose_index = int(rows[bad_pos])
            rot = int(group_rotamers[bad_pos])
            raise ValueError(
                f"Pose {pose_index} in {source_name}: rotamer index {rot} out of range "
                f"for conformer {conf} (n={len(rv)})"
            )
        rotvec_dofs[rows, :3] = rv[group_rotamers]

    return rotvec_dofs, conformers


def convert_poses(
    poses_path: Path,
    offsets_path: Path,
    sequence: str,
    verify_checksums: bool,
) -> tuple[np.ndarray, np.ndarray]:
    """Return (rotvec_dofs, conformers) arrays.

    rotvec_dofs : (N, 6) float64 — (v0, v1, v2, tx, ty, tz)
      tx/ty/tz are world-frame translations from the offsets table
    conformers  : (N,) int32 — 0-based conformer index
    """
    from poses import load_offset_table, open_pose_array

    poses = open_pose_array(poses_path)
    offset_table = load_offset_table(offsets_path)
    rotvec_cache: dict[int, np.ndarray] = {}
    lib = _load_rotamer_library(sequence, verify_checksums=verify_checksums)
    return _convert_loaded_poses(
        poses,
        offset_table,
        lib,
        rotvec_cache,
        str(poses_path),
    )


def convert_pose_range(
    pose_dir: Path,
    first_index: int,
    last_index: int,
    sequence: str,
    verify_checksums: bool,
) -> tuple[np.ndarray, np.ndarray]:
    from poses import load_offset_table, open_pose_array

    if first_index > last_index:
        raise ValueError("first_index must be <= last_index")

    lib = _load_rotamer_library(sequence, verify_checksums=verify_checksums)
    rotvec_cache: dict[int, np.ndarray] = {}
    rotvec_chunks: list[np.ndarray] = []
    conformer_chunks: list[np.ndarray] = []

    for index in range(first_index, last_index + 1):
        poses_path = pose_dir / f"poses-{index}.npy.zst"
        if not poses_path.exists():
            poses_path = pose_dir / f"poses-{index}.npy"
        offsets_path = pose_dir / f"offsets-{index}.dat"
        if not poses_path.is_file():
            raise FileNotFoundError(f"missing poses file for index {index}: {poses_path}")
        if not offsets_path.is_file():
            raise FileNotFoundError(
                f"missing offsets file for index {index}: {offsets_path}"
            )

        poses = open_pose_array(poses_path)
        offset_table = load_offset_table(offsets_path)
        rotvec_dofs, conformers = _convert_loaded_poses(
            poses,
            offset_table,
            lib,
            rotvec_cache,
            str(poses_path),
        )
        rotvec_chunks.append(rotvec_dofs)
        conformer_chunks.append(conformers)

    if not rotvec_chunks:
        return np.empty((0, 6), dtype=np.float64), np.empty((0,), dtype=np.int32)

    return np.concatenate(rotvec_chunks, axis=0), np.concatenate(conformer_chunks, axis=0)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--poses", type=_existing_file, help="poses-*.npy(.zst)")
    ap.add_argument(
        "--offsets", type=_existing_file, help="offsets-*.dat"
    )
    ap.add_argument(
        "--pose-dir",
        type=_existing_dir,
        help="Directory containing poses-<index>.npy(.zst) and offsets-<index>.dat",
    )
    ap.add_argument(
        "--first-index",
        type=_positive_int,
        help="First inclusive pose/offets shard index",
    )
    ap.add_argument(
        "--last-index",
        type=_positive_int,
        help="Last inclusive pose/offsets shard index",
    )
    ap.add_argument(
        "--sequence",
        required=True,
        type=_dinucleotide_sequence,
        help="Dinucleotide sequence (AA/AC/.../UU)",
    )
    ap.add_argument(
        "--output-prefix", required=True, help="output prefix (no extension)"
    )
    ap.add_argument(
        "--verify-checksums",
        action="store_true",
        help="Enable fraglib checksum verification",
    )
    args = ap.parse_args()

    if args.pose_dir is not None:
        if args.poses is not None or args.offsets is not None:
            ap.error("--pose-dir cannot be combined with --poses/--offsets")
        if args.first_index is None or args.last_index is None:
            ap.error("--pose-dir requires both --first-index and --last-index")
        rotvec_dofs, conformers = convert_pose_range(
            Path(args.pose_dir),
            args.first_index,
            args.last_index,
            args.sequence,
            verify_checksums=args.verify_checksums,
        )
    else:
        if args.poses is None or args.offsets is None:
            ap.error("either --pose-dir or both --poses/--offsets are required")
        if args.first_index is not None or args.last_index is not None:
            ap.error("--first-index/--last-index require --pose-dir")
        rotvec_dofs, conformers = convert_poses(
            Path(args.poses),
            Path(args.offsets),
            args.sequence,
            verify_checksums=args.verify_checksums,
        )

    prefix = args.output_prefix
    np.save(prefix + ".rotvec.npy", rotvec_dofs)
    np.save(prefix + ".conformers.npy", conformers)
    print(
        f"Wrote {prefix}.rotvec.npy ({rotvec_dofs.shape}) "
        f"and {prefix}.conformers.npy ({conformers.shape})"
    )


if __name__ == "__main__":
    main()
