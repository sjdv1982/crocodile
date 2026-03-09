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
    from library import config

    poses = np.asarray(open_pose_array(poses_path), dtype=np.uint16)
    if poses.ndim != 2 or poses.shape[1] != 3:
        raise ValueError(f"Invalid pose array shape in {poses_path}: {poses.shape}")

    offset_table = load_offset_table(offsets_path)
    offset_indices = poses[:, 2].astype(np.int64, copy=False)
    if offset_indices.size and (offset_indices.max() >= len(offset_table)):
        raise ValueError("Pose offset index out of range for offsets table")
    trans_world = (
        offset_table[offset_indices].astype(np.float64, copy=False) * GRID_SPACING
    )

    libraries, _ = config(verify_checksums=verify_checksums)
    if sequence not in libraries:
        raise ValueError(f"Sequence {sequence} not available in fragment library")

    libf = libraries[sequence]
    libf.load_rotaconformers()
    lib = libf.create(None, with_rotaconformers=True)

    n = len(poses)
    rotvecs = np.zeros((n, 3), dtype=np.float64)
    conformers = poses[:, 0].astype(np.int32) - 1

    conf_indices_sorted = np.argsort(poses[:, 0], kind="stable")
    rotvec_cache: dict[int, np.ndarray] = {}

    for i in conf_indices_sorted:
        conf_u16, rot_u16, _ = poses[i]
        conf = int(conf_u16)
        rot = int(rot_u16)
        if conf not in rotvec_cache:
            rotvec_cache[conf] = _rotamers_to_rotvecs(lib.get_rotamers(conf))
        rv = rotvec_cache[conf]
        if rot >= len(rv):
            raise ValueError(
                f"Pose {int(i)}: rotamer index {rot} out of range for conformer {conf} "
                f"(n={len(rv)})"
            )
        rotvecs[int(i)] = rv[rot]

    rotvec_dofs = np.concatenate([rotvecs, trans_world], axis=1)
    return rotvec_dofs, conformers


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--poses", required=True, type=_existing_file, help="poses-*.npy")
    ap.add_argument("--offsets", required=True, type=_existing_file, help="offsets-*.dat")
    ap.add_argument(
        "--sequence",
        required=True,
        type=_dinucleotide_sequence,
        help="Dinucleotide sequence (AA/AC/.../UU)",
    )
    ap.add_argument("--output-prefix", required=True, help="output prefix (no extension)")
    ap.add_argument(
        "--verify-checksums",
        action="store_true",
        help="Enable fraglib checksum verification",
    )
    args = ap.parse_args()

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
