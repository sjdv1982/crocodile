from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from poses import load_offset_table, open_pose_array


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


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="decode_rotamer_matrices",
        description=(
            "Decode one poses file + offsets file and convert each pose to a 4x4 "
            "homogeneous transform matrix using rotamers from the fragment library."
        ),
    )
    parser.add_argument("--poses", required=True, type=_existing_file, help="Path to poses-*.npy (or .npy.zst)")
    parser.add_argument("--offsets", required=True, type=_existing_file, help="Path to offsets-*.dat")
    parser.add_argument("--sequence", required=True, type=_dinucleotide_sequence, help="Dinucleotide sequence (AA/AC/...)")
    parser.add_argument("--output", required=True, help="Output path for matrices (.npy or .txt)")
    parser.add_argument(
        "--format",
        choices=("npy", "txt"),
        default="npy",
        help="Output format (default: npy)",
    )
    parser.add_argument(
        "--verify-checksums",
        action="store_true",
        help="Enable fraglib checksum verification when loading library config.",
    )
    return parser


def _load_poses(path: Path) -> np.ndarray:
    poses = np.asarray(open_pose_array(path), dtype=np.uint16)
    if poses.ndim != 2 or poses.shape[1] != 3:
        raise ValueError(f"Invalid pose array shape in {path}: {poses.shape}")
    return poses


def _write_txt(path: Path, matrices: np.ndarray) -> None:
    with path.open("w") as handle:
        for i, mat in enumerate(matrices):
            if i:
                handle.write("\n")
            for row in mat:
                handle.write(" ".join(f"{x:.8f}" for x in row))
                handle.write("\n")


def decode_to_matrices(
    poses_path: Path,
    offsets_path: Path,
    sequence: str,
    verify_checksums: bool,
) -> np.ndarray:
    from library import config

    poses = _load_poses(poses_path)
    offset_table = load_offset_table(offsets_path)

    offset_indices = poses[:, 2].astype(np.int64, copy=False)
    if offset_indices.size and (offset_indices.max() >= len(offset_table)):
        raise ValueError("Pose offset index out of range for offsets table")
    translations = offset_table[offset_indices].astype(np.float32, copy=False)

    libraries, _ = config(verify_checksums=verify_checksums)
    if sequence not in libraries:
        raise ValueError(f"Sequence {sequence} not available in fragment library")

    libf = libraries[sequence]
    libf.load_rotaconformers()
    lib = libf.create(None, with_rotaconformers=True)

    matrices = np.zeros((len(poses), 4, 4), dtype=np.float32)
    matrices[:, 3, 3] = 1.0

    for i, (conf_u16, rot_u16, _) in enumerate(poses):
        conf = int(conf_u16)
        rot = int(rot_u16)
        rotamers = lib.get_rotamers(conf)
        if rot >= len(rotamers):
            raise ValueError(
                f"Pose {i}: rotamer index {rot} out of range for conformer {conf} "
                f"(n={len(rotamers)})"
            )
        matrices[i, :3, :3] = rotamers[rot].astype(np.float32, copy=False)
        matrices[i, :3, 3] = translations[i]

    return matrices


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    poses_path = Path(args.poses)
    offsets_path = Path(args.offsets)
    output_path = Path(args.output)

    matrices = decode_to_matrices(
        poses_path,
        offsets_path,
        args.sequence,
        verify_checksums=args.verify_checksums,
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    if args.format == "npy":
        np.save(output_path, matrices)
    else:
        _write_txt(output_path, matrices)


if __name__ == "__main__":
    main()
