#!/usr/bin/env python3
from __future__ import annotations

import argparse
import io
from pathlib import Path

import numpy as np


_POSE_ZSTD_SUFFIX = ".npy.zst"


def _existing_dir(path: str) -> str:
    p = Path(path)
    if not p.exists():
        raise argparse.ArgumentTypeError(f"directory does not exist: {path}")
    if not p.is_dir():
        raise argparse.ArgumentTypeError(f"not a directory: {path}")
    return path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="lookup_pose_index",
        description=(
            "Look up the global 0-based index of a single-pose best-fit directory "
            "inside a larger pose directory."
        ),
    )
    parser.add_argument(
        "--best-dir",
        required=True,
        type=_existing_dir,
        help="Directory that must contain exactly one pose",
    )
    parser.add_argument(
        "--pose-dir",
        required=True,
        type=_existing_dir,
        help="Directory containing poses-*.npy(.zst) and offsets-*.dat shards",
    )
    return parser


def _pose_index_from_name(name: str) -> int | None:
    if name.endswith(_POSE_ZSTD_SUFFIX):
        base = name[: -len(".zst")]
    elif name.endswith(".npy"):
        base = name
    else:
        return None
    if not base.startswith("poses-") or not base.endswith(".npy"):
        return None
    index_text = base[len("poses-") : -4]
    if not index_text.isdigit():
        return None
    return int(index_text)


def _pose_path_priority(path: Path) -> int:
    return 1 if path.name.endswith(_POSE_ZSTD_SUFFIX) else 0


def _discover_pose_pairs(directory: Path) -> list[tuple[int, Path, Path]]:
    pose_files = list(directory.glob("poses-*.npy")) + list(
        directory.glob(f"poses-*{_POSE_ZSTD_SUFFIX}")
    )
    if not pose_files:
        raise FileNotFoundError(f"No pose files found in {directory}")

    indexed: dict[int, Path] = {}
    for pose_path in pose_files:
        index = _pose_index_from_name(pose_path.name)
        if index is None:
            continue
        previous = indexed.get(index)
        if previous is None or _pose_path_priority(pose_path) > _pose_path_priority(
            previous
        ):
            indexed[index] = pose_path

    if not indexed:
        raise FileNotFoundError(f"No valid pose files found in {directory}")

    pairs: list[tuple[int, Path, Path]] = []
    for index in sorted(indexed):
        offsets_path = directory / f"offsets-{index}.dat"
        if not offsets_path.is_file():
            raise FileNotFoundError(f"Missing {offsets_path} for {indexed[index]}")
        pairs.append((index, indexed[index], offsets_path))
    return pairs


def _require_zstandard(action: str):
    try:
        import zstandard as zstd
    except ImportError as exc:
        raise ImportError(
            f"zstandard is required to {action} compressed pose files ({_POSE_ZSTD_SUFFIX})"
        ) from exc
    return zstd


def _open_pose_array_lowlevel(path: Path) -> np.ndarray:
    if path.name.endswith(_POSE_ZSTD_SUFFIX):
        zstd = _require_zstandard("read")
        with path.open("rb") as compressed:
            with zstd.ZstdDecompressor().stream_reader(compressed) as reader:
                data = reader.read()
        return np.load(io.BytesIO(data), allow_pickle=False)
    return np.load(path, mmap_mode="r", allow_pickle=False)


def _read_npy_header(fileobj) -> tuple[tuple[int, ...], bool, np.dtype]:
    version = np.lib.format.read_magic(fileobj)
    if version == (1, 0):
        shape, fortran_order, dtype = np.lib.format.read_array_header_1_0(fileobj)
    elif version == (2, 0):
        shape, fortran_order, dtype = np.lib.format.read_array_header_2_0(fileobj)
    elif version == (3, 0) and hasattr(np.lib.format, "read_array_header_3_0"):
        shape, fortran_order, dtype = np.lib.format.read_array_header_3_0(fileobj)
    else:
        raise ValueError(f"Unsupported .npy version {version}")
    return shape, fortran_order, dtype


def _pose_count(path: Path) -> int:
    if path.name.endswith(_POSE_ZSTD_SUFFIX):
        zstd = _require_zstandard("read")
        with path.open("rb") as compressed:
            with zstd.ZstdDecompressor().stream_reader(compressed) as reader:
                shape, _, _ = _read_npy_header(reader)
    else:
        with path.open("rb") as handle:
            shape, _, _ = _read_npy_header(handle)
    if len(shape) != 2 or shape[1] != 3:
        raise ValueError(f"Invalid pose array shape in {path}: {shape}")
    return int(shape[0])


def _read_offsets_file(path: Path) -> np.ndarray:
    raw = path.read_bytes()
    if len(raw) < 6:
        raise ValueError(f"{path} is too small")
    mean_offset = np.frombuffer(raw[:6], dtype=np.int16)
    remainder = raw[6:]
    if len(remainder) % 3 != 0:
        raise ValueError(f"{path} remainder length must be divisible by 3")
    offsets_uint8 = np.frombuffer(remainder, dtype=np.uint8)
    offsets_uint8 = offsets_uint8.reshape((-1, 3))
    offsets_int8 = offsets_uint8.view(np.int8)
    return offsets_int8.astype(np.int16) + mean_offset.astype(np.int16)


def _load_single_pose(directory: Path) -> tuple[int, int, np.ndarray]:
    pairs = _discover_pose_pairs(directory)
    total = sum(_pose_count(pose_path) for _, pose_path, _ in pairs)
    if total != 1:
        raise ValueError(f"{directory} must contain exactly one pose; found {total}")

    for _, pose_path, offsets_path in pairs:
        poses = _open_pose_array_lowlevel(pose_path)
        if len(poses) == 0:
            continue
        pose = np.asarray(poses[0], dtype=np.uint16)
        offset_table = _read_offsets_file(offsets_path)
        offset_index = int(pose[2])
        if offset_index >= len(offset_table):
            raise ValueError(f"Pose offset index out of range in {pose_path}")
        return int(pose[0]), int(pose[1]), offset_table[offset_index].astype(np.int16)

    raise RuntimeError(f"{directory} claims to contain one pose but none was found")


def lookup_pose_index(best_dir: Path, pose_dir: Path) -> int:
    best_conf, best_rot, best_translation = _load_single_pose(best_dir)

    global_offset = 0
    for _, pose_path, offsets_path in _discover_pose_pairs(pose_dir):
        offset_table = _read_offsets_file(offsets_path)
        if len(offset_table) == 0:
            global_offset += _pose_count(pose_path)
            continue

        translation_mask = np.all(offset_table == best_translation[None, :], axis=1)
        if not np.any(translation_mask):
            global_offset += _pose_count(pose_path)
            continue

        matching_offset_indices = np.flatnonzero(translation_mask).astype(np.uint16)
        poses = _open_pose_array_lowlevel(pose_path)
        if poses.ndim != 2 or poses.shape[1] != 3:
            raise ValueError(f"Invalid pose array shape in {pose_path}: {poses.shape}")
        if len(poses) == 0:
            global_offset += len(poses)
            continue

        mask = np.isin(poses[:, 2], matching_offset_indices)
        if not np.any(mask):
            global_offset += len(poses)
            continue

        mask &= poses[:, 0] == best_conf
        mask &= poses[:, 1] == best_rot
        if np.any(mask):
            local_index = int(np.flatnonzero(mask)[0])
            return global_offset + local_index

        global_offset += len(poses)

    return -1


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    result = lookup_pose_index(Path(args.best_dir), Path(args.pose_dir))
    print(result)


if __name__ == "__main__":
    main()
