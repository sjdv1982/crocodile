from __future__ import annotations

import argparse
import math
import tempfile
from collections import OrderedDict
from pathlib import Path
from typing import Iterable

import numpy as np

from poses import (
    PoseStreamAccumulator,
    discover_pose_pairs,
    load_offset_table,
    open_pose_array,
)


KEY_DTYPE = np.dtype(
    [
        ("conf", "<u2"),
        ("rot", "<u2"),
        ("x", "<i2"),
        ("y", "<i2"),
        ("z", "<i2"),
    ],
    align=False,
)
RECORD_DTYPE = np.dtype(KEY_DTYPE.descr + [("idx", "<u8")], align=False)
RECORD_BYTES = RECORD_DTYPE.itemsize


class BucketWriters:
    def __init__(self, base: Path, max_open_files: int = 128) -> None:
        self.base = base
        self.base.mkdir(parents=True, exist_ok=True)
        self.max_open_files = max(8, max_open_files)
        self._handles: OrderedDict[int, object] = OrderedDict()

    def write(self, bucket: int, array: np.ndarray) -> None:
        if array.size == 0:
            return
        handle = self._handles.get(bucket)
        if handle is None:
            if len(self._handles) >= self.max_open_files:
                _, old = self._handles.popitem(last=False)
                old.close()
            path = self.base / f"bucket-{bucket:08d}.bin"
            handle = path.open("ab")
            self._handles[bucket] = handle
        else:
            self._handles.move_to_end(bucket)
        array.tofile(handle)

    def close(self) -> None:
        while self._handles:
            _, handle = self._handles.popitem(last=False)
            handle.close()


def _pow2_ceil(value: int) -> int:
    if value <= 1:
        return 1
    return 1 << (value - 1).bit_length()


def _existing_dir(path: str) -> str:
    p = Path(path)
    if not p.exists():
        raise argparse.ArgumentTypeError(f"directory does not exist: {path}")
    if not p.is_dir():
        raise argparse.ArgumentTypeError(f"not a directory: {path}")
    return path


def _count_poses(pairs: Iterable[tuple[Path, Path]]) -> int:
    total = 0
    for pose_path, _ in pairs:
        with open_pose_array(pose_path) as arr:
            if arr.ndim != 2 or arr.shape[1] != 3:
                raise ValueError(f"Invalid pose array shape in {pose_path}: {arr.shape}")
            total += int(arr.shape[0])
    return total


def _make_records(poses_block: np.ndarray, offset_table: np.ndarray, start_idx: int) -> np.ndarray:
    n = int(poses_block.shape[0])
    records = np.empty(n, dtype=RECORD_DTYPE)
    records["conf"] = poses_block[:, 0]
    records["rot"] = poses_block[:, 1]
    offset_indices = poses_block[:, 2].astype(np.int64, copy=False)
    coords = offset_table[offset_indices]
    records["x"] = coords[:, 0]
    records["y"] = coords[:, 1]
    records["z"] = coords[:, 2]
    records["idx"] = np.arange(start_idx, start_idx + n, dtype=np.uint64)
    return records


def _bucket_ids(records: np.ndarray, nbuckets: int, salt: int = 0) -> np.ndarray:
    conf = records["conf"].astype(np.uint64)
    rot = records["rot"].astype(np.uint64)
    x = records["x"].view(np.uint16).astype(np.uint64)
    y = records["y"].view(np.uint16).astype(np.uint64)
    z = records["z"].view(np.uint16).astype(np.uint64)

    h = np.uint64(0x9E3779B97F4A7C15 ^ salt)
    h ^= conf * np.uint64(0xBF58476D1CE4E5B9)
    h ^= rot * np.uint64(0x94D049BB133111EB)
    h ^= x * np.uint64(0xD6E8FEB86659FD93)
    h ^= y * np.uint64(0xA0761D6478BD642F)
    h ^= z * np.uint64(0xE7037ED1A0B428DB)
    return (h & np.uint64(nbuckets - 1)).astype(np.uint32)


def _partition_pose_set(
    pairs: list[tuple[Path, Path]],
    outdir: Path,
    nbuckets: int,
    block_size: int,
) -> np.ndarray:
    counts = np.zeros(nbuckets, dtype=np.uint64)
    writers = BucketWriters(outdir)
    global_idx = 0

    try:
        for pose_path, offset_path in pairs:
            offset_table = load_offset_table(offset_path)
            with open_pose_array(pose_path) as pose_arr:
                for start in range(0, pose_arr.shape[0], block_size):
                    stop = min(start + block_size, pose_arr.shape[0])
                    block = np.asarray(pose_arr[start:stop], dtype=np.uint16)
                    records = _make_records(block, offset_table, global_idx)
                    buckets = _bucket_ids(records, nbuckets)

                    unique_buckets = np.unique(buckets)
                    for bucket in unique_buckets:
                        mask = buckets == bucket
                        subset = records[mask]
                        writers.write(int(bucket), subset)
                        counts[int(bucket)] += np.uint64(subset.shape[0])

                    global_idx += block.shape[0]
    finally:
        writers.close()

    return counts


def _partition_existing_records(
    src: Path,
    outdir: Path,
    nbuckets: int,
    block_records: int,
    salt: int,
) -> np.ndarray:
    counts = np.zeros(nbuckets, dtype=np.uint64)
    if not src.exists() or src.stat().st_size == 0:
        return counts

    writers = BucketWriters(outdir)
    try:
        with src.open("rb") as handle:
            while True:
                records = np.fromfile(handle, dtype=RECORD_DTYPE, count=block_records)
                if records.size == 0:
                    break
                buckets = _bucket_ids(records, nbuckets, salt=salt)
                for bucket in np.unique(buckets):
                    mask = buckets == bucket
                    subset = records[mask]
                    writers.write(int(bucket), subset)
                    counts[int(bucket)] += np.uint64(subset.shape[0])
    finally:
        writers.close()
    return counts


def _match_loaded_records(
    records_a: np.ndarray,
    records_b: np.ndarray,
) -> np.ndarray:
    if records_a.size == 0 or records_b.size == 0:
        return np.empty((0,), dtype=KEY_DTYPE)

    order_a = np.lexsort(
        (
            records_a["idx"],
            records_a["z"],
            records_a["y"],
            records_a["x"],
            records_a["rot"],
            records_a["conf"],
        )
    )
    order_b = np.lexsort(
        (
            records_b["idx"],
            records_b["z"],
            records_b["y"],
            records_b["x"],
            records_b["rot"],
            records_b["conf"],
        )
    )

    records_a = records_a[order_a]
    records_b = records_b[order_b]

    keys_a = np.empty(records_a.shape[0], dtype=KEY_DTYPE)
    keys_b = np.empty(records_b.shape[0], dtype=KEY_DTYPE)
    for field in ("conf", "rot", "x", "y", "z"):
        keys_a[field] = records_a[field]
        keys_b[field] = records_b[field]

    keys_a_void = keys_a.view(f"V{KEY_DTYPE.itemsize}")
    keys_b_void = keys_b.view(f"V{KEY_DTYPE.itemsize}")

    unique_a, start_a, count_a = np.unique(
        keys_a_void, return_index=True, return_counts=True
    )
    unique_b, start_b, count_b = np.unique(
        keys_b_void, return_index=True, return_counts=True
    )

    _, ia, ib = np.intersect1d(unique_a, unique_b, assume_unique=True, return_indices=True)
    if ia.size == 0:
        return np.empty((0,), dtype=KEY_DTYPE)

    take_counts = np.minimum(count_a[ia], count_b[ib]).astype(np.int64, copy=False)
    total = int(take_counts.sum())
    if total == 0:
        return np.empty((0,), dtype=KEY_DTYPE)

    out = np.empty(total, dtype=KEY_DTYPE)
    out_pos = 0
    for pos_a, n_take in zip(ia, take_counts):
        if n_take == 0:
            continue
        start = int(start_a[pos_a])
        end = start + int(n_take)
        subset = records_a[start:end]
        dest = out[out_pos : out_pos + int(n_take)]
        dest["conf"] = subset["conf"]
        dest["rot"] = subset["rot"]
        dest["x"] = subset["x"]
        dest["y"] = subset["y"]
        dest["z"] = subset["z"]
        out_pos += int(n_take)

    return out


def _write_match_records(
    records: np.ndarray,
    writer: PoseStreamAccumulator,
    block_size: int,
) -> int:
    if records.size == 0:
        return 0
    total = int(records.shape[0])
    for start in range(0, total, block_size):
        stop = min(start + block_size, total)
        block = records[start:stop]
        translations = np.empty((block.shape[0], 3), dtype=np.int16)
        translations[:, 0] = block["x"]
        translations[:, 1] = block["y"]
        translations[:, 2] = block["z"]
        writer.add_chunk(block["conf"], block["rot"], translations)
    return total


def _match_bucket_recursive(
    file_a: Path,
    file_b: Path,
    writer: PoseStreamAccumulator,
    max_bucket_bytes: int,
    block_records: int,
    depth: int,
    max_depth: int,
    write_block: int,
) -> int:
    if not file_a.exists() or not file_b.exists():
        return 0
    if file_a.stat().st_size == 0 or file_b.stat().st_size == 0:
        return 0

    size_a = file_a.stat().st_size
    size_b = file_b.stat().st_size
    largest = max(size_a, size_b)

    if largest <= max_bucket_bytes or depth >= max_depth:
        rec_a = np.fromfile(file_a, dtype=RECORD_DTYPE)
        rec_b = np.fromfile(file_b, dtype=RECORD_DTYPE)
        keep = _match_loaded_records(rec_a, rec_b)
        return _write_match_records(keep, writer, write_block)

    with tempfile.TemporaryDirectory(prefix=f"pose_filter_rebucket_{depth}_") as tdir:
        tpath = Path(tdir)
        sub_a = tpath / "a"
        sub_b = tpath / "b"

        # Keep the expected records per bucket conservative for sort/unique overhead.
        nrec_max = max(1, max_bucket_bytes // max(RECORD_BYTES * 3, 1))
        current_recs = math.ceil(largest / RECORD_BYTES)
        sub_nbuckets = _pow2_ceil(math.ceil(current_recs / nrec_max))
        sub_nbuckets = int(min(max(sub_nbuckets, 8), 1 << 16))

        salt = depth + 1
        _partition_existing_records(file_a, sub_a, sub_nbuckets, block_records, salt)
        _partition_existing_records(file_b, sub_b, sub_nbuckets, block_records, salt)

        kept = 0
        for bucket in range(sub_nbuckets):
            ca = _match_bucket_recursive(
                sub_a / f"bucket-{bucket:08d}.bin",
                sub_b / f"bucket-{bucket:08d}.bin",
                writer,
                max_bucket_bytes,
                block_records,
                depth + 1,
                max_depth,
                write_block,
            )
            kept += ca

        return kept


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="filter_identical_poses",
        description=(
            "Intersect two pose directories without loading all poses into memory. "
            "Writes matched poses into an output directory."
        ),
    )
    parser.add_argument(
        "set_a",
        type=_existing_dir,
        help="Input pose directory A.",
    )
    parser.add_argument(
        "set_b",
        type=_existing_dir,
        help="Input pose directory B.",
    )
    parser.add_argument(
        "output",
        help="Output pose directory (must not exist).",
    )
    parser.add_argument(
        "--tmpdir",
        default=None,
        help="Optional scratch directory (default: system temp).",
    )
    parser.add_argument(
        "--block-size",
        type=int,
        default=2_000_000,
        help="Number of poses to process per block while partitioning (default: 2,000,000).",
    )
    parser.add_argument(
        "--memory-gb",
        type=float,
        default=4.0,
        help="Approximate memory budget in GB for matching buckets (default: 4.0).",
    )
    parser.add_argument(
        "--max-poses-per-chunk",
        type=int,
        default=2**32,
        help="Maximum number of poses per output file pair (default: 2**32).",
    )
    parser.add_argument(
        "--max-depth",
        type=int,
        default=4,
        help="Maximum recursive rebucketing depth (default: 4).",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.block_size <= 0:
        raise ValueError("--block-size must be positive")
    if args.memory_gb <= 0:
        raise ValueError("--memory-gb must be positive")
    if args.max_depth < 0:
        raise ValueError("--max-depth must be >= 0")
    if args.max_poses_per_chunk <= 0:
        raise ValueError("--max-poses-per-chunk must be positive")

    outdir = Path(args.output)
    if outdir.exists():
        raise ValueError(f"Output directory already exists: {outdir}")

    set_a = discover_pose_pairs(Path(args.set_a))
    set_b = discover_pose_pairs(Path(args.set_b))

    total_a = _count_poses(set_a)
    total_b = _count_poses(set_b)
    if total_a == 0 or total_b == 0:
        writer = PoseStreamAccumulator(outdir, max_poses_per_chunk=args.max_poses_per_chunk)
        writer.finish()
        print(f"set-a poses: {total_a}")
        print(f"set-b poses: {total_b}")
        print("matches: 0")
        return 0

    mem_bytes = int(args.memory_gb * (1024**3))
    # Heuristic: sorting + unique over structured arrays can briefly need ~3x record bytes.
    target_records_per_bucket = max(1, mem_bytes // (RECORD_BYTES * 3))
    nbuckets = _pow2_ceil(math.ceil(max(total_a, total_b) / target_records_per_bucket))
    nbuckets = int(min(max(nbuckets, 8), 1 << 20))
    max_bucket_bytes = target_records_per_bucket * RECORD_BYTES

    temp_parent = Path(args.tmpdir) if args.tmpdir else None
    with tempfile.TemporaryDirectory(prefix="pose_filter_", dir=temp_parent) as tdir:
        tpath = Path(tdir)
        bucket_a = tpath / "a"
        bucket_b = tpath / "b"

        print(f"set-a poses: {total_a}")
        print(f"set-b poses: {total_b}")
        print(f"buckets: {nbuckets}")

        _partition_pose_set(set_a, bucket_a, nbuckets, args.block_size)
        _partition_pose_set(set_b, bucket_b, nbuckets, args.block_size)
        writer = PoseStreamAccumulator(outdir, max_poses_per_chunk=args.max_poses_per_chunk)
        kept = 0
        try:
            rebucket_block_records = max(1, args.block_size // 4)
            write_block = max(1, args.block_size)
            for bucket in range(nbuckets):
                file_a = bucket_a / f"bucket-{bucket:08d}.bin"
                file_b = bucket_b / f"bucket-{bucket:08d}.bin"
                kept += _match_bucket_recursive(
                    file_a,
                    file_b,
                    writer,
                    max_bucket_bytes=max_bucket_bytes,
                    block_records=rebucket_block_records,
                    depth=0,
                    max_depth=args.max_depth,
                    write_block=write_block,
                )
            writer.finish()
        finally:
            writer.cleanup()

    print(f"matches: {kept}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
