from __future__ import annotations

import argparse
import ast
import math
import tempfile
from collections import OrderedDict
from pathlib import Path
from typing import Iterable

import numpy as np


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

UINT64 = np.dtype("<u8")


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


class IndexWriter:
    def __init__(self, path: Path) -> None:
        self.path = path
        self._handle = path.open("wb")

    def write(self, indices: np.ndarray) -> None:
        if indices.size:
            np.asarray(indices, dtype=np.uint64).tofile(self._handle)

    def close(self) -> None:
        self._handle.close()


def _pow2_ceil(value: int) -> int:
    if value <= 1:
        return 1
    return 1 << (value - 1).bit_length()


def _existing_path(path: str) -> str:
    p = Path(path)
    if not p.exists():
        raise argparse.ArgumentTypeError(f"path does not exist: {path}")
    return path


def _parse_pair_literal(value: str) -> tuple[Path, Path] | None:
    text = value.strip()
    if not text.startswith("["):
        return None
    try:
        parsed = ast.literal_eval(text)
    except (SyntaxError, ValueError):
        return None
    if not isinstance(parsed, (list, tuple)) or len(parsed) != 2:
        return None
    return Path(str(parsed[0])), Path(str(parsed[1]))


def _read_pair_list(path: Path) -> list[tuple[Path, Path]]:
    pairs: list[tuple[Path, Path]] = []
    for line_no, line in enumerate(path.read_text().splitlines(), start=1):
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        fields = stripped.replace(",", " ").split()
        if len(fields) != 2:
            raise ValueError(
                f"Invalid pair list line {line_no} in {path}: expected 2 paths"
            )
        pairs.append((Path(fields[0]), Path(fields[1])))
    if not pairs:
        raise ValueError(f"No path pairs found in list file: {path}")
    return pairs


def _discover_dir_pairs(directory: Path) -> list[tuple[Path, Path]]:
    single_pose = directory / "poses.npy"
    single_offset = directory / "offsets.dat"
    if single_pose.exists():
        if not single_offset.exists():
            raise FileNotFoundError(f"Missing offsets.dat for {single_pose}")
        return [(single_pose, single_offset)]

    pose_files = sorted(directory.glob("poses-*.npy"))
    if not pose_files:
        raise FileNotFoundError(f"No pose files found in {directory}")

    pairs: list[tuple[Path, Path]] = []
    for pose_path in pose_files:
        suffix = pose_path.stem.split("poses-")[-1]
        offset_path = directory / f"offsets-{suffix}.dat"
        if not offset_path.exists():
            raise FileNotFoundError(f"Missing {offset_path} for {pose_path}")
        pairs.append((pose_path, offset_path))
    return pairs


def resolve_pose_set(values: list[str], label: str) -> list[tuple[Path, Path]]:
    if len(values) == 2:
        pair = (Path(values[0]), Path(values[1]))
        return [pair]

    if len(values) == 1:
        literal_pair = _parse_pair_literal(values[0])
        if literal_pair is not None:
            return [literal_pair]

        path = Path(values[0])
        if path.is_dir():
            return _discover_dir_pairs(path)
        if path.is_file():
            suffix = path.suffix.lower()
            if suffix in {".txt", ".list", ".lst"}:
                return _read_pair_list(path)
            raise ValueError(
                f"{label}: single file input must be a list file (.txt/.list/.lst), got {path}"
            )

    raise ValueError(
        f"{label}: provide exactly two paths (poses offsets), or one directory, or one list file"
    )


def _validate_pairs(pairs: list[tuple[Path, Path]], label: str) -> list[tuple[Path, Path]]:
    checked: list[tuple[Path, Path]] = []
    for pose_path, offset_path in pairs:
        if not pose_path.exists() or not pose_path.is_file():
            raise FileNotFoundError(f"{label}: pose file not found: {pose_path}")
        if not offset_path.exists() or not offset_path.is_file():
            raise FileNotFoundError(f"{label}: offsets file not found: {offset_path}")
        checked.append((pose_path, offset_path))
    return checked


def _load_offset_table(path: Path) -> np.ndarray:
    raw = path.read_bytes()
    if len(raw) < 6:
        raise ValueError(f"Offsets file too small: {path}")
    mean = np.frombuffer(raw[:6], dtype=np.int16).copy()
    rem = raw[6:]
    if len(rem) % 3 != 0:
        raise ValueError(f"Invalid offsets remainder length in {path}")
    table_u8 = np.frombuffer(rem, dtype=np.uint8)
    table_u8 = table_u8.reshape((-1, 3))
    return table_u8.view(np.int8).astype(np.int16) + mean.astype(np.int16)


def _count_poses(pairs: Iterable[tuple[Path, Path]]) -> int:
    total = 0
    for pose_path, _ in pairs:
        arr = np.load(pose_path, mmap_mode="r")
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
            pose_arr = np.load(pose_path, mmap_mode="r")
            offset_table = _load_offset_table(offset_path)

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
) -> tuple[np.ndarray, np.ndarray]:
    if records_a.size == 0 or records_b.size == 0:
        return np.empty((0,), dtype=np.uint64), np.empty((0,), dtype=np.uint64)

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
        return np.empty((0,), dtype=np.uint64), np.empty((0,), dtype=np.uint64)

    keep_a: list[np.ndarray] = []
    keep_b: list[np.ndarray] = []
    for pos_a, pos_b in zip(ia, ib):
        n = min(int(count_a[pos_a]), int(count_b[pos_b]))
        if n == 0:
            continue
        a_slice = slice(int(start_a[pos_a]), int(start_a[pos_a]) + n)
        b_slice = slice(int(start_b[pos_b]), int(start_b[pos_b]) + n)
        keep_a.append(records_a["idx"][a_slice])
        keep_b.append(records_b["idx"][b_slice])

    if keep_a:
        out_a = np.concatenate(keep_a).astype(np.uint64, copy=False)
        out_b = np.concatenate(keep_b).astype(np.uint64, copy=False)
        return out_a, out_b

    return np.empty((0,), dtype=np.uint64), np.empty((0,), dtype=np.uint64)


def _match_bucket_recursive(
    file_a: Path,
    file_b: Path,
    out_a: IndexWriter,
    out_b: IndexWriter,
    max_bucket_bytes: int,
    block_records: int,
    depth: int,
    max_depth: int,
) -> tuple[int, int]:
    if not file_a.exists() or not file_b.exists():
        return 0, 0
    if file_a.stat().st_size == 0 or file_b.stat().st_size == 0:
        return 0, 0

    size_a = file_a.stat().st_size
    size_b = file_b.stat().st_size
    largest = max(size_a, size_b)

    if largest <= max_bucket_bytes or depth >= max_depth:
        rec_a = np.fromfile(file_a, dtype=RECORD_DTYPE)
        rec_b = np.fromfile(file_b, dtype=RECORD_DTYPE)
        keep_a, keep_b = _match_loaded_records(rec_a, rec_b)
        out_a.write(keep_a)
        out_b.write(keep_b)
        return int(keep_a.size), int(keep_b.size)

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

        kept_a = 0
        kept_b = 0
        for bucket in range(sub_nbuckets):
            ca, cb = _match_bucket_recursive(
                sub_a / f"bucket-{bucket:08d}.bin",
                sub_b / f"bucket-{bucket:08d}.bin",
                out_a,
                out_b,
                max_bucket_bytes,
                block_records,
                depth + 1,
                max_depth,
            )
            kept_a += ca
            kept_b += cb

        return kept_a, kept_b


def _raw_file_size_in_items(path: Path, dtype: np.dtype) -> int:
    if not path.exists():
        return 0
    size = path.stat().st_size
    if size % dtype.itemsize != 0:
        raise ValueError(f"Corrupt raw file size for {path}")
    return size // dtype.itemsize


def _raw_to_npy(raw_path: Path, out_path: Path) -> int:
    n = _raw_file_size_in_items(raw_path, UINT64)
    out = np.lib.format.open_memmap(out_path, mode="w+", dtype=np.uint64, shape=(n,))
    if n == 0:
        del out
        return 0

    chunk = 4_000_000
    with raw_path.open("rb") as handle:
        start = 0
        while start < n:
            count = min(chunk, n - start)
            part = np.fromfile(handle, dtype=np.uint64, count=count)
            if part.size != count:
                raise RuntimeError("Unexpected EOF while writing output .npy")
            out[start : start + count] = part
            start += count

    del out
    return n


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="filter_identical_poses",
        description=(
            "Intersect two pose sets without loading all poses into memory. "
            "A set can be: two paths (poses + offsets), one directory, "
            "or one list file containing 'poses offsets' pairs."
        ),
    )
    parser.add_argument(
        "--set-a",
        nargs="+",
        required=True,
        type=_existing_path,
        help="Set A specification.",
    )
    parser.add_argument(
        "--set-b",
        nargs="+",
        required=True,
        type=_existing_path,
        help="Set B specification.",
    )
    parser.add_argument(
        "--out-a",
        required=True,
        help="Output .npy file with global indices of matching poses in set A.",
    )
    parser.add_argument(
        "--out-b",
        required=True,
        help="Output .npy file with global indices of matching poses in set B.",
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

    set_a = _validate_pairs(resolve_pose_set(args.set_a, "set-a"), "set-a")
    set_b = _validate_pairs(resolve_pose_set(args.set_b, "set-b"), "set-b")

    total_a = _count_poses(set_a)
    total_b = _count_poses(set_b)
    if total_a == 0 or total_b == 0:
        out_a_path = Path(args.out_a)
        out_b_path = Path(args.out_b)
        np.save(out_a_path, np.empty((0,), dtype=np.uint64))
        np.save(out_b_path, np.empty((0,), dtype=np.uint64))
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

        raw_a = tpath / "matched-a.raw"
        raw_b = tpath / "matched-b.raw"
        writer_a = IndexWriter(raw_a)
        writer_b = IndexWriter(raw_b)
        kept_a = 0
        kept_b = 0

        try:
            rebucket_block_records = max(1, args.block_size // 4)
            for bucket in range(nbuckets):
                file_a = bucket_a / f"bucket-{bucket:08d}.bin"
                file_b = bucket_b / f"bucket-{bucket:08d}.bin"
                ca, cb = _match_bucket_recursive(
                    file_a,
                    file_b,
                    writer_a,
                    writer_b,
                    max_bucket_bytes=max_bucket_bytes,
                    block_records=rebucket_block_records,
                    depth=0,
                    max_depth=args.max_depth,
                )
                kept_a += ca
                kept_b += cb
        finally:
            writer_a.close()
            writer_b.close()

        out_a_count = _raw_to_npy(raw_a, Path(args.out_a))
        out_b_count = _raw_to_npy(raw_b, Path(args.out_b))

    assert kept_a == out_a_count
    assert kept_b == out_b_count

    print(f"matches: {kept_a}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
