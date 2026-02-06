import io
from pathlib import Path
import tempfile

import numpy as np

from offsets import expand_discrete_offsets, gather_discrete_offsets


def _as_uint16_array(name: str, arr: np.ndarray) -> np.ndarray:
    arr = np.asarray(arr)
    if arr.ndim != 1:
        raise ValueError(f"{name} must be a 1D array")
    if arr.size == 0:
        return arr.astype(np.uint16, copy=False)
    if arr.min() < 0 or arr.max() > np.iinfo(np.uint16).max:
        raise ValueError(f"{name} values must fit in uint16")
    return arr.astype(np.uint16, copy=False)


def _finalize_chunk(
    results: list[tuple[np.ndarray, np.ndarray, np.ndarray]],
    conf_list: list[int],
    rot_list: list[int],
    off_idx_list: list[int],
    offsets_list: list[np.ndarray],
) -> None:
    if not conf_list:
        return

    offsets_arr = np.array(offsets_list, dtype=np.int16)
    max_xyz = offsets_arr.max(axis=0).astype(np.int32)
    mean_offset = np.clip(max_xyz - 127, -32768, 32767).astype(np.int16)
    offsets_int8 = (offsets_arr - mean_offset).astype(np.int8)
    if offsets_int8.max() > 127 or offsets_int8.min() < -128:
        raise ValueError("Offsets exceed int8 range after mean subtraction")

    offsets_uint8 = offsets_int8.view(np.uint8)

    poses = np.column_stack(
        (
            np.array(conf_list, dtype=np.uint16),
            np.array(rot_list, dtype=np.uint16),
            np.array(off_idx_list, dtype=np.uint16),
        )
    )
    results.append((poses, mean_offset, offsets_uint8))


def _pack_expanded_poses_slow(
    conf: np.ndarray,
    rot: np.ndarray,
    translations: np.ndarray,
) -> list[tuple[np.ndarray, np.ndarray, np.ndarray]]:
    results: list[tuple[np.ndarray, np.ndarray, np.ndarray]] = []
    conf_list: list[int] = []
    rot_list: list[int] = []
    off_idx_list: list[int] = []
    offsets_list: list[np.ndarray] = []
    offset_map: dict[tuple[int, int, int], int] = {}
    min_xyz: np.ndarray | None = None
    max_xyz: np.ndarray | None = None
    pose_count = 0

    for i in range(len(translations)):
        trans = translations[i].astype(np.int16, copy=False)
        key = (int(trans[0]), int(trans[1]), int(trans[2]))
        while True:
            new_offset = key not in offset_map
            if new_offset:
                new_min = trans.astype(np.int32)
                new_max = trans.astype(np.int32)
                if min_xyz is not None:
                    new_min = np.minimum(min_xyz, new_min)
                    new_max = np.maximum(max_xyz, new_max)
                spread = new_max - new_min
                too_wide = np.any(spread > 255)
                too_many_offsets = len(offsets_list) + 1 > 2**16
            else:
                too_wide = False
                too_many_offsets = False

            too_many_poses = pose_count + 1 > 2**32
            need_new_chunk = too_wide or too_many_offsets or too_many_poses

            if need_new_chunk and pose_count > 0:
                _finalize_chunk(results, conf_list, rot_list, off_idx_list, offsets_list)
                conf_list = []
                rot_list = []
                off_idx_list = []
                offsets_list = []
                offset_map = {}
                min_xyz = None
                max_xyz = None
                pose_count = 0
                continue

            if need_new_chunk and pose_count == 0:
                raise ValueError("Single offset violates packing constraints")

            break

        if new_offset:
            offset_map[key] = len(offsets_list)
            offsets_list.append(trans.copy())
            if min_xyz is None:
                min_xyz = trans.astype(np.int32)
                max_xyz = trans.astype(np.int32)
            else:
                min_xyz = np.minimum(min_xyz, trans.astype(np.int32))
                max_xyz = np.maximum(max_xyz, trans.astype(np.int32))

        conf_list.append(int(conf[i]))
        rot_list.append(int(rot[i]))
        off_idx_list.append(offset_map[key])
        pose_count += 1

    _finalize_chunk(results, conf_list, rot_list, off_idx_list, offsets_list)
    return results


def pack_expanded_poses(
    conformer_indices: np.ndarray,
    rotamer_indices: np.ndarray,
    translations: np.ndarray,
) -> list[tuple[np.ndarray, np.ndarray, np.ndarray]]:
    """
    Pack already-expanded poses and translations into the on-disk pose format.
    """
    conf = _as_uint16_array("conformer_indices", conformer_indices)
    rot = _as_uint16_array("rotamer_indices", rotamer_indices)
    if conf.shape != rot.shape:
        raise ValueError("conformer_indices and rotamer_indices must have same shape")

    translations = np.asarray(translations, dtype=np.int16)
    if translations.ndim != 2 or translations.shape[1] != 3:
        raise ValueError("translations must be a [N,3] array")
    if len(translations) != len(conf):
        raise ValueError("translations length must match conformer/rotamer indices")
    if len(translations) == 0:
        return []

    if len(translations) <= 2**32:
        min_xyz = translations.min(axis=0).astype(np.int32)
        max_xyz = translations.max(axis=0).astype(np.int32)
        spread = max_xyz - min_xyz
        if np.all(spread <= 255):
            translations_c = np.ascontiguousarray(translations, dtype=np.int16)
            packed_dtype = np.dtype((np.void, translations_c.dtype.itemsize * 3))
            packed_offsets = translations_c.view(packed_dtype).ravel()
            unique_packed, inverse = np.unique(packed_offsets, return_inverse=True)
            unique_offsets = unique_packed.view(np.int16).reshape(-1, 3)
            if len(unique_offsets) <= 2**16:
                max_xyz = unique_offsets.max(axis=0).astype(np.int32)
                mean_offset = np.clip(max_xyz - 127, -32768, 32767).astype(np.int16)
                offsets_int16 = unique_offsets.astype(np.int16)
                offsets_int8 = (offsets_int16 - mean_offset).astype(np.int8)
                if offsets_int8.max() <= 127 and offsets_int8.min() >= -128:
                    poses = np.column_stack(
                        (
                            conf,
                            rot,
                            inverse.astype(np.uint16, copy=False),
                        )
                    )
                    offsets_uint8 = offsets_int8.view(np.uint8)
                    return [(poses, mean_offset, offsets_uint8)]

    return _pack_expanded_poses_slow(conf, rot, translations)


def pack_poses_from_discrete_offsets(
    conformer_indices: np.ndarray,
    rotamer_indices: np.ndarray,
    offsets_tuple: tuple[np.ndarray, np.ndarray, np.ndarray, dict],
    *,
    reverse_table: np.ndarray | None = None,
) -> list[tuple[np.ndarray, np.ndarray, np.ndarray]]:
    """
    Pack poses directly from flattened discrete-offset indices.

    `conformer_indices` and `rotamer_indices` are indexed by displacement index.
    `offsets_tuple` is `(disp_indices, p_indices, rounded, reverse_map)` where
    `disp_indices` and `p_indices` are already flattened candidate assignments.
    """
    conf_by_disp = _as_uint16_array("conformer_indices", conformer_indices)
    rot_by_disp = _as_uint16_array("rotamer_indices", rotamer_indices)
    if conf_by_disp.shape != rot_by_disp.shape:
        raise ValueError("conformer_indices and rotamer_indices must have same shape")

    disp_indices, p_indices, rounded, reverse_map = offsets_tuple
    disp_indices = np.asarray(disp_indices)
    p_indices = np.asarray(p_indices)
    rounded = np.asarray(rounded, dtype=np.int16)

    if disp_indices.ndim != 1 or p_indices.ndim != 1:
        raise ValueError("disp_indices and p_indices must be 1D arrays")
    if len(disp_indices) != len(p_indices):
        raise ValueError("disp_indices and p_indices must have the same length")
    if rounded.ndim != 2 or rounded.shape[1] != 3:
        raise ValueError("rounded must be a [N,3] array")
    if len(disp_indices) == 0:
        return []

    if disp_indices.min() < 0 or disp_indices.max() >= len(conf_by_disp):
        raise ValueError("disp_indices contains out-of-range displacement index")

    if reverse_table is None:
        max_p_index = max(reverse_map.keys()) if reverse_map else -1
        reverse_table = np.empty((max_p_index + 1, 3), dtype=np.int16)
        for p_index, xyz in reverse_map.items():
            reverse_table[p_index] = xyz
    else:
        reverse_table = np.asarray(reverse_table, dtype=np.int16)
        if reverse_table.ndim != 2 or reverse_table.shape[1] != 3:
            raise ValueError("reverse_table must be a [P,3] array")

    if p_indices.min() < 0 or p_indices.max() >= len(reverse_table):
        raise ValueError("p_indices contains out-of-range P index")

    disp_indices = disp_indices.astype(np.int64, copy=False)
    p_indices = p_indices.astype(np.int64, copy=False)
    expanded_conf = conf_by_disp[disp_indices]
    expanded_rot = rot_by_disp[disp_indices]
    expanded_offsets = rounded[disp_indices] + reverse_table[p_indices]
    return pack_expanded_poses(expanded_conf, expanded_rot, expanded_offsets)


class PoseStreamAccumulator:
    def __init__(
        self,
        outdir: str | Path,
        *,
        max_poses_per_chunk: int = 2**32,
    ) -> None:
        self._outdir = Path(outdir)
        self._max_poses_per_chunk = int(max_poses_per_chunk)
        if self._max_poses_per_chunk <= 0:
            raise ValueError("max_poses_per_chunk must be positive")
        if self._max_poses_per_chunk > 2**32:
            raise ValueError("max_poses_per_chunk must be <= 2**32")
        self._chunk_index = 1
        self._written_chunks: list[tuple[Path, Path]] = []
        self._pose_count = 0
        self.total_poses = 0
        self._new_tempfile()
        self._offset_map: dict[tuple[int, int, int], int] = {}
        self._offsets: list[np.ndarray] = []
        self._min_xyz: np.ndarray | None = None
        self._max_xyz: np.ndarray | None = None

    def _new_tempfile(self) -> None:
        self._tmp = tempfile.NamedTemporaryFile(
            prefix="poses_stream_", suffix=".bin", delete=False
        )
        self._tmp_path = Path(self._tmp.name)
        self._tmp.close()

    def _current_paths(self) -> tuple[Path, Path]:
        return (
            self._outdir / f"poses-{self._chunk_index}.npy",
            self._outdir / f"offsets-{self._chunk_index}.dat",
        )

    def _register_offset(self, xyz: np.ndarray) -> int:
        key = (int(xyz[0]), int(xyz[1]), int(xyz[2]))
        existing = self._offset_map.get(key)
        if existing is not None:
            return existing

        index = len(self._offsets)
        if index >= 2**16:
            raise ValueError("Too many unique offsets (>65536) for single chunk output")

        xyz32 = xyz.astype(np.int32)
        if self._min_xyz is None:
            new_min = xyz32
            new_max = xyz32
        else:
            new_min = np.minimum(self._min_xyz, xyz32)
            new_max = np.maximum(self._max_xyz, xyz32)

        if np.any((new_max - new_min) > 255):
            raise ValueError("Offset spread exceeds 255 for single chunk output")

        self._offset_map[key] = index
        self._offsets.append(xyz.astype(np.int16, copy=True))
        self._min_xyz = new_min
        self._max_xyz = new_max
        return index

    def _try_add_batch(
        self,
        conformer_indices: np.ndarray,
        rotamer_indices: np.ndarray,
        translations: np.ndarray,
    ) -> bool:
        conf = _as_uint16_array("conformer_indices", conformer_indices)
        rot = _as_uint16_array("rotamer_indices", rotamer_indices)
        if conf.shape != rot.shape:
            raise ValueError("conformer_indices and rotamer_indices must have same shape")

        translations = np.asarray(translations, dtype=np.int16)
        if translations.ndim != 2 or translations.shape[1] != 3:
            raise ValueError("translations must be a [N,3] array")
        if len(translations) != len(conf):
            raise ValueError("translations length must match conformer/rotamer indices")
        if len(translations) == 0:
            return True

        if self._pose_count + len(translations) > self._max_poses_per_chunk:
            return False

        translations_c = np.ascontiguousarray(translations, dtype=np.int16)
        packed_dtype = np.dtype((np.void, translations_c.dtype.itemsize * 3))
        packed_offsets = translations_c.view(packed_dtype).ravel()
        unique_packed, inverse = np.unique(packed_offsets, return_inverse=True)
        unique_offsets = unique_packed.view(np.int16).reshape(-1, 3)

        new_offsets: list[np.ndarray] = []
        new_min = None
        new_max = None
        unique_to_global = np.empty(len(unique_offsets), dtype=np.uint16)
        for idx, xyz in enumerate(unique_offsets):
            key = (int(xyz[0]), int(xyz[1]), int(xyz[2]))
            existing = self._offset_map.get(key)
            if existing is not None:
                unique_to_global[idx] = existing
                continue
            new_offsets.append(xyz)
            unique_to_global[idx] = len(self._offsets) + len(new_offsets) - 1
            xyz32 = xyz.astype(np.int32)
            if new_min is None:
                new_min = xyz32
                new_max = xyz32
            else:
                new_min = np.minimum(new_min, xyz32)
                new_max = np.maximum(new_max, xyz32)

        if len(self._offsets) + len(new_offsets) > 2**16:
            return False

        if new_min is not None:
            if self._min_xyz is None:
                combined_min = new_min
                combined_max = new_max
            else:
                combined_min = np.minimum(self._min_xyz, new_min)
                combined_max = np.maximum(self._max_xyz, new_max)
            if np.any((combined_max - combined_min) > 255):
                return False

        for xyz in new_offsets:
            key = (int(xyz[0]), int(xyz[1]), int(xyz[2]))
            self._offset_map[key] = len(self._offsets)
            self._offsets.append(xyz.astype(np.int16, copy=True))
            if self._min_xyz is None:
                self._min_xyz = xyz.astype(np.int32)
                self._max_xyz = xyz.astype(np.int32)
            else:
                self._min_xyz = np.minimum(self._min_xyz, xyz.astype(np.int32))
                self._max_xyz = np.maximum(self._max_xyz, xyz.astype(np.int32))

        offset_indices = unique_to_global[inverse]
        poses_chunk = np.column_stack((conf, rot, offset_indices))
        with self._tmp_path.open("ab") as handle:
            poses_chunk.tofile(handle)
        self._pose_count += len(translations)
        self.total_poses += len(translations)
        return True

    def add_chunk(
        self,
        conformer_indices: np.ndarray,
        rotamer_indices: np.ndarray,
        translations: np.ndarray,
    ) -> None:
        queue = [(conformer_indices, rotamer_indices, translations)]
        while queue:
            conf, rot, trans = queue.pop(0)
            if len(trans) == 0:
                continue
            if self._try_add_batch(conf, rot, trans):
                continue
            if self._pose_count > 0:
                self._flush_current()
                queue.insert(0, (conf, rot, trans))
            else:
                if len(trans) == 1:
                    raise ValueError("Single pose violates packing constraints")
                mid = len(trans) // 2
                queue.insert(0, (conf[mid:], rot[mid:], trans[mid:]))
                queue.insert(0, (conf[:mid], rot[:mid], trans[:mid]))

    def _flush_current(self) -> None:
        if self._pose_count == 0:
            self.cleanup()
            self._new_tempfile()
            return

        poses_path, offsets_path = self._current_paths()
        self._outdir.mkdir(parents=True, exist_ok=True)

        poses_mmap = np.lib.format.open_memmap(
            poses_path,
            mode="w+",
            dtype=np.uint16,
            shape=(self._pose_count, 3),
        )
        chunk_rows = 5_000_000
        with self._tmp_path.open("rb") as handle:
            row = 0
            while row < self._pose_count:
                nrows = min(chunk_rows, self._pose_count - row)
                buf = np.fromfile(handle, dtype=np.uint16, count=nrows * 3)
                if len(buf) != nrows * 3:
                    raise ValueError("Unexpected EOF while materializing poses.npy")
                poses_mmap[row : row + nrows] = buf.reshape((nrows, 3))
                row += nrows
        del poses_mmap

        offsets_arr = np.array(self._offsets, dtype=np.int16)
        max_xyz = offsets_arr.max(axis=0).astype(np.int32)
        mean_offset = np.clip(max_xyz - 127, -32768, 32767).astype(np.int16)
        offsets_int8 = (offsets_arr - mean_offset).astype(np.int8)
        if offsets_int8.max() > 127 or offsets_int8.min() < -128:
            raise ValueError("Offsets exceed int8 range after mean subtraction")
        _write_offsets_file(offsets_path, mean_offset, offsets_int8.view(np.uint8))

        self._written_chunks.append((poses_path, offsets_path))
        self._chunk_index += 1
        self._pose_count = 0
        self._offset_map = {}
        self._offsets = []
        self._min_xyz = None
        self._max_xyz = None
        self.cleanup()
        self._new_tempfile()

    def finish(self) -> list[tuple[Path, Path]]:
        if self.total_poses == 0:
            poses_path, offsets_path = self._current_paths()
            self._outdir.mkdir(parents=True, exist_ok=True)
            np.save(poses_path, np.empty((0, 3), dtype=np.uint16))
            _write_offsets_file(
                offsets_path,
                np.zeros((3,), dtype=np.int16),
                np.empty((0, 3), dtype=np.uint8),
            )
            self.cleanup()
            return [(poses_path, offsets_path)]

        self._flush_current()
        self.cleanup()
        return self._written_chunks

    def cleanup(self) -> None:
        if self._tmp_path.exists():
            self._tmp_path.unlink()


def pack_all_poses(
    conformer_indices: np.ndarray,
    rotamer_indices: np.ndarray,
    offsets_tuple: tuple[np.ndarray, np.ndarray, np.ndarray, dict],
) -> list[tuple[np.ndarray, np.ndarray, np.ndarray]]:
    """
    Pack poses into the on-disk representation (poses.npy + offsets.dat).

    Inputs are expanded per translation. Offsets are deduplicated and stored
    in a table, with poses referencing them by index.
    """
    conf = _as_uint16_array("conformer_indices", conformer_indices)
    rot = _as_uint16_array("rotamer_indices", rotamer_indices)
    if conf.shape != rot.shape:
        raise ValueError("conformer_indices and rotamer_indices must have same shape")

    disp_indices, p_indices, rounded, reverse_map = offsets_tuple
    translation_lists = expand_discrete_offsets(
        disp_indices, p_indices, rounded, reverse_map
    )
    if not translation_lists:
        return []

    _, tdata = gather_discrete_offsets(translation_lists)
    if len(conf) != len(tdata):
        raise ValueError(
            "Input indices length must match gathered translation length"
        )
    return pack_expanded_poses(conf, rot, tdata)


def unpack_poses(
    poses: np.ndarray, mean_offset: np.ndarray, offsets: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Unpack a pose tuple into indices and a deduplicated offset table.
    """
    poses = np.asarray(poses)
    if poses.ndim != 2 or poses.shape[1] != 3:
        raise ValueError("poses must be a [N,3] array")

    mean_offset = np.asarray(mean_offset, dtype=np.int16)
    if mean_offset.shape != (3,):
        raise ValueError("mean_offset must have shape (3,)")

    offsets_uint8 = np.asarray(offsets, dtype=np.uint8)
    if offsets_uint8.ndim != 2 or offsets_uint8.shape[1] != 3:
        raise ValueError("offsets must be a [K,3] array")

    offsets_int8 = offsets_uint8.view(np.int8)
    offset_table = offsets_int8.astype(np.int16) + mean_offset.astype(np.int16)

    conformer_indices = poses[:, 0].astype(np.uint16, copy=False)
    rotamer_indices = poses[:, 1].astype(np.uint16, copy=False)
    offset_indices = poses[:, 2].astype(np.uint16, copy=False)

    return conformer_indices, rotamer_indices, offset_indices, offset_table


def _write_offsets_file(path: Path, mean_offset: np.ndarray, offsets: np.ndarray) -> None:
    mean_offset = np.asarray(mean_offset, dtype=np.int16)
    if mean_offset.shape != (3,):
        raise ValueError("mean_offset must have shape (3,)")

    offsets_uint8 = np.asarray(offsets, dtype=np.uint8)
    if offsets_uint8.ndim != 2 or offsets_uint8.shape[1] != 3:
        raise ValueError("offsets must be a [K,3] array")

    with path.open("wb") as handle:
        handle.write(mean_offset.tobytes(order="C"))
        handle.write(offsets_uint8.tobytes(order="C"))


def _read_offsets_file(path: Path) -> tuple[np.ndarray, np.ndarray]:
    raw = path.read_bytes()
    if len(raw) < 6:
        raise ValueError("offsets.dat is too small")
    mean_offset = np.frombuffer(raw[:6], dtype=np.int16).copy()
    remainder = raw[6:]
    if len(remainder) % 3 != 0:
        raise ValueError("offsets.dat remainder length must be divisible by 3")
    offsets_uint8 = np.frombuffer(remainder, dtype=np.uint8).copy()
    offsets_uint8 = offsets_uint8.reshape((-1, 3))
    return mean_offset, offsets_uint8


def write_pose_files(
    outdir: str | Path, packed: list[tuple[np.ndarray, np.ndarray, np.ndarray]]
) -> list[tuple[Path, Path]]:
    """
    Write poses-*.npy and offsets-*.dat files to disk (1-based indices).
    """
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    written: list[tuple[Path, Path]] = []
    for i, (poses, mean_offset, offsets) in enumerate(packed, start=1):
        poses_path = outdir / f"poses-{i}.npy"
        offsets_path = outdir / f"offsets-{i}.dat"

        np.save(poses_path, np.asarray(poses, dtype=np.uint16))
        _write_offsets_file(offsets_path, mean_offset, offsets)
        written.append((poses_path, offsets_path))

    return written


def pose_index_from_name(name: str) -> int | None:
    if name.endswith(".npy.zst"):
        base = name[:-4]
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


def discover_pose_pairs(directory: str | Path) -> list[tuple[Path, Path]]:
    directory = Path(directory)
    if not directory.exists() or not directory.is_dir():
        raise FileNotFoundError(f"Pose directory not found: {directory}")

    pose_files = list(directory.glob("poses-*.npy")) + list(
        directory.glob("poses-*.npy.zst")
    )
    if not pose_files:
        raise FileNotFoundError(f"No pose files found in {directory}")

    indexed: dict[int, Path] = {}
    for pose_path in pose_files:
        index = pose_index_from_name(pose_path.name)
        if index is None:
            continue
        if index not in indexed or indexed[index].name.endswith(".npy.zst"):
            indexed[index] = pose_path

    if not indexed:
        raise FileNotFoundError(f"No poses-*.npy found in {directory}")

    pairs: list[tuple[Path, Path]] = []
    for index in sorted(indexed):
        pose_path = indexed[index]
        offsets_path = directory / f"offsets-{index}.dat"
        if not offsets_path.exists():
            raise FileNotFoundError(f"Missing {offsets_path} for {pose_path}")
        pairs.append((pose_path, offsets_path))
    return pairs


def open_pose_array(path: str | Path) -> np.ndarray:
    path = Path(path)
    if path.name.endswith(".npy.zst"):
        try:
            import zstandard as zstd
        except ImportError as exc:
            raise ImportError("zstandard is required to read .npy.zst pose files") from exc
        with path.open("rb") as compressed:
            dctx = zstd.ZstdDecompressor()
            with dctx.stream_reader(compressed) as reader:
                decompressed = reader.read()
        return np.load(io.BytesIO(decompressed))

    return np.load(path, mmap_mode="r")


def _load_pose_array(path: str | Path) -> np.ndarray:
    return open_pose_array(path)


def load_offset_table(path: str | Path) -> np.ndarray:
    path = Path(path)
    mean_offset, offsets_uint8 = _read_offsets_file(path)
    offsets_int8 = offsets_uint8.view(np.int8)
    return offsets_int8.astype(np.int16) + mean_offset.astype(np.int16)


def read_pose_files(outdir: str | Path) -> list[tuple[np.ndarray, np.ndarray, np.ndarray]]:
    """
    Read poses-*.npy (or poses-*.npy.zst) and offsets-*.dat files from a directory.
    Returns a list of (poses, mean_offset, offsets) tuples.
    """
    results: list[tuple[np.ndarray, np.ndarray, np.ndarray]] = []
    for pose_path, offsets_path in discover_pose_pairs(outdir):
        poses = _load_pose_array(pose_path)
        mean_offset, offsets = _read_offsets_file(offsets_path)
        results.append((poses, mean_offset, offsets))

    return results
