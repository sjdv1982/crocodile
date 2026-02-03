from pathlib import Path

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

    results: list[tuple[np.ndarray, np.ndarray, np.ndarray]] = []
    conf_list: list[int] = []
    rot_list: list[int] = []
    off_idx_list: list[int] = []
    offsets_list: list[np.ndarray] = []
    offset_map: dict[tuple[int, int, int], int] = {}
    min_xyz: np.ndarray | None = None
    max_xyz: np.ndarray | None = None
    pose_count = 0

    for i in range(len(tdata)):
        trans = tdata[i].astype(np.int16, copy=False)
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
                _finalize_chunk(
                    results, conf_list, rot_list, off_idx_list, offsets_list
                )
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
    Write poses.npy and offsets.dat files to disk.

    If packed has a single entry, files are named poses.npy and offsets.dat.
    If multiple entries, files are named poses-{i}.npy and offsets-{i}.dat.
    """
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    written: list[tuple[Path, Path]] = []
    multi = len(packed) > 1
    for i, (poses, mean_offset, offsets) in enumerate(packed):
        if multi:
            poses_path = outdir / f"poses-{i}.npy"
            offsets_path = outdir / f"offsets-{i}.dat"
        else:
            poses_path = outdir / "poses.npy"
            offsets_path = outdir / "offsets.dat"

        np.save(poses_path, np.asarray(poses, dtype=np.uint16))
        _write_offsets_file(offsets_path, mean_offset, offsets)
        written.append((poses_path, offsets_path))

    return written


def read_pose_files(outdir: str | Path) -> list[tuple[np.ndarray, np.ndarray, np.ndarray]]:
    """
    Read poses.npy and offsets.dat files from disk.
    Returns a list of (poses, mean_offset, offsets) tuples.
    """
    outdir = Path(outdir)
    single_pose = outdir / "poses.npy"
    single_offsets = outdir / "offsets.dat"

    if single_pose.exists():
        poses = np.load(single_pose)
        mean_offset, offsets = _read_offsets_file(single_offsets)
        return [(poses, mean_offset, offsets)]

    pose_files = sorted(outdir.glob("poses-*.npy"))
    if not pose_files:
        raise FileNotFoundError("No poses.npy or poses-*.npy found")

    results: list[tuple[np.ndarray, np.ndarray, np.ndarray]] = []
    for pose_path in pose_files:
        suffix = pose_path.stem.split("poses-")[-1]
        offsets_path = outdir / f"offsets-{suffix}.dat"
        if not offsets_path.exists():
            raise FileNotFoundError(f"Missing offsets file for {pose_path.name}")
        poses = np.load(pose_path)
        mean_offset, offsets = _read_offsets_file(offsets_path)
        results.append((poses, mean_offset, offsets))

    return results
