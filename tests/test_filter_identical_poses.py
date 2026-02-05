import os
import sys
from pathlib import Path

import numpy as np

sys.path.append(os.path.join(os.path.dirname(__file__), "..", "code"))
from filter_identical_poses import main
from poses import read_pose_files, unpack_poses


def _write_pose_pair(
    pose_path: Path,
    offset_path: Path,
    poses: np.ndarray,
    mean: np.ndarray,
    offset_table_abs: np.ndarray,
) -> None:
    np.save(pose_path, np.asarray(poses, dtype=np.uint16))
    mean = np.asarray(mean, dtype=np.int16)
    offsets = np.asarray(offset_table_abs, dtype=np.int16)
    centered = offsets - mean
    if centered.min() < -128 or centered.max() > 127:
        raise ValueError("offset table cannot be encoded as int8 around mean")
    encoded = centered.astype(np.int8).view(np.uint8)
    with offset_path.open("wb") as handle:
        handle.write(mean.tobytes(order="C"))
        handle.write(encoded.tobytes(order="C"))


def _poses_to_tuples(packed: list[tuple[np.ndarray, np.ndarray, np.ndarray]]) -> list[tuple[int, int, int, int, int]]:
    out: list[tuple[int, int, int, int, int]] = []
    for poses, mean, offsets in packed:
        conf, rot, off_idx, offset_table = unpack_poses(poses, mean, offsets)
        coords = offset_table[off_idx.astype(np.int64)]
        for c, r, xyz in zip(conf, rot, coords):
            out.append((int(c), int(r), int(xyz[0]), int(xyz[1]), int(xyz[2])))
    return out


def test_filter_identical_poses_handles_different_offset_means(tmp_path: Path) -> None:
    set_a_dir = tmp_path / "set-a"
    set_b_dir = tmp_path / "set-b"
    set_a_dir.mkdir()
    set_b_dir.mkdir()

    offsets_a = np.array([[10, 20, 30], [5, 5, 5]], dtype=np.int16)
    offsets_b = np.array([[10, 20, 30], [5, 5, 5], [7, 8, 9]], dtype=np.int16)

    poses_a = np.array(
        [
            [1, 2, 0],
            [3, 4, 1],
            [1, 2, 0],
        ],
        dtype=np.uint16,
    )
    poses_b = np.array(
        [
            [1, 2, 0],
            [9, 9, 2],
            [3, 4, 1],
        ],
        dtype=np.uint16,
    )

    _write_pose_pair(
        set_a_dir / "poses-1.npy",
        set_a_dir / "offsets-1.dat",
        poses_a,
        np.array([10, 20, 30]),
        offsets_a,
    )
    _write_pose_pair(
        set_b_dir / "poses-1.npy",
        set_b_dir / "offsets-1.dat",
        poses_b,
        np.array([0, 0, 0]),
        offsets_b,
    )

    out_dir = tmp_path / "out-poses"
    rc = main(
        [
            str(set_a_dir),
            str(set_b_dir),
            str(out_dir),
            "--memory-gb",
            "0.01",
            "--block-size",
            "2",
        ]
    )
    assert rc == 0

    packed = read_pose_files(out_dir)
    tuples = _poses_to_tuples(packed)
    assert len(tuples) == 2
    assert set(tuples) == {(1, 2, 10, 20, 30), (3, 4, 5, 5, 5)}


def test_filter_identical_poses_accepts_directory(tmp_path: Path) -> None:
    set_a_dir = tmp_path / "set-a"
    set_b_dir = tmp_path / "set-b"
    set_a_dir.mkdir()
    set_b_dir.mkdir()
    offsets = np.array([[1, 2, 3], [4, 5, 6]], dtype=np.int16)

    _write_pose_pair(
        set_a_dir / "poses-1.npy",
        set_a_dir / "offsets-1.dat",
        np.array([[10, 20, 0]], dtype=np.uint16),
        np.array([0, 0, 0], dtype=np.int16),
        offsets,
    )
    _write_pose_pair(
        set_a_dir / "poses-2.npy",
        set_a_dir / "offsets-2.dat",
        np.array([[11, 21, 1]], dtype=np.uint16),
        np.array([0, 0, 0], dtype=np.int16),
        offsets,
    )

    _write_pose_pair(
        set_b_dir / "poses-1.npy",
        set_b_dir / "offsets-1.dat",
        np.array([[10, 20, 0], [50, 60, 1]], dtype=np.uint16),
        np.array([0, 0, 0], dtype=np.int16),
        offsets,
    )

    out_dir = tmp_path / "dir-keep-out"
    rc = main(
        [
            str(set_a_dir),
            str(set_b_dir),
            str(out_dir),
            "--memory-gb",
            "0.01",
        ]
    )
    assert rc == 0

    packed = read_pose_files(out_dir)
    tuples = _poses_to_tuples(packed)
    assert tuples == [(10, 20, 1, 2, 3)]
