import os
import sys
from pathlib import Path

import numpy as np

sys.path.append(os.path.join(os.path.dirname(__file__), "..", "code"))
from filter_identical_poses import main


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


def test_filter_identical_poses_handles_different_offset_means(tmp_path: Path) -> None:
    set_a_pose = tmp_path / "a-poses.npy"
    set_a_off = tmp_path / "a-offsets.dat"
    set_b_pose = tmp_path / "b-poses.npy"
    set_b_off = tmp_path / "b-offsets.dat"

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

    _write_pose_pair(set_a_pose, set_a_off, poses_a, np.array([10, 20, 30]), offsets_a)
    _write_pose_pair(set_b_pose, set_b_off, poses_b, np.array([0, 0, 0]), offsets_b)

    out_a = tmp_path / "keep-a.npy"
    out_b = tmp_path / "keep-b.npy"
    rc = main(
        [
            "--set-a",
            str(set_a_pose),
            str(set_a_off),
            "--set-b",
            str(set_b_pose),
            str(set_b_off),
            "--out-a",
            str(out_a),
            "--out-b",
            str(out_b),
            "--memory-gb",
            "0.01",
            "--block-size",
            "2",
        ]
    )
    assert rc == 0

    keep_a = np.load(out_a)
    keep_b = np.load(out_b)

    assert np.array_equal(np.sort(keep_a), np.array([0, 1], dtype=np.uint64))
    assert np.array_equal(np.sort(keep_b), np.array([0, 2], dtype=np.uint64))


def test_filter_identical_poses_accepts_directory_and_list_file(tmp_path: Path) -> None:
    # Set A as split directory
    set_a_dir = tmp_path / "set-a"
    set_a_dir.mkdir()
    offsets = np.array([[1, 2, 3], [4, 5, 6]], dtype=np.int16)

    _write_pose_pair(
        set_a_dir / "poses-0.npy",
        set_a_dir / "offsets-0.dat",
        np.array([[10, 20, 0]], dtype=np.uint16),
        np.array([0, 0, 0], dtype=np.int16),
        offsets,
    )
    _write_pose_pair(
        set_a_dir / "poses-1.npy",
        set_a_dir / "offsets-1.dat",
        np.array([[11, 21, 1]], dtype=np.uint16),
        np.array([0, 0, 0], dtype=np.int16),
        offsets,
    )

    # Set B as list-of-two-files input
    set_b_pose = tmp_path / "b-poses.npy"
    set_b_off = tmp_path / "b-offsets.dat"
    _write_pose_pair(
        set_b_pose,
        set_b_off,
        np.array([[10, 20, 0], [50, 60, 1]], dtype=np.uint16),
        np.array([0, 0, 0], dtype=np.int16),
        offsets,
    )

    list_file = tmp_path / "set-b.list"
    list_file.write_text(f"{set_b_pose} {set_b_off}\n")

    out_a = tmp_path / "dir-list-keep-a.npy"
    out_b = tmp_path / "dir-list-keep-b.npy"
    rc = main(
        [
            "--set-a",
            str(set_a_dir),
            "--set-b",
            str(list_file),
            "--out-a",
            str(out_a),
            "--out-b",
            str(out_b),
            "--memory-gb",
            "0.01",
        ]
    )
    assert rc == 0

    keep_a = np.load(out_a)
    keep_b = np.load(out_b)

    assert np.array_equal(keep_a, np.array([0], dtype=np.uint64))
    assert np.array_equal(keep_b, np.array([0], dtype=np.uint64))
