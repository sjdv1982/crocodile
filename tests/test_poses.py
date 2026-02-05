import os
import sys

import tempfile

import numpy as np

here = os.path.dirname(__file__)
sys.path.append(os.path.join(here, "..", "code"))
sys.path.append(here)
from pose_test_data import generate_pose_test_data
from poses import pack_all_poses, read_pose_files, unpack_poses, write_pose_files


def consume_pose_test_data(data: dict[str, object]) -> None:
    conf_indices = data["conf_indices"]
    rot_indices = data["rot_indices"]
    disp_indices, p_indices, rounded, reverse_map = data["offsets_tuple"]
    tinds, tdata = data["gathered"]

    conf_expanded = conf_indices[tinds]
    rot_expanded = rot_indices[tinds]

    packed = pack_all_poses(
        conf_expanded,
        rot_expanded,
        (disp_indices, p_indices, rounded, reverse_map),
    )
    assert len(packed) == 1
    poses, mean_offset, offsets = packed[0]
    conf2, rot2, off_idx2, offset_table = unpack_poses(poses, mean_offset, offsets)

    assert np.array_equal(conf2, conf_expanded)
    assert np.array_equal(rot2, rot_expanded)
    reconstructed = offset_table[off_idx2.astype(np.int64)]
    assert np.array_equal(reconstructed, tdata)


def test_poses():
    data = generate_pose_test_data()
    consume_pose_test_data(data)


def test_poses_disk_roundtrip():
    data = generate_pose_test_data()
    conf_indices = data["conf_indices"]
    rot_indices = data["rot_indices"]
    disp_indices, p_indices, rounded, reverse_map = data["offsets_tuple"]
    tinds, tdata = data["gathered"]

    conf_expanded = conf_indices[tinds]
    rot_expanded = rot_indices[tinds]

    packed = pack_all_poses(
        conf_expanded,
        rot_expanded,
        (disp_indices, p_indices, rounded, reverse_map),
    )

    with tempfile.TemporaryDirectory(prefix="pose_test_") as tmpdir:
        write_pose_files(tmpdir, packed)
        loaded = read_pose_files(tmpdir)

    assert len(loaded) == len(packed)
    for (poses_a, mean_a, offsets_a), (poses_b, mean_b, offsets_b) in zip(
        packed, loaded
    ):
        assert np.array_equal(poses_a, poses_b)
        assert np.array_equal(mean_a, mean_b)
        assert np.array_equal(offsets_a, offsets_b)

    poses, mean_offset, offsets = loaded[0]
    conf2, rot2, off_idx2, offset_table = unpack_poses(poses, mean_offset, offsets)
    reconstructed = offset_table[off_idx2.astype(np.int64)]
    assert np.array_equal(reconstructed, tdata)
