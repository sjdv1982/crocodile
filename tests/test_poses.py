import os
import sys

import tempfile

import numpy as np

here = os.path.dirname(__file__)
sys.path.append(os.path.join(here, "..", "code"))
sys.path.append(here)
from pose_test_data import generate_pose_test_data
from poses import (
    PoseStreamAccumulator,
    (
    PoseStreamAccumulator,
    pack_all_poses,
   
    read_pose_files,
   
    unpack_poses,
   
    write_pose_files,
),
)


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


def test_pose_stream_accumulator_emits_canonical_centers_by_default():
    conf = np.array([10, 11, 12, 13], dtype=np.uint16)
    rot = np.array([20, 21, 22, 23], dtype=np.uint16)
    trans = np.array(
        [
            [0, 0, 0],       # center (0,0,0)
            [255, 1, -1],    # center (256,0,0)
            [256, 2, 2],     # center (256,0,0)
            [-129, 0, 0],    # center (-256,0,0)
        ],
        dtype=np.int16,
    )

    with tempfile.TemporaryDirectory(prefix="pose_stream_test_") as tmpdir:
        acc = PoseStreamAccumulator(tmpdir, max_poses_per_chunk=100)
        acc.add_chunk(conf, rot, trans)
        acc.finish()
        loaded = read_pose_files(tmpdir)

    reconstructed_rows = []
    for poses, center, offsets in loaded:
        assert np.all((center.astype(np.int32) % 256) == 0)
        conf_i, rot_i, off_idx_i, offset_table_i = unpack_poses(poses, center, offsets)
        abs_xyz_i = offset_table_i[off_idx_i.astype(np.int64)]
        expected_center_i = ((abs_xyz_i.astype(np.int32) + 128) // 256) * 256
        assert np.all(expected_center_i == center.astype(np.int32))
        reconstructed_rows.append(
            np.column_stack(
                (
                    conf_i.astype(np.int32),
                    rot_i.astype(np.int32),
                    abs_xyz_i.astype(np.int32),
                )
            )
        )

    reconstructed = np.vstack(reconstructed_rows)
    original = np.column_stack(
        (
            conf.astype(np.int32),
            rot.astype(np.int32),
            trans.astype(np.int32),
        )
    )
    recon_sort = np.lexsort(
        (
            reconstructed[:, 4],
            reconstructed[:, 3],
            reconstructed[:, 2],
            reconstructed[:, 1],
            reconstructed[:, 0],
        )
    )
    orig_sort = np.lexsort(
        (original[:, 4], original[:, 3], original[:, 2], original[:, 1], original[:, 0])
    )
    assert np.array_equal(reconstructed[recon_sort], original[orig_sort])


def test_pose_stream_accumulator_zstd_roundtrip():
    data = generate_pose_test_data()
    conf_indices = data["conf_indices"]
    rot_indices = data["rot_indices"]
    tinds, tdata = data["gathered"]

    conf_expanded = conf_indices[tinds]
    rot_expanded = rot_indices[tinds]

    with tempfile.TemporaryDirectory(prefix="pose_stream_zstd_") as tmpdir:
        writer = PoseStreamAccumulator(tmpdir, zstd=True)
        writer.add_chunk(conf_expanded, rot_expanded, tdata)
        written = writer.finish()
        writer.cleanup()

        assert len(written) == 1
        poses_path, offsets_path = written[0]
        assert poses_path.name == "poses-1.npy.zst"
        assert poses_path.exists()
        assert offsets_path.name == "offsets-1.dat"

        loaded = read_pose_files(tmpdir)

    assert len(loaded) == 1
    poses, mean_offset, offsets = loaded[0]
    conf2, rot2, off_idx2, offset_table = unpack_poses(poses, mean_offset, offsets)

    assert np.array_equal(conf2, conf_expanded)
    assert np.array_equal(rot2, rot_expanded)
    reconstructed = offset_table[off_idx2.astype(np.int64)]
    assert np.array_equal(reconstructed, tdata)
