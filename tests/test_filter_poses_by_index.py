import os
import sys
import tempfile
from pathlib import Path

import numpy as np
import pytest

sys.path.append(os.path.join(os.path.dirname(__file__), "..", "code"))
import filter_poses_by_index as filter_module
from filter_poses_by_index import filter_poses_by_index, main
from poses import read_pose_files, unpack_poses


def _write_pose_pair(
    pose_path: Path,
    offset_path: Path,
    rows: list[tuple[int, int, tuple[int, int, int]]],
    *,
    zstd: bool = False,
) -> None:
    poses = np.array(
        [[conf, rot, idx] for idx, (conf, rot, _) in enumerate(rows)], dtype=np.uint16
    )
    offsets = np.array([xyz for _, _, xyz in rows], dtype=np.int16)
    mean = offsets[0].copy()
    centered = offsets - mean
    if centered.min() < -128 or centered.max() > 127:
        raise ValueError("offset table cannot be encoded as int8 around chosen mean")
    encoded = centered.astype(np.int8).view(np.uint8)

    if zstd:
        zstandard = pytest.importorskip("zstandard")
        with tempfile.NamedTemporaryFile(suffix=".npy", delete=False) as handle:
            tmp_npy = Path(handle.name)
        try:
            np.save(tmp_npy, poses)
            with tmp_npy.open("rb") as src, pose_path.open("wb") as dst:
                zstandard.ZstdCompressor().copy_stream(src, dst)
        finally:
            if tmp_npy.exists():
                tmp_npy.unlink()
    else:
        np.save(pose_path, poses)

    with offset_path.open("wb") as handle:
        handle.write(mean.astype(np.int16).tobytes(order="C"))
        handle.write(encoded.tobytes(order="C"))


def _write_pose_dir(
    directory: Path,
    shards: list[list[tuple[int, int, tuple[int, int, int]]]],
    *,
    zstd: bool = False,
) -> list[tuple[int, int, tuple[int, int, int]]]:
    directory.mkdir()
    rows: list[tuple[int, int, tuple[int, int, int]]] = []
    for index, shard_rows in enumerate(shards, start=1):
        suffix = ".npy.zst" if zstd else ".npy"
        _write_pose_pair(
            directory / f"poses-{index}{suffix}",
            directory / f"offsets-{index}.dat",
            shard_rows,
            zstd=zstd,
        )
        rows.extend(shard_rows)
    return rows


def _load_pose_tuples(directory: Path) -> list[tuple[int, int, tuple[int, int, int]]]:
    out: list[tuple[int, int, tuple[int, int, int]]] = []
    for poses, mean, offsets in read_pose_files(directory):
        conf, rot, off_idx, offset_table = unpack_poses(poses, mean, offsets)
        coords = offset_table[off_idx.astype(np.int64)]
        for c, r, xyz in zip(conf, rot, coords):
            out.append((int(c), int(r), (int(xyz[0]), int(xyz[1]), int(xyz[2]))))
    return out


def test_filter_poses_by_index_skips_untouched_zstd_chunks(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    input_dir = tmp_path / "input-zstd"
    rows = _write_pose_dir(
        input_dir,
        [
            [(10, 20, (0, 0, 0)), (11, 21, (1, 1, 1))],
            [(12, 22, (256, 0, 0)), (13, 23, (257, 1, 1))],
            [(14, 24, (-256, 0, 0)), (15, 25, (-255, 1, 1))],
        ],
        zstd=True,
    )
    indices_path = tmp_path / "indices.npy"
    np.save(indices_path, np.array([2, 3], dtype=np.int64))

    real_open = filter_module.open_pose_array
    opened: list[str] = []

    def guarded_open(path: str | Path):
        name = Path(path).name
        opened.append(name)
        if name != "poses-2.npy.zst":
            raise AssertionError(f"unexpected chunk opened: {name}")
        return real_open(path)

    monkeypatch.setattr(filter_module, "open_pose_array", guarded_open)

    output_dir = tmp_path / "filtered-zstd"
    written = filter_poses_by_index(input_dir, indices_path, output_dir)

    assert len(written) == 1
    assert opened == ["poses-2.npy.zst"]
    assert _load_pose_tuples(output_dir) == rows[2:4]


def test_filter_poses_by_index_cli_preserves_order_and_duplicates(tmp_path: Path) -> None:
    input_dir = tmp_path / "input"
    rows = _write_pose_dir(
        input_dir,
        [
            [(10, 20, (0, 0, 0)), (11, 21, (1, 1, 1))],
            [(12, 22, (256, 0, 0)), (13, 23, (257, 1, 1))],
            [(14, 24, (-256, 0, 0)), (15, 25, (-255, 1, 1))],
        ],
    )
    requested = np.array([5, 0, 3, 3, 1], dtype=np.int64)
    indices_path = tmp_path / "indices.npy"
    np.save(indices_path, requested)

    output_dir = tmp_path / "filtered"
    rc = main(
        [
            "--input-dir",
            str(input_dir),
            "--indices",
            str(indices_path),
            "--output-dir",
            str(output_dir),
        ]
    )

    assert rc == 0
    assert _load_pose_tuples(output_dir) == [rows[i] for i in requested]


def test_filter_poses_by_index_writes_empty_output_for_empty_selection(
    tmp_path: Path,
) -> None:
    input_dir = tmp_path / "input-empty"
    _write_pose_dir(
        input_dir,
        [[(10, 20, (0, 0, 0)), (11, 21, (1, 1, 1))]],
    )
    indices_path = tmp_path / "indices-empty.npy"
    np.save(indices_path, np.array([], dtype=np.int64))

    output_dir = tmp_path / "filtered-empty"
    written = filter_poses_by_index(input_dir, indices_path, output_dir)
    loaded = read_pose_files(output_dir)

    assert len(written) == 1
    assert len(loaded) == 1
    assert loaded[0][0].shape == (0, 3)


@pytest.mark.parametrize(
    ("values", "message"),
    [
        (np.array([-1], dtype=np.int64), "non-negative"),
        (np.array([2], dtype=np.int64), "out-of-range"),
    ],
)
def test_filter_poses_by_index_rejects_invalid_indices(
    tmp_path: Path, values: np.ndarray, message: str
) -> None:
    input_dir = tmp_path / "input-invalid"
    _write_pose_dir(input_dir, [[(10, 20, (0, 0, 0)), (11, 21, (1, 1, 1))]])
    indices_path = tmp_path / "indices-invalid.npy"
    np.save(indices_path, values)

    with pytest.raises(ValueError, match=message):
        filter_poses_by_index(input_dir, indices_path, tmp_path / "filtered-invalid")


def test_filter_poses_by_index_npz_requires_key_when_archive_is_ambiguous(
    tmp_path: Path,
) -> None:
    input_dir = tmp_path / "input-npz"
    rows = _write_pose_dir(input_dir, [[(10, 20, (0, 0, 0)), (11, 21, (1, 1, 1))]])
    indices_path = tmp_path / "indices.npz"
    np.savez(indices_path, first=np.array([1], dtype=np.int64), second=np.array([0]))

    with pytest.raises(ValueError, match="indices-key"):
        filter_poses_by_index(input_dir, indices_path, tmp_path / "filtered-npz-fail")

    output_dir = tmp_path / "filtered-npz-ok"
    written = filter_poses_by_index(
        input_dir,
        indices_path,
        output_dir,
        indices_key="second",
    )

    assert len(written) == 1
    assert _load_pose_tuples(output_dir) == [rows[0]]
