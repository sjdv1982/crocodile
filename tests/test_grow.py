import os
import sys
import tempfile
import importlib.util

import numpy as np

here = os.path.dirname(__file__)
code_dir = os.path.abspath(os.path.join(here, "..", "code"))


def _load_module(name: str, filename: str):
    module_path = os.path.join(code_dir, filename)
    spec = importlib.util.spec_from_file_location(name, module_path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


grow = _load_module("grow", "grow.py")
poses = _load_module("poses", "poses.py")

_expand_source_instances = getattr(grow, "_expand_source_instances")
_load_source_pool = getattr(grow, "_load_source_pool")
_resolve_growth_layout = getattr(grow, "_resolve_growth_layout")
_stable_pose_order = getattr(grow, "_stable_pose_order")
SourceConformerCache = getattr(grow, "SourceConformerCache")
GrowthLayout = getattr(grow, "GrowthLayout")
PoseStreamAccumulator = getattr(poses, "PoseStreamAccumulator")


def test_load_source_pool_sorts_and_groups_rows():
    conf = np.array([5, 2, 5, 2, 2, 5], dtype=np.uint16)
    rot = np.array([1, 8, 0, 8, 4, 0], dtype=np.uint16)
    trans = np.array(
        [
            [9, 0, 0],
            [1, 0, 0],
            [8, 0, 0],
            [2, 0, 0],
            [3, 0, 0],
            [7, 0, 0],
        ],
        dtype=np.int16,
    )

    with tempfile.TemporaryDirectory(prefix="grow_pool_") as tmpdir:
        writer = PoseStreamAccumulator(tmpdir, zstd=True, canonical_centers=False)
        writer.add_chunk(conf, rot, trans)
        writer.finish()
        pool = _load_source_pool(tmpdir)

    assert np.array_equal(pool.unique_conformers, np.array([2, 5], dtype=np.int64))
    assert np.array_equal(pool.conformer_starts, np.array([0, 3], dtype=np.int64))
    assert np.array_equal(pool.conformer_counts, np.array([3, 3], dtype=np.int64))

    expected_rows = np.array(
        [
            [2, 4, 3, 0, 0],
            [2, 8, 1, 0, 0],
            [2, 8, 2, 0, 0],
            [5, 0, 8, 0, 0],
            [5, 0, 7, 0, 0],
            [5, 1, 9, 0, 0],
        ],
        dtype=np.int64,
    )
    actual_rows = np.column_stack(
        (
            pool.conformers.astype(np.int64),
            pool.rotamers.astype(np.int64),
            pool.translations.astype(np.int64),
        )
    )
    assert np.array_equal(actual_rows, expected_rows)


def test_expand_source_instances_respects_group_boundaries():
    cache = SourceConformerCache(
        conformer=7,
        coords=np.zeros((2, 3), dtype=np.float32),
        centered=np.zeros((2, 3), dtype=np.float32),
        mean0=np.zeros((3,), dtype=np.float32),
        pose_trace=0.0,
        rotamer_indices=np.array([3, 4, 8], dtype=np.int64),
        rotamer_matrices=np.zeros((3, 3, 3), dtype=np.float32),
        rotamer_flat=np.zeros((3, 9), dtype=np.float32),
        mean_rotated=np.zeros((3, 3), dtype=np.float32),
        instance_translations=np.array(
            [
                [10, 0, 0],
                [11, 0, 0],
                [20, 0, 0],
                [30, 0, 0],
                [31, 0, 0],
                [32, 0, 0],
            ],
            dtype=np.int16,
        ),
        instance_starts=np.array([0, 2, 3], dtype=np.int64),
        instance_counts=np.array([2, 1, 3], dtype=np.int64),
    )

    repeat_idx, translations = _expand_source_instances(
        cache,
        np.array([2, 0], dtype=np.int64),
    )

    assert np.array_equal(repeat_idx, np.array([0, 0, 0, 1, 1], dtype=np.int64))
    assert np.array_equal(
        translations,
        np.array(
            [
                [30, 0, 0],
                [31, 0, 0],
                [32, 0, 0],
                [10, 0, 0],
                [11, 0, 0],
            ],
            dtype=np.int16,
        ),
    )


def test_resolve_growth_layout_sets_crmsd_orientation_explicitly():
    forward = _resolve_growth_layout("GU", "UU", "forward")
    backward = _resolve_growth_layout("GU", "UG", "backward")

    assert forward == GrowthLayout(
        source_mask=(False, True),
        target_mask=(True, False),
        crmsd_ab="GU",
        crmsd_bc="UU",
        source_on_rows=True,
    )
    assert backward == GrowthLayout(
        source_mask=(True, False),
        target_mask=(False, True),
        crmsd_ab="UG",
        crmsd_bc="GU",
        source_on_rows=False,
    )


def test_stable_pose_order_sorts_rotamer_then_translation():
    rotamers = np.array([4, 2, 4, 2], dtype=np.int64)
    translations = np.array(
        [
            [8, 0, 0],
            [9, 0, 0],
            [7, 0, 0],
            [1, 0, 0],
        ],
        dtype=np.int16,
    )

    order = _stable_pose_order(rotamers, translations)

    assert np.array_equal(order, np.array([3, 1, 2, 0], dtype=np.int64))
