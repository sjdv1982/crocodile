import sys, os

import numpy as np

sys.path.append(os.path.join(os.path.dirname(__file__), "..", "code"))
from offsets import get_discrete_offsets, expand_discrete_offsets


def test_offsets():
    np.random.seed(0)

    offsets = np.random.uniform(-5.0, 5.0, size=(10, 3))
    radii = np.concatenate(([1.0], np.random.uniform(2.0, 5.0, size=8)))

    offsets_all = np.repeat(offsets, len(radii), axis=0)
    radii_all = np.tile(radii, len(offsets))

    grid_ax = np.arange(-10, 11, dtype=int)
    gx, gy, gz = np.meshgrid(grid_ax, grid_ax, grid_ax, indexing="ij")
    grid_positions = np.column_stack([gx.ravel(), gy.ravel(), gz.ravel()])

    disp_idx, p_idx, rounded, reverse_map = get_discrete_offsets(
        offsets_all, radii_all
    )
    expanded = expand_discrete_offsets(disp_idx, p_idx, rounded, reverse_map)

    assert disp_idx.size > 0
    assert np.unique(disp_idx).size == len(offsets_all)
    assert len(expanded) == int(disp_idx.max()) + 1

    min_radius = radii.min()

    for idx, got in enumerate(expanded):
        if got is None:
            continue

        offset = offsets_all[idx]
        radius = radii_all[idx]
        dif = grid_positions - offset
        within = (dif * dif).sum(axis=1) < radius * radius
        brute = {tuple(p) for p in grid_positions[within]}

        got_set = {tuple(p) for p in got}

        if brute != got_set and radius == min_radius:
            print("brute:", sorted(brute))
            print("got:", sorted(got_set))

        assert brute == got_set
