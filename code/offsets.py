"""
Module to obtain all translations within a certain radius
"""

import numpy as np


def _generate_sphere(dim: int) -> np.ndarray:
    _mesh_ax = np.arange(-dim + 1, dim + 1, 1, dtype=int)
    _mx, _my, _mz = np.meshgrid(_mesh_ax, _mesh_ax, _mesh_ax)
    _mx2, _my2, _mz2 = (
        np.fix(_mx - 0.001).astype(int),
        np.fix(_my - 0.001).astype(int),
        np.fix(_mz - 0.001).astype(int),
    )
    _msq = (_mx2**2 + _my2**2 + _mz2**2).astype(int)
    max_radius_sq = dim**2
    return _msq[_msq < max_radius_sq]


_sphere = _generate_sphere(15)  # 15 grid units should be plenty
_rsq0 = _sphere[:, 0] ** 2 + _sphere[:, 1] ** 2 + _sphere[:, 2] ** 2
_rsq = _rsq0.ravel().argsort()
_sphere = np.column_stack(np.unravel_index(_rsq, _sphere.shape)).astype(np.uint8)


def get_discrete_offsets(
    continuous_offsets: np.ndarray, radius: float
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Obtain all discrete offset vectors within a certain radius from each continuous offset
    continuous_offsets is a [N,3] array of continuous offset vectors.

    Both continuous_offsets and radius are in grid units;
    they must have been pre-divided by the grid spacing

    Returns:
    - T: a [X, 3] uint8 array of centered discrete offsets, sorted in ascending order
    - C: a [N] uint16 array of cutoff points within T, one for each continuous offset.
    - P: a [N, 3] int32 array of positions to add to the centered discrete offsets.

    Essentially, to get the results, take:
        T[:C[n]] + P[n] for n in range(N)

    """

    positions = np.round(continuous_offsets).astype(np.int32)
    dif = continuous_offsets - positions
    displacement = np.einsum("i,i->i", dif, dif)

    remaining = radius**2 - displacement
    cutoff_points = np.searchsorted(_sphere, remaining).astype(np.uint16)
    translations = _sphere[: cutoff_points.max()]
    return translations, cutoff_points, positions


def expand_discrete_offsets(translations, cutoff_points, positions):
    """
    Expand the results of get_discrete_offsets into a list of numpy arrays

    :param translations: translations (T) obtained by get_discrete_offsets
    :param cutoff_points: cutoff points (C) obtained by get_discrete_offsets
    :param positions: positions (P) obtained by get_discrete_offsets

    Returns a list of numpy arrays, one for each cutoff point
    """
    results = []
    for c, p in zip(cutoff_points, positions):
        results.append(p + translations[:c])
    return results
