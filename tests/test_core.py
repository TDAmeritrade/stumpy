import functools
import math
import os
from unittest.mock import patch

import naive
import numpy as np
import numpy.testing as npt
import pandas as pd
import pytest
from numba import cuda
from scipy.spatial.distance import cdist

from stumpy import config, core, stump

if cuda.is_available():

    @cuda.jit("(f8[:, :], f8[:], i8[:], i8, b1, i8[:])")
    def _gpu_searchsorted_kernel(a, v, bfs, nlevel, is_left, idx):
        # A wrapper kernel for calling device function _gpu_searchsorted_left/right.
        i = cuda.grid(1)
        if i < a.shape[0]:
            if is_left:
                idx[i] = core._gpu_searchsorted_left(a[i], v[i], bfs, nlevel)
            else:
                idx[i] = core._gpu_searchsorted_right(a[i], v[i], bfs, nlevel)


try:
    from numba.errors import NumbaPerformanceWarning
except ModuleNotFoundError:
    from numba.core.errors import NumbaPerformanceWarning

TEST_THREADS_PER_BLOCK = 10


def naive_rolling_window_dot_product(Q, T):
    window = len(Q)
    result = np.zeros(len(T) - window + 1)
    for i in range(len(result)):
        result[i] = np.dot(T[i : i + window], Q)
    return result


def naive_compute_mean_std_multidimensional(T, m):
    n = T.shape[1]
    nrows, ncols = T.shape

    cumsum_T = np.empty((nrows, ncols + 1))
    np.cumsum(T, axis=1, out=cumsum_T[:, 1:])  # store output in cumsum_T[1:]
    cumsum_T[:, 0] = 0

    cumsum_T_squared = np.empty((nrows, ncols + 1))
    np.cumsum(np.square(T), axis=1, out=cumsum_T_squared[:, 1:])
    cumsum_T_squared[:, 0] = 0

    subseq_sum_T = cumsum_T[:, m:] - cumsum_T[:, : n - m + 1]
    subseq_sum_T_squared = cumsum_T_squared[:, m:] - cumsum_T_squared[:, : n - m + 1]
    M_T = subseq_sum_T / m
    Σ_T = np.abs((subseq_sum_T_squared / m) - np.square(M_T))
    Σ_T = np.sqrt(Σ_T)

    return M_T, Σ_T


def naive_idx_to_mp(I, T, m, normalize=True, p=2.0, T_subseq_isconstant=None):
    I = I.astype(np.int64)
    T = T.copy()

    if normalize:
        if T_subseq_isconstant is None:
            T_subseq_isconstant = naive.rolling_isconstant(T, m)

    T_isfinite = np.isfinite(T)
    T_subseq_isfinite = np.all(core.rolling_window(T_isfinite, m), axis=1)

    T[~T_isfinite] = 0.0
    T_subseqs = core.rolling_window(T, m)
    nn_subseqs = T_subseqs[I]
    if normalize:
        P = naive.distance(
            naive.z_norm(T_subseqs, axis=1), naive.z_norm(nn_subseqs, axis=1), axis=1
        )
        for i, nn_i in enumerate(I):
            if T_subseq_isconstant[i] and T_subseq_isconstant[nn_i]:
                P[i] = 0
            elif T_subseq_isconstant[i] or T_subseq_isconstant[nn_i]:
                P[i] = np.sqrt(m)
            else:  # pragma: no cover
                pass
    else:
        P = naive.distance(T_subseqs, nn_subseqs, axis=1, p=p)

    P[~T_subseq_isfinite] = np.inf
    P[I < 0] = np.inf

    return P


def split(node, out):
    mid = len(node) // 2
    out.append(node[mid])
    return node[:mid], node[mid + 1 :]


def naive_bfs_indices(n, fill_value=None):
    a = np.arange(n)
    nodes = [a.tolist()]
    out = []

    while nodes:
        tmp = []
        for node in nodes:
            for n in split(node, out):
                if n:
                    tmp.append(n)
        nodes = tmp

    out = np.array(out)

    if fill_value is not None:
        remainder = out.shape[0]
        level = 0
        count = np.power(2, level)

        while remainder >= count:
            remainder -= count
            level += 1
            count = np.power(2, level)

        if remainder > 0:
            out = out[:-remainder]
            last_level = np.empty(np.power(2, level), dtype=np.int64)
            last_level[0::2] = out[-np.power(2, level - 1) :] - 1
            last_level[1::2] = out[-np.power(2, level - 1) :] + 1
            mask = np.isin(last_level, out)
            last_level[mask] = fill_value
            n = len(a)
            last_level[last_level >= n] = fill_value
            out = np.concatenate([out, last_level])

    return out


test_data = [
    (np.array([-1, 1, 2], dtype=np.float64), np.array(range(5), dtype=np.float64)),
    (
        np.array([9, 8100, -60], dtype=np.float64),
        np.array([584, -11, 23, 79, 1001], dtype=np.float64),
    ),
    (np.random.uniform(-1000, 1000, [8]), np.random.uniform(-1000, 1000, [64])),
]

n = list(range(1, 50))


def test_check_bad_dtype():
    for dtype in [np.int32, np.int64, np.float32]:
        with pytest.raises(TypeError):
            core.check_dtype(np.random.rand(10).astype(dtype))


def test_check_dtype_float64():
    assert core.check_dtype(np.random.rand(10))


def test_get_max_window_size():
    for n in range(3, 10):
        ref_max_m = (
            int(
                n
                - math.floor(
                    (n + (config.STUMPY_EXCL_ZONE_DENOM - 1))
                    // (config.STUMPY_EXCL_ZONE_DENOM + 1)
                )
            )
            - 1
        )
        cmp_max_m = core.get_max_window_size(n)
        assert ref_max_m == cmp_max_m


def test_check_window_size():
    for m in range(-1, 3):
        with pytest.raises(ValueError):
            core.check_window_size(m)


def test_check_max_window_size():
    for m in range(4, 7):
        with pytest.raises(ValueError):
            core.check_window_size(m, max_size=3)


def test_check_window_size_excl_zone():
    # To ensure warning is raised if there is at least one subsequence
    # that has no non-trivial neighbor
    T = np.random.rand(10)
    m = 7

    # For `len(T) == 10` and `m == 7`, the `excl_zone` is ceil(m / 4) = 2.
    # In this case, there are `10 - 7 + 1 = 4` subsequences of length 7,
    # starting at indices 0, 1, 2, and 3. For a subsequence that starts at
    # index 1, there are no non-trivial neighbors. So, a warning should be
    # raised.
    with pytest.warns(UserWarning):
        core.check_window_size(m, max_size=len(T), n=len(T))


@pytest.mark.parametrize("Q, T", test_data)
def test_njit_sliding_dot_product(Q, T):
    ref_mp = naive_rolling_window_dot_product(Q, T)
    comp_mp = core._sliding_dot_product(Q, T)
    npt.assert_almost_equal(ref_mp, comp_mp)


@pytest.mark.parametrize("Q, T", test_data)
def test_sliding_dot_product(Q, T):
    ref_mp = naive_rolling_window_dot_product(Q, T)
    comp_mp = core.sliding_dot_product(Q, T)
    npt.assert_almost_equal(ref_mp, comp_mp)


def test_welford_nanvar():
    T = np.random.rand(64)
    m = 10

    ref_var = np.nanvar(T)
    comp_var = core.welford_nanvar(T)
    npt.assert_almost_equal(ref_var, comp_var)

    ref_var = np.nanvar(core.rolling_window(T, m), axis=1)
    comp_var = core.welford_nanvar(T, m)
    npt.assert_almost_equal(ref_var, comp_var)


def test_welford_nanvar_catastrophic_cancellation():
    T = np.array([4.0, 7.0, 13.0, 16.0, 10.0]) + 10**8
    m = 4

    ref_var = np.nanvar(core.rolling_window(T, m), axis=1)
    comp_var = core.welford_nanvar(T, m)
    npt.assert_almost_equal(ref_var, comp_var)


def test_welford_nanvar_nan():
    T = np.random.rand(64)
    m = 10

    T[1] = np.nan
    T[10] = np.nan
    T[13:18] = np.nan

    ref_var = np.nanvar(T)
    comp_var = core.welford_nanvar(T)
    npt.assert_almost_equal(ref_var, comp_var)

    ref_var = np.nanvar(core.rolling_window(T, m), axis=1)
    comp_var = core.welford_nanvar(T, m)
    npt.assert_almost_equal(ref_var, comp_var)


def test_welford_nanstd():
    T = np.random.rand(64)
    m = 10

    ref_var = np.nanstd(T)
    comp_var = core.welford_nanstd(T)
    npt.assert_almost_equal(ref_var, comp_var)

    ref_var = np.nanstd(core.rolling_window(T, m), axis=1)
    comp_var = core.welford_nanstd(T, m)
    npt.assert_almost_equal(ref_var, comp_var)


def test_rolling_std_1d():
    a = np.random.rand(64)
    for w in range(3, 6):
        ref_std = naive.rolling_nanstd(a, w)

        # welford = False (default)
        comp_std = core.rolling_nanstd(a, w)
        npt.assert_almost_equal(ref_std, comp_std)

        # welford = True
        comp_std = core.rolling_nanstd(a, w, welford=True)
        npt.assert_almost_equal(ref_std, comp_std)


def test_rolling_std_2d():
    w = 5
    for n_rows in range(1, 4):
        a = np.random.rand(n_rows * 64).reshape(n_rows, 64)
        ref_std = naive.rolling_nanstd(a, w)

        # welford = False (default)
        comp_std = core.rolling_nanstd(a, w)
        npt.assert_almost_equal(ref_std, comp_std)

        # welford = True
        comp_std = core.rolling_nanstd(a, w, welford=True)
        npt.assert_almost_equal(ref_std, comp_std)


def test_rolling_nanmin_1d():
    T = np.random.rand(64)
    for m in range(1, 12):
        ref_min = np.nanmin(T)
        comp_min = core._rolling_nanmin_1d(T)
        npt.assert_almost_equal(ref_min, comp_min)

        ref_min = np.nanmin(T)
        comp_min = core._rolling_nanmin_1d(T)
        npt.assert_almost_equal(ref_min, comp_min)


def test_rolling_nanmin():
    T = np.random.rand(64)
    for m in range(1, 12):
        ref_min = np.nanmin(core.rolling_window(T, m), axis=1)
        comp_min = core.rolling_nanmin(T, m)
        npt.assert_almost_equal(ref_min, comp_min)

        ref_min = np.nanmin(core.rolling_window(T, m), axis=1)
        comp_min = core.rolling_nanmin(T, m)
        npt.assert_almost_equal(ref_min, comp_min)


def test_rolling_nanmax_1d():
    T = np.random.rand(64)
    for m in range(1, 12):
        ref_max = np.nanmax(T)
        comp_max = core._rolling_nanmax_1d(T)
        npt.assert_almost_equal(ref_max, comp_max)

        ref_max = np.nanmax(T)
        comp_max = core._rolling_nanmax_1d(T)
        npt.assert_almost_equal(ref_max, comp_max)


def test_rolling_nanmax():
    T = np.random.rand(64)
    for m in range(1, 12):
        ref_max = np.nanmax(core.rolling_window(T, m), axis=1)
        comp_max = core.rolling_nanmax(T, m)
        npt.assert_almost_equal(ref_max, comp_max)

        ref_max = np.nanmax(core.rolling_window(T, m), axis=1)
        comp_max = core.rolling_nanmax(T, m)
        npt.assert_almost_equal(ref_max, comp_max)


@pytest.mark.parametrize("Q, T", test_data)
def test_compute_mean_std(Q, T):
    m = Q.shape[0]

    ref_μ_Q, ref_σ_Q = naive.compute_mean_std(Q, m)
    ref_M_T, ref_Σ_T = naive.compute_mean_std(T, m)
    comp_μ_Q, comp_σ_Q = core.compute_mean_std(Q, m)
    comp_M_T, comp_Σ_T = core.compute_mean_std(T, m)

    npt.assert_almost_equal(ref_μ_Q, comp_μ_Q)
    npt.assert_almost_equal(ref_σ_Q, comp_σ_Q)
    npt.assert_almost_equal(ref_M_T, comp_M_T)
    npt.assert_almost_equal(ref_Σ_T, comp_Σ_T)


@pytest.mark.parametrize("Q, T", test_data)
def test_compute_mean_std_chunked(Q, T):
    m = Q.shape[0]

    with patch("stumpy.config.STUMPY_MEAN_STD_NUM_CHUNKS", 2):
        ref_μ_Q, ref_σ_Q = naive.compute_mean_std(Q, m)
        ref_M_T, ref_Σ_T = naive.compute_mean_std(T, m)
        comp_μ_Q, comp_σ_Q = core.compute_mean_std(Q, m)
        comp_M_T, comp_Σ_T = core.compute_mean_std(T, m)

    npt.assert_almost_equal(ref_μ_Q, comp_μ_Q)
    npt.assert_almost_equal(ref_σ_Q, comp_σ_Q)
    npt.assert_almost_equal(ref_M_T, comp_M_T)
    npt.assert_almost_equal(ref_Σ_T, comp_Σ_T)


@pytest.mark.parametrize("Q, T", test_data)
def test_compute_mean_std_chunked_many(Q, T):
    m = Q.shape[0]

    with patch("stumpy.config.STUMPY_MEAN_STD_NUM_CHUNKS", 128):
        ref_μ_Q, ref_σ_Q = naive.compute_mean_std(Q, m)
        ref_M_T, ref_Σ_T = naive.compute_mean_std(T, m)
        comp_μ_Q, comp_σ_Q = core.compute_mean_std(Q, m)
        comp_M_T, comp_Σ_T = core.compute_mean_std(T, m)

    npt.assert_almost_equal(ref_μ_Q, comp_μ_Q)
    npt.assert_almost_equal(ref_σ_Q, comp_σ_Q)
    npt.assert_almost_equal(ref_M_T, comp_M_T)
    npt.assert_almost_equal(ref_Σ_T, comp_Σ_T)


@pytest.mark.parametrize("Q, T", test_data)
def test_compute_mean_std_multidimensional(Q, T):
    m = Q.shape[0]

    Q = np.array([Q, np.random.uniform(-1000, 1000, [Q.shape[0]])])
    T = np.array([T, T, np.random.uniform(-1000, 1000, [T.shape[0]])])

    ref_μ_Q, ref_σ_Q = naive_compute_mean_std_multidimensional(Q, m)
    ref_M_T, ref_Σ_T = naive_compute_mean_std_multidimensional(T, m)
    comp_μ_Q, comp_σ_Q = core.compute_mean_std(Q, m)
    comp_M_T, comp_Σ_T = core.compute_mean_std(T, m)

    npt.assert_almost_equal(ref_μ_Q, comp_μ_Q)
    npt.assert_almost_equal(ref_σ_Q, comp_σ_Q)
    npt.assert_almost_equal(ref_M_T, comp_M_T)
    npt.assert_almost_equal(ref_Σ_T, comp_Σ_T)


@pytest.mark.parametrize("Q, T", test_data)
def test_compute_mean_std_multidimensional_chunked(Q, T):
    m = Q.shape[0]

    Q = np.array([Q, np.random.uniform(-1000, 1000, [Q.shape[0]])])
    T = np.array([T, T, np.random.uniform(-1000, 1000, [T.shape[0]])])

    with patch("stumpy.config.STUMPY_MEAN_STD_NUM_CHUNKS", 2):
        ref_μ_Q, ref_σ_Q = naive_compute_mean_std_multidimensional(Q, m)
        ref_M_T, ref_Σ_T = naive_compute_mean_std_multidimensional(T, m)
        comp_μ_Q, comp_σ_Q = core.compute_mean_std(Q, m)
        comp_M_T, comp_Σ_T = core.compute_mean_std(T, m)

    npt.assert_almost_equal(ref_μ_Q, comp_μ_Q)
    npt.assert_almost_equal(ref_σ_Q, comp_σ_Q)
    npt.assert_almost_equal(ref_M_T, comp_M_T)
    npt.assert_almost_equal(ref_Σ_T, comp_Σ_T)


@pytest.mark.parametrize("Q, T", test_data)
def test_compute_mean_std_multidimensional_chunked_many(Q, T):
    m = Q.shape[0]

    Q = np.array([Q, np.random.uniform(-1000, 1000, [Q.shape[0]])])
    T = np.array([T, T, np.random.uniform(-1000, 1000, [T.shape[0]])])

    with patch("stumpy.config.STUMPY_MEAN_STD_NUM_CHUNKS", 128):
        ref_μ_Q, ref_σ_Q = naive_compute_mean_std_multidimensional(Q, m)
        ref_M_T, ref_Σ_T = naive_compute_mean_std_multidimensional(T, m)
        comp_μ_Q, comp_σ_Q = core.compute_mean_std(Q, m)
        comp_M_T, comp_Σ_T = core.compute_mean_std(T, m)

    npt.assert_almost_equal(ref_μ_Q, comp_μ_Q)
    npt.assert_almost_equal(ref_σ_Q, comp_σ_Q)
    npt.assert_almost_equal(ref_M_T, comp_M_T)
    npt.assert_almost_equal(ref_Σ_T, comp_Σ_T)


@pytest.mark.parametrize("Q, T", test_data)
def test_calculate_squared_distance_profile(Q, T):
    m = Q.shape[0]
    ref = (
        np.linalg.norm(
            core.z_norm(core.rolling_window(T, m), 1) - core.z_norm(Q), axis=1
        )
        ** 2
    )

    QT = core.sliding_dot_product(Q, T)
    Q_subseq_isconstant = core.rolling_isconstant(Q, m)[0]
    μ_Q, σ_Q = [arr[0] for arr in core.compute_mean_std(Q, m)]

    T_subseq_isconstant = core.rolling_isconstant(T, m)
    M_T, Σ_T = core.compute_mean_std(T, m)

    comp = core._calculate_squared_distance_profile(
        m,
        QT,
        μ_Q,
        σ_Q,
        M_T,
        Σ_T,
        Q_subseq_isconstant,
        T_subseq_isconstant,
    )
    npt.assert_almost_equal(ref, comp)


@pytest.mark.parametrize("Q, T", test_data)
def test_calculate_distance_profile(Q, T):
    m = Q.shape[0]
    ref = np.linalg.norm(
        core.z_norm(core.rolling_window(T, m), 1) - core.z_norm(Q), axis=1
    )

    QT = core.sliding_dot_product(Q, T)
    Q_subseq_isconstant = core.rolling_isconstant(Q, m)[0]
    μ_Q, σ_Q = [arr[0] for arr in core.compute_mean_std(Q, m)]

    T_subseq_isconstant = core.rolling_isconstant(T, m)
    M_T, Σ_T = core.compute_mean_std(T, m)

    comp = core.calculate_distance_profile(
        m,
        QT,
        μ_Q,
        σ_Q,
        M_T,
        Σ_T,
        Q_subseq_isconstant,
        T_subseq_isconstant,
    )
    npt.assert_almost_equal(ref, comp)


@pytest.mark.parametrize("Q, T", test_data)
def test_mueen_calculate_distance_profile(Q, T):
    m = Q.shape[0]
    ref = np.linalg.norm(
        core.z_norm(core.rolling_window(T, m), 1) - core.z_norm(Q), axis=1
    )
    comp = core.mueen_calculate_distance_profile(Q, T)
    npt.assert_almost_equal(ref, comp)


@pytest.mark.parametrize("Q, T", test_data)
def test_mass(Q, T):
    Q = Q.copy()
    T = T.copy()
    m = Q.shape[0]
    ref = np.linalg.norm(
        core.z_norm(core.rolling_window(T, m), 1) - core.z_norm(Q), axis=1
    )
    comp = core.mass(Q, T)
    npt.assert_almost_equal(ref, comp)


@pytest.mark.parametrize("Q, T", test_data)
def test_mass_Q_nan(Q, T):
    Q = Q.copy()
    Q[1] = np.nan
    T = T.copy()
    m = Q.shape[0]

    ref = np.linalg.norm(
        core.z_norm(core.rolling_window(T, m), 1) - core.z_norm(Q), axis=1
    )
    ref[np.isnan(ref)] = np.inf

    comp = core.mass(Q, T)
    npt.assert_almost_equal(ref, comp)


@pytest.mark.parametrize("Q, T", test_data)
def test_mass_Q_inf(Q, T):
    Q = Q.copy()
    Q[1] = np.inf
    T = T.copy()
    m = Q.shape[0]

    ref = np.linalg.norm(
        core.z_norm(core.rolling_window(T, m), 1) - core.z_norm(Q), axis=1
    )
    ref[np.isnan(ref)] = np.inf

    comp = core.mass(Q, T)
    npt.assert_almost_equal(ref, comp)
    T[1] = 1e10


@pytest.mark.parametrize("Q, T", test_data)
def test_mass_T_nan(Q, T):
    Q = Q.copy()
    T = T.copy()
    T[1] = np.nan
    m = Q.shape[0]

    ref = np.linalg.norm(
        core.z_norm(core.rolling_window(T, m), 1) - core.z_norm(Q), axis=1
    )
    ref[np.isnan(ref)] = np.inf

    comp = core.mass(Q, T)
    npt.assert_almost_equal(ref, comp)


@pytest.mark.parametrize("Q, T", test_data)
def test_mass_T_inf(Q, T):
    Q = Q.copy()
    T = T.copy()
    T[1] = np.inf
    m = Q.shape[0]

    ref = np.linalg.norm(
        core.z_norm(core.rolling_window(T, m), 1) - core.z_norm(Q), axis=1
    )
    ref[np.isnan(ref)] = np.inf

    comp = core.mass(Q, T)
    npt.assert_almost_equal(ref, comp)
    T[1] = 1e10


@pytest.mark.parametrize("Q, T", test_data)
def test_p_norm_distance_profile(Q, T):
    Q = Q.copy()
    T = T.copy()
    m = Q.shape[0]
    for p in [1.0, 1.5, 2.0]:
        ref = cdist(
            core.rolling_window(Q, m),
            core.rolling_window(T, m),
            metric="minkowski",
            p=p,
        ).flatten()
        ref = np.power(ref, p)
        cmp = core._p_norm_distance_profile(Q, T, p)
        npt.assert_almost_equal(ref, cmp)


@pytest.mark.parametrize("Q, T", test_data)
def test_mass_absolute(Q, T):
    Q = Q.copy()
    T = T.copy()
    m = Q.shape[0]
    for p in [1.0, 2.0, 3.0]:
        ref = np.linalg.norm(core.rolling_window(T, m) - Q, axis=1, ord=p)
        comp = core.mass_absolute(Q, T, p=p)
        npt.assert_almost_equal(ref, comp)


@pytest.mark.parametrize("Q, T", test_data)
def test_mass_absolute_Q_nan(Q, T):
    Q = Q.copy()
    Q[1] = np.nan
    T = T.copy()
    m = Q.shape[0]

    ref = np.linalg.norm(core.rolling_window(T, m) - Q, axis=1)
    ref[np.isnan(ref)] = np.inf

    comp = core.mass_absolute(Q, T)
    npt.assert_almost_equal(ref, comp)


@pytest.mark.parametrize("Q, T", test_data)
def test_mass_absolute_Q_inf(Q, T):
    Q = Q.copy()
    Q[1] = np.inf
    T = T.copy()
    m = Q.shape[0]

    ref = np.linalg.norm(core.rolling_window(T, m) - Q, axis=1)
    ref[np.isnan(ref)] = np.inf

    comp = core.mass_absolute(Q, T)
    npt.assert_almost_equal(ref, comp)


@pytest.mark.parametrize("Q, T", test_data)
def test_mass_absolute_T_nan(Q, T):
    Q = Q.copy()
    T = T.copy()
    T[1] = np.nan
    m = Q.shape[0]

    ref = np.linalg.norm(core.rolling_window(T, m) - Q, axis=1)
    ref[np.isnan(ref)] = np.inf

    comp = core.mass_absolute(Q, T)
    npt.assert_almost_equal(ref, comp)


@pytest.mark.parametrize("Q, T", test_data)
def test_mass_absolute_T_inf(Q, T):
    Q = Q.copy()
    T = T.copy()
    T[1] = np.inf
    m = Q.shape[0]

    ref = np.linalg.norm(core.rolling_window(T, m) - Q, axis=1)
    ref[np.isnan(ref)] = np.inf

    comp = core.mass_absolute(Q, T)
    npt.assert_almost_equal(ref, comp)


def test_mass_absolute_sqrt_input_negative():
    Q = np.array(
        [
            -13.09,
            -14.1,
            -15.08,
            -16.31,
            -17.13,
            -17.5,
            -18.07,
            -18.07,
            -17.48,
            -16.24,
            -14.88,
            -13.56,
            -12.65,
            -11.93,
            -11.48,
            -11.06,
            -10.83,
            -10.67,
            -10.59,
            -10.81,
            -10.92,
            -11.15,
            -11.37,
            -11.53,
            -11.19,
            -11.08,
            -10.48,
            -10.14,
            -9.92,
            -9.99,
            -10.11,
            -9.92,
            -9.7,
            -9.47,
            -9.06,
            -9.01,
            -8.79,
            -8.67,
            -8.33,
            -8.0,
            -8.26,
            -8.0,
            -7.54,
            -7.32,
            -7.13,
            -7.24,
            -7.43,
            -7.93,
            -8.8,
            -9.71,
        ]
    )
    ref = 0.0
    comp = core.mass_absolute(Q, Q)
    npt.assert_almost_equal(ref, comp)


@pytest.mark.parametrize("T_A, T_B", test_data)
def test_mass_distance_matrix(T_A, T_B):
    m = 3

    ref_distance_matrix = naive.distance_matrix(T_A, T_B, m)
    k = T_A.shape[0] - m + 1
    l = T_B.shape[0] - m + 1
    comp_distance_matrix = np.full((k, l), np.inf)
    core.mass_distance_matrix(T_A, T_B, m, comp_distance_matrix)

    npt.assert_almost_equal(ref_distance_matrix, comp_distance_matrix)


@pytest.mark.parametrize("T_A, T_B", test_data)
def test_mass_absolute_distance_matrix(T_A, T_B):
    m = 3

    ref_distance_matrix = cdist(
        core.rolling_window(T_A, m), core.rolling_window(T_B, m)
    )
    k = T_A.shape[0] - m + 1
    l = T_B.shape[0] - m + 1
    comp_distance_matrix = np.full((k, l), np.inf)
    core._mass_absolute_distance_matrix(T_A, T_B, m, comp_distance_matrix)

    npt.assert_almost_equal(ref_distance_matrix, comp_distance_matrix)


def test_apply_exclusion_zone():
    T = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9], dtype=np.float64)
    ref = np.empty(T.shape, dtype=np.float64)
    comp = np.empty(T.shape, dtype=np.float64)
    exclusion_zone = 2

    for i in range(T.shape[0]):
        ref[:] = T[:]
        naive.apply_exclusion_zone(ref, i, exclusion_zone, np.inf)

        comp[:] = T[:]
        core.apply_exclusion_zone(comp, i, exclusion_zone, np.inf)

        naive.replace_inf(ref)
        naive.replace_inf(comp)
        npt.assert_array_equal(ref, comp)


def test_apply_exclusion_zone_int():
    T = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9], dtype=np.int64)
    ref = np.empty(T.shape, dtype=np.int64)
    comp = np.empty(T.shape, dtype=np.int64)
    exclusion_zone = 2

    for i in range(T.shape[0]):
        ref[:] = T[:]
        naive.apply_exclusion_zone(ref, i, exclusion_zone, -1)

        comp[:] = T[:]
        core.apply_exclusion_zone(comp, i, exclusion_zone, -1)

        naive.replace_inf(ref)
        naive.replace_inf(comp)
        npt.assert_array_equal(ref, comp)


def test_apply_exclusion_zone_bool():
    T = np.ones(10, dtype=bool)
    ref = np.empty(T.shape, dtype=bool)
    comp = np.empty(T.shape, dtype=bool)
    exclusion_zone = 2

    for i in range(T.shape[0]):
        ref[:] = T[:]
        naive.apply_exclusion_zone(ref, i, exclusion_zone, False)

        comp[:] = T[:]
        core.apply_exclusion_zone(comp, i, exclusion_zone, False)

        naive.replace_inf(ref)
        naive.replace_inf(comp)
        npt.assert_array_equal(ref, comp)


def test_apply_exclusion_zone_multidimensional():
    T = np.array(
        [[0, 1, 2, 3, 4, 5, 6, 7, 8, 9], [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]],
        dtype=np.float64,
    )
    ref = np.empty(T.shape, dtype=np.float64)
    comp = np.empty(T.shape, dtype=np.float64)
    exclusion_zone = 2

    for i in range(T.shape[1]):
        ref[:, :] = T[:, :]
        naive.apply_exclusion_zone(ref, i, exclusion_zone, np.inf)

        comp[:, :] = T[:, :]
        core.apply_exclusion_zone(comp, i, exclusion_zone, np.inf)

        naive.replace_inf(ref)
        naive.replace_inf(comp)
        npt.assert_array_equal(ref, comp)


def test_preprocess():
    T = np.array([0, np.nan, 2, 3, 4, 5, 6, 7, np.inf, 9])
    m = 3

    ref_T = np.array([0, 0, 2, 3, 4, 5, 6, 7, 0, 9], dtype=float)
    ref_subseq_isconstant = naive.rolling_isconstant(T, m)
    ref_M, ref_Σ = naive.compute_mean_std(T, m)

    comp_T, comp_M, comp_Σ, comp_subseq_isconstant = core.preprocess(T, m)

    npt.assert_almost_equal(ref_T, comp_T)
    npt.assert_almost_equal(ref_M, comp_M)
    npt.assert_almost_equal(ref_Σ, comp_Σ)
    npt.assert_almost_equal(ref_subseq_isconstant, comp_subseq_isconstant)

    T = pd.Series(T)
    comp_T, comp_M, comp_Σ, comp_subseq_isconstant = core.preprocess(T, m)

    npt.assert_almost_equal(ref_T, comp_T)
    npt.assert_almost_equal(ref_M, comp_M)
    npt.assert_almost_equal(ref_Σ, comp_Σ)
    npt.assert_almost_equal(ref_subseq_isconstant, comp_subseq_isconstant)


def test_preprocess_non_normalized():
    T = np.array([0, np.nan, 2, 3, 4, 5, 6, 7, np.inf, 9])
    m = 3

    ref_T_subseq_isfinite = np.full(T.shape[0] - m + 1, False, dtype=bool)
    for i in range(T.shape[0] - m + 1):
        if np.all(np.isfinite(T[i : i + m])):
            ref_T_subseq_isfinite[i] = True

    ref_T = np.array([0, 0, 2, 3, 4, 5, 6, 7, 0, 9], dtype=float)

    comp_T, comp_T_subseq_isfinite = core.preprocess_non_normalized(T, m)

    npt.assert_almost_equal(ref_T, comp_T)
    npt.assert_almost_equal(ref_T_subseq_isfinite, comp_T_subseq_isfinite)

    T = pd.Series(T)
    comp_T, comp_T_subseq_isfinite = core.preprocess_non_normalized(T, m)

    npt.assert_almost_equal(ref_T, comp_T)
    npt.assert_almost_equal(ref_T_subseq_isfinite, comp_T_subseq_isfinite)


def test_preprocess_diagonal():
    T = np.array([0, np.nan, 2, 3, 4, 5, 6, 7, np.inf, 9])
    m = 3

    ref_T = np.array([0, 0, 2, 3, 4, 5, 6, 7, 0, 9], dtype=float)
    ref_M, ref_Σ = naive.compute_mean_std(ref_T, m)
    ref_Σ_inverse = 1.0 / ref_Σ
    ref_M_m_1, _ = naive.compute_mean_std(ref_T, m - 1)

    (
        comp_T,
        comp_M,
        comp_Σ_inverse,
        comp_M_m_1,
        comp_T_subseq_isfinite,
        comp_T_subseq_isconstant,
    ) = core.preprocess_diagonal(T, m)

    npt.assert_almost_equal(ref_T, comp_T)
    npt.assert_almost_equal(ref_M, comp_M)
    npt.assert_almost_equal(ref_Σ_inverse, comp_Σ_inverse)
    npt.assert_almost_equal(ref_M_m_1, comp_M_m_1)

    T = pd.Series(T)
    (
        comp_T,
        comp_M,
        comp_Σ_inverse,
        comp_M_m_1,
        comp_T_subseq_isfinite,
        comp_T_subseq_isconstant,
    ) = core.preprocess_diagonal(T, m)

    npt.assert_almost_equal(ref_T, comp_T)
    npt.assert_almost_equal(ref_M, comp_M)
    npt.assert_almost_equal(ref_Σ_inverse, comp_Σ_inverse)
    npt.assert_almost_equal(ref_M_m_1, comp_M_m_1)


def test_replace_distance():
    right = np.random.rand(30).reshape(5, 6)
    left = right.copy()
    np.fill_diagonal(right, config.STUMPY_MAX_DISTANCE - 1e-9)
    np.fill_diagonal(left, np.inf)
    core.replace_distance(right, config.STUMPY_MAX_DISTANCE, np.inf, 1e-6)


def test_array_to_temp_file():
    left = np.random.rand()
    fname = core.array_to_temp_file(left)
    right = np.load(fname, allow_pickle=False)
    os.remove(fname)

    npt.assert_almost_equal(left, right)


def test_count_diagonal_ndist():
    for n_A in range(10, 15):
        for n_B in range(10, 15):
            for m in range(3, 6):
                diags = np.random.permutation(
                    range(-(n_A - m + 1) + 1, n_B - m + 1)
                ).astype(np.int64)
                ones_matrix = np.ones((n_A - m + 1, n_B - m + 1), dtype=np.int64)
                ref_ndist_counts = np.empty(len(diags))
                for i, diag in enumerate(diags):
                    ref_ndist_counts[i] = ones_matrix.diagonal(offset=diag).sum()

                comp_ndist_counts = core._count_diagonal_ndist(diags, m, n_A, n_B)

                npt.assert_almost_equal(ref_ndist_counts, comp_ndist_counts)


def test_get_array_ranges():
    x = np.array([3, 9, 2, 1, 5, 4, 7, 7, 8, 6], dtype=np.int64)
    for n_chunks in range(2, 5):
        ref = naive.get_array_ranges(x, n_chunks, False)

        cmp = core._get_array_ranges(x, n_chunks, False)
        npt.assert_almost_equal(ref, cmp)


def test_get_array_ranges_exhausted():
    x = np.array([3, 3, 3, 11, 11, 11], dtype=np.int64)
    n_chunks = 6

    ref = naive.get_array_ranges(x, n_chunks, False)

    cmp = core._get_array_ranges(x, n_chunks, False)
    npt.assert_almost_equal(ref, cmp)


def test_get_array_ranges_exhausted_truncated():
    x = np.array([3, 3, 3, 11, 11, 11], dtype=np.int64)
    n_chunks = 6

    ref = naive.get_array_ranges(x, n_chunks, True)

    cmp = core._get_array_ranges(x, n_chunks, True)
    npt.assert_almost_equal(ref, cmp)


def test_get_array_ranges_empty_array():
    x = np.array([], dtype=np.int64)
    n_chunks = 6

    ref = naive.get_array_ranges(x, n_chunks, False)

    cmp = core._get_array_ranges(x, n_chunks, False)
    npt.assert_almost_equal(ref, cmp)


def test_get_ranges():
    ref = np.array([[0, 3], [3, 6]])
    size = 6
    n_chunks = 2
    cmp = core._get_ranges(size, n_chunks, False)
    npt.assert_almost_equal(ref, cmp)


def test_get_ranges_exhausted():
    ref = np.array([[0, 1], [1, 2], [2, 3], [3, 3], [3, 4], [4, 5], [5, 6], [6, 6]])
    size = 6
    n_chunks = 8
    cmp = core._get_ranges(size, n_chunks, False)
    npt.assert_almost_equal(ref, cmp)


def test_get_ranges_exhausted_truncated():
    ref = np.array([[0, 1], [1, 2], [2, 3], [3, 4], [4, 5], [5, 6]])
    size = 6
    n_chunks = 8
    cmp = core._get_ranges(size, n_chunks, True)
    npt.assert_almost_equal(ref, cmp)


def test_get_ranges_zero_size():
    ref = np.empty((0, 2))
    size = 0
    n_chunks = 8
    cmp = core._get_ranges(size, n_chunks, True)
    npt.assert_almost_equal(ref, cmp)


def test_rolling_isfinite():
    a = np.arange(12).astype(np.float64)
    w = 3

    a[1] = np.nan
    a[5] = np.nan
    a[9] = np.nan

    ref = np.all(core.rolling_window(np.isfinite(a), w), axis=1)
    comp = core.rolling_isfinite(a, w)

    npt.assert_almost_equal(ref, comp)

    # test `a` as all boolean isfinite array
    comp = core.rolling_isfinite(np.isfinite(a), w)
    npt.assert_almost_equal(ref, comp)


def test_rolling_isconstant():
    a = np.arange(12).astype(np.float64)
    w = 3

    a[:3] = 77.0
    a[1] = np.inf
    a[4:7] = 77.0
    a[9:12] = [77.0, np.nan, 77.0]

    ref = naive.rolling_isconstant(a, w)
    comp = core.rolling_isconstant(a, w)

    npt.assert_almost_equal(ref, comp)


def test_compare_parameters():
    assert (
        core._compare_parameters(core.rolling_window, core.z_norm, exclude=[]) is False
    )


def test_jagged_list_to_array():
    arr = [np.array([0, 1]), np.array([0]), np.array([0, 1, 2, 3])]

    left = np.array([[0, 1, -1, -1], [0, -1, -1, -1], [0, 1, 2, 3]], dtype="int64")
    right = core._jagged_list_to_array(arr, fill_value=-1, dtype="int64")
    npt.assert_array_equal(left, right)

    left = np.array(
        [[0, 1, np.nan, np.nan], [0, np.nan, np.nan, np.nan], [0, 1, 2, 3]],
        dtype="float64",
    )
    right = core._jagged_list_to_array(arr, fill_value=np.nan, dtype="float64")
    npt.assert_array_equal(left, right)


def test_jagged_list_to_array_empty():
    arr = []

    left = np.array([[]], dtype="int64")
    right = core._jagged_list_to_array(arr, fill_value=-1, dtype="int64")
    npt.assert_array_equal(left, right)

    left = np.array([[]], dtype="float64")
    right = core._jagged_list_to_array(arr, fill_value=np.nan, dtype="float64")
    npt.assert_array_equal(left, right)


def test_get_mask_slices():
    bool_list = [False, True]
    mask_cases = [
        [x, y, z, w]
        for x in bool_list
        for y in bool_list
        for z in bool_list
        for w in bool_list
    ]

    for mask in mask_cases:
        ref_slices = naive._get_mask_slices(mask)
        comp_slices = core._get_mask_slices(mask)
        npt.assert_array_equal(ref_slices, comp_slices)


def test_idx_to_mp():
    n = 64
    m = 5
    T = np.random.rand(n)
    # T[1] = np.nan
    # T[8] = np.inf
    # T[:] = 1.0
    l = n - m + 1
    I = np.random.randint(0, l, l)

    # `normalize == True` and `T_subseq_isconstant` is None (default)
    ref_mp = naive_idx_to_mp(I, T, m)
    cmp_mp = core._idx_to_mp(I, T, m)
    npt.assert_almost_equal(ref_mp, cmp_mp)

    # `normalize == True` and `T_subseq_isconstant` is provided
    T_subseq_isconstant = np.random.choice([True, False], l, replace=True)
    ref_mp = naive_idx_to_mp(I, T, m, T_subseq_isconstant=T_subseq_isconstant)
    cmp_mp = core._idx_to_mp(I, T, m, T_subseq_isconstant=T_subseq_isconstant)
    npt.assert_almost_equal(ref_mp, cmp_mp)

    # `normalize == False`
    for p in range(1, 4):
        ref_mp = naive_idx_to_mp(I, T, m, normalize=False, p=p)
        cmp_mp = core._idx_to_mp(I, T, m, normalize=False, p=p)
        npt.assert_almost_equal(ref_mp, cmp_mp)


def test_total_diagonal_ndists():
    tile_height = 9
    tile_width = 11
    for tile_lower_diag in range(-tile_height - 2, tile_width + 2):
        for tile_upper_diag in range(tile_lower_diag, tile_width + 2):
            assert naive._total_diagonal_ndists(
                tile_lower_diag, tile_upper_diag, tile_height, tile_width
            ) == core._total_diagonal_ndists(
                tile_lower_diag, tile_upper_diag, tile_height, tile_width
            )

    tile_height = 11
    tile_width = 9
    for tile_lower_diag in range(-tile_height - 2, tile_width + 2):
        for tile_upper_diag in range(tile_lower_diag, tile_width + 2):
            assert naive._total_diagonal_ndists(
                tile_lower_diag, tile_upper_diag, tile_height, tile_width
            ) == core._total_diagonal_ndists(
                tile_lower_diag, tile_upper_diag, tile_height, tile_width
            )


@pytest.mark.parametrize("n", n)
def test_bfs_indices(n):
    ref_bfs_indices = naive_bfs_indices(n)
    cmp_bfs_indices = np.array(list(core._bfs_indices(n)))

    npt.assert_almost_equal(ref_bfs_indices, cmp_bfs_indices)


@pytest.mark.parametrize("n", n)
def test_bfs_indices_fill_value(n):
    ref_bfs_indices = naive_bfs_indices(n, -1)
    cmp_bfs_indices = np.array(list(core._bfs_indices(n, -1)))

    npt.assert_almost_equal(ref_bfs_indices, cmp_bfs_indices)


def test_select_P_ABBA_val_inf():
    P_ABBA = np.random.rand(10)
    k = 2
    P_ABBA[k:] = np.inf
    p_abba = P_ABBA.copy()

    comp = core._select_P_ABBA_value(P_ABBA, k=k)
    p_abba.sort()
    ref = p_abba[k - 1]
    npt.assert_almost_equal(ref, comp)


def test_merge_topk_PI_without_overlap():
    # This is to test function `core._merge_topk_PI(PA, PB, IA, IB)` when there
    # is no overlap between row IA[i] and row IB[i].
    n = 50
    for k in range(1, 6):
        PA = np.random.rand(n * k).reshape(n, k)
        PA[:, :] = np.sort(PA, axis=1)  # sorting each row separately

        PB = np.random.rand(n * k).reshape(n, k)
        col_idx = np.random.randint(0, k, size=n)
        for i in range(n):  # creating ties between values of PA and PB
            PB[i, col_idx[i]] = np.random.choice(PA[i], size=1, replace=False)
        PB[:, :] = np.sort(PB, axis=1)  # sorting each row separately

        IA = np.arange(n * k).reshape(n, k)
        IB = IA + n * k

        ref_P = PA.copy()
        ref_I = IA.copy()

        comp_P = PA.copy()
        comp_I = IA.copy()

        naive.merge_topk_PI(ref_P, PB.copy(), ref_I, IB.copy())
        core._merge_topk_PI(comp_P, PB.copy(), comp_I, IB.copy())

        npt.assert_almost_equal(ref_P, comp_P)
        npt.assert_almost_equal(ref_I, comp_I)


def test_merge_topk_PI_with_overlap():
    # This is to test function `core._merge_topk_PI(PA, PB, IA, IB)` when there
    # is overlap between row IA[i] and row IB[i].
    n = 50
    for k in range(1, 6):
        # note: we do not have overlap issue when k is 1. The `k=1` is considered
        # for the sake of consistency with the `without-overlap` test function.
        PA = np.random.rand(n * k).reshape(n, k)
        PB = np.random.rand(n * k).reshape(n, k)

        IA = np.arange(n * k).reshape(n, k)
        IB = IA + n * k

        num_overlaps = np.random.randint(1, k + 1, size=n)
        for i in range(n):
            # create overlaps
            col_IDX = np.random.choice(np.arange(k), num_overlaps[i], replace=False)
            imprecision = np.random.uniform(low=-1e-06, high=1e-06, size=len(col_IDX))
            PB[i, col_IDX] = PA[i, col_IDX] + imprecision
            IB[i, col_IDX] = IA[i, col_IDX]

        # sort each row of PA/PB (and update  IA/IB accordingly)
        IDX = np.argsort(PA, axis=1)
        PA[:, :] = np.take_along_axis(PA, IDX, axis=1)
        IA[:, :] = np.take_along_axis(IA, IDX, axis=1)

        IDX = np.argsort(PB, axis=1)
        PB[:, :] = np.take_along_axis(PB, IDX, axis=1)
        IB[:, :] = np.take_along_axis(IB, IDX, axis=1)

        ref_P = PA.copy()
        ref_I = IA.copy()

        comp_P = PA.copy()
        comp_I = IA.copy()

        naive.merge_topk_PI(ref_P, PB.copy(), ref_I, IB.copy())
        core._merge_topk_PI(comp_P, PB.copy(), comp_I, IB.copy())

        npt.assert_almost_equal(ref_P, comp_P)
        npt.assert_almost_equal(ref_I, comp_I)


def test_merge_topk_PI_with_1D_input():
    # including some overlaps randomly
    n = 50
    PA = np.random.rand(n)
    PB = np.random.rand(n)

    IA = np.arange(n)
    IB = IA + n

    n_overlaps = np.random.randint(1, n + 1)
    IDX_rows_with_overlaps = np.random.choice(np.arange(n), n_overlaps, replace=False)
    imprecision = np.random.uniform(low=-1e-06, high=1e-06, size=n_overlaps)
    PB[IDX_rows_with_overlaps] = PA[IDX_rows_with_overlaps] + imprecision
    IB[IDX_rows_with_overlaps] = IA[IDX_rows_with_overlaps]

    ref_P = PA.copy()
    ref_I = IA.copy()
    comp_P = PA.copy()
    comp_I = IA.copy()

    naive.merge_topk_PI(ref_P, PB.copy(), ref_I, IB.copy())
    core._merge_topk_PI(comp_P, PB.copy(), comp_I, IB.copy())

    npt.assert_almost_equal(ref_P, comp_P)
    npt.assert_almost_equal(ref_I, comp_I)


def test_merge_topk_PI_with_1D_input_hardcoded():
    # It is possible that the generated arrays in the test function
    # `test_merge_topk_PI_with_1D_input` does not trigger the if-block
    # `merge_topk_PI` in 1D case. This test function ensure that the if-block
    # will be executed.
    PA = np.array([0.1, 0.3, 0.5, 0.7, 0.9])
    PB = np.array([0.2, 0.3, 0.6, 0.8, 1.0])

    IA = np.array([0, 1, 2, 3, 4])
    IB = np.array([10, 1, 12, 13, 14])

    ref_P = PA.copy()
    ref_I = IA.copy()

    comp_P = PA.copy()
    comp_I = IA.copy()

    naive.merge_topk_PI(ref_P, PB.copy(), ref_I, IB.copy())
    core._merge_topk_PI(comp_P, PB.copy(), comp_I, IB.copy())

    npt.assert_almost_equal(ref_P, comp_P)
    npt.assert_almost_equal(ref_I, comp_I)


def test_merge_topk_ρI_without_overlap():
    # This is to test function `core._merge_topk_ρI(ρA, ρB, IA, IB)` when there
    # is no overlap between row IA[i] and row IB[i].
    n = 50
    for k in range(1, 6):
        ρA = np.random.rand(n * k).reshape(n, k)
        ρA[:, :] = np.sort(ρA, axis=1)  # sorting each row separately

        ρB = np.random.rand(n * k).reshape(n, k)
        col_idx = np.random.randint(0, k, size=n)
        for i in range(n):  # creating ties between values of PA and PB
            ρB[i, col_idx[i]] = np.random.choice(ρA[i], size=1, replace=False)
        ρB[:, :] = np.sort(ρB, axis=1)  # sorting each row separately

        IA = np.arange(n * k).reshape(n, k)
        IB = IA + n * k

        ref_ρ = ρA.copy()
        ref_I = IA.copy()

        comp_ρ = ρA.copy()
        comp_I = IA.copy()

        naive.merge_topk_ρI(ref_ρ, ρB.copy(), ref_I, IB.copy())
        core._merge_topk_ρI(comp_ρ, ρB.copy(), comp_I, IB.copy())

        npt.assert_almost_equal(ref_ρ, comp_ρ)
        npt.assert_almost_equal(ref_I, comp_I)


def test_merge_topk_ρI_with_overlap():
    # This is to test function `core._merge_topk_ρI(ρA, ρB, IA, IB)` when there
    # is overlap between row IA[i] and row IB[i].
    n = 50
    for k in range(1, 6):
        # note: we do not have overlap issue when k is 1. The `k=1` is considered
        # for the sake of consistency with the `without-overlap` test function.
        ρA = np.random.rand(n * k).reshape(n, k)
        ρB = np.random.rand(n * k).reshape(n, k)

        IA = np.arange(n * k).reshape(n, k)
        IB = IA + n * k

        num_overlaps = np.random.randint(1, k + 1, size=n)
        for i in range(n):
            # create overlaps
            col_IDX = np.random.choice(np.arange(k), num_overlaps[i], replace=False)
            imprecision = np.random.uniform(low=-1e-06, high=1e-06, size=len(col_IDX))
            ρB[i, col_IDX] = ρA[i, col_IDX] + imprecision
            IB[i, col_IDX] = IA[i, col_IDX]

        # sort each row of ρA/ρB (and update IA/IB accordingly)
        IDX = np.argsort(ρA, axis=1)
        ρA[:, :] = np.take_along_axis(ρA, IDX, axis=1)
        IA[:, :] = np.take_along_axis(IA, IDX, axis=1)

        IDX = np.argsort(ρB, axis=1)
        ρB[:, :] = np.take_along_axis(ρB, IDX, axis=1)
        IB[:, :] = np.take_along_axis(IB, IDX, axis=1)

        ref_ρ = ρA.copy()
        ref_I = IA.copy()

        comp_ρ = ρA.copy()
        comp_I = IA.copy()

        naive.merge_topk_ρI(ref_ρ, ρB.copy(), ref_I, IB.copy())
        core._merge_topk_ρI(comp_ρ, ρB.copy(), comp_I, IB.copy())

        npt.assert_almost_equal(ref_ρ, comp_ρ)
        npt.assert_almost_equal(ref_I, comp_I)


def test_merge_topk_ρI_with_1D_input():
    # including some overlaps randomly
    n = 50
    ρA = np.random.rand(n)
    ρB = np.random.rand(n)

    IA = np.arange(n)
    IB = IA + n

    n_overlaps = np.random.randint(1, n + 1)
    IDX_rows_with_overlaps = np.random.choice(np.arange(n), n_overlaps, replace=False)
    imprecision = np.random.uniform(low=-1e-06, high=1e-06, size=n_overlaps)
    ρB[IDX_rows_with_overlaps] = ρA[IDX_rows_with_overlaps] + imprecision
    IB[IDX_rows_with_overlaps] = IA[IDX_rows_with_overlaps]

    ref_ρ = ρA.copy()
    ref_I = IA.copy()
    comp_ρ = ρA.copy()
    comp_I = IA.copy()

    naive.merge_topk_ρI(ref_ρ, ρB.copy(), ref_I, IB.copy())
    core._merge_topk_ρI(comp_ρ, ρB.copy(), comp_I, IB.copy())

    npt.assert_almost_equal(ref_ρ, comp_ρ)
    npt.assert_almost_equal(ref_I, comp_I)


def test_merge_topk_ρI_with_1D_input_hardcoded():
    # It is possible that the generated arrays in the test function
    # `test_merge_topk_ρI_with_1D_input` does not trigger the if-block
    # `merge_topk_ρI` in 1D case. This test function ensure that the if-block
    # will be executed.
    ρA = np.array([0.1, 0.3, 0.5, 0.7, 0.9])
    ρB = np.array([0.2, 0.3, 0.6, 0.8, 1.0])

    IA = np.array([0, 1, 2, 3, 4])
    IB = np.array([10, 1, 12, 13, 14])

    ref_ρ = ρA.copy()
    ref_I = IA.copy()

    comp_ρ = ρA.copy()
    comp_I = IA.copy()

    naive.merge_topk_ρI(ref_ρ, ρB.copy(), ref_I, IB.copy())
    core._merge_topk_ρI(comp_ρ, ρB.copy(), comp_I, IB.copy())

    npt.assert_almost_equal(ref_ρ, comp_ρ)
    npt.assert_almost_equal(ref_I, comp_I)


def test_shift_insert_at_index():
    for k in range(1, 6):
        a = np.random.rand(k)
        ref = np.empty(k, dtype=np.float64)
        comp = np.empty(k, dtype=np.float64)

        indices = np.arange(k + 1)
        values = np.random.rand(k + 1)

        # test shift = "right"
        for idx, v in zip(indices, values):
            ref[:] = a
            comp[:] = a

            ref = np.insert(ref, idx, v)[:-1]
            core._shift_insert_at_index(
                comp, idx, v, shift="right"
            )  # update comp in place

            npt.assert_almost_equal(ref, comp)

        # test shift = "left"
        for idx, v in zip(indices, values):
            ref[:] = a
            comp[:] = a

            ref = np.insert(ref, idx, v)[1:]
            core._shift_insert_at_index(
                comp, idx, v, shift="left"
            )  # update comp in place

            npt.assert_almost_equal(ref, comp)


def test_check_P():
    with pytest.raises(ValueError):
        core._check_P(np.random.rand(10).reshape(2, 5))


def test_find_matches_all():
    # max_matches: None, i.e. find all matches
    max_distance = np.inf
    D = np.random.rand(64)
    for excl_zone in range(3):
        ref = naive.find_matches(D, excl_zone, max_distance, max_matches=None)
        comp = core._find_matches(D, excl_zone, max_distance, max_matches=None)

        npt.assert_almost_equal(ref, comp)


def test_find_matches_maxmatch():
    max_distance = np.inf
    D = np.random.rand(64)
    for excl_zone in range(3):
        max_matches = np.random.randint(0, 100)
        ref = naive.find_matches(D, excl_zone, max_distance, max_matches)
        comp = core._find_matches(D, excl_zone, max_distance, max_matches)

        npt.assert_almost_equal(ref, comp)


@pytest.mark.filterwarnings("ignore", category=NumbaPerformanceWarning)
@patch("stumpy.config.STUMPY_THREADS_PER_BLOCK", TEST_THREADS_PER_BLOCK)
def test_gpu_searchsorted():
    if not cuda.is_available():  # pragma: no cover
        pytest.skip("Skipping Tests No GPUs Available")

    n = 3 * config.STUMPY_THREADS_PER_BLOCK + 1
    V = np.empty(n, dtype=np.float64)

    threads_per_block = config.STUMPY_THREADS_PER_BLOCK
    blocks_per_grid = math.ceil(n / threads_per_block)

    for k in range(1, 32):
        device_bfs = cuda.to_device(core._bfs_indices(k, fill_value=-1))
        nlevel = np.floor(np.log2(k) + 1).astype(np.int64)

        A = np.sort(np.random.rand(n, k), axis=1)
        device_A = cuda.to_device(A)

        V[:] = np.random.rand(n)
        for i, idx in enumerate(np.random.choice(np.arange(n), size=k, replace=False)):
            V[idx] = A[idx, i]  # create ties
        device_V = cuda.to_device(V)

        is_left = True  # test case
        ref_IDX = [np.searchsorted(A[i], V[i], side="left") for i in range(n)]
        ref_IDX = np.asarray(ref_IDX, dtype=np.int64)

        comp_IDX = np.full(n, -1, dtype=np.int64)
        device_comp_IDX = cuda.to_device(comp_IDX)
        _gpu_searchsorted_kernel[blocks_per_grid, threads_per_block](
            device_A, device_V, device_bfs, nlevel, is_left, device_comp_IDX
        )
        comp_IDX = device_comp_IDX.copy_to_host()
        npt.assert_array_equal(ref_IDX, comp_IDX)

        is_left = False  # test case
        ref_IDX = [np.searchsorted(A[i], V[i], side="right") for i in range(n)]
        ref_IDX = np.asarray(ref_IDX, dtype=np.int64)

        comp_IDX = np.full(n, -1, dtype=np.int64)
        device_comp_IDX = cuda.to_device(comp_IDX)
        _gpu_searchsorted_kernel[blocks_per_grid, threads_per_block](
            device_A, device_V, device_bfs, nlevel, is_left, device_comp_IDX
        )
        comp_IDX = device_comp_IDX.copy_to_host()
        npt.assert_array_equal(ref_IDX, comp_IDX)


def test_client_to_func():
    with pytest.raises(NotImplementedError):
        core._client_to_func(core)


def test_apply_include():
    D = np.random.uniform(-1000, 1000, [10, 20]).astype(np.float64)
    ref_D = np.empty(D.shape)
    comp_D = np.empty(D.shape)
    for width in range(D.shape[0]):
        for i in range(D.shape[0] - width):
            ref_D[:, :] = D[:, :]
            comp_D[:, :] = D[:, :]
            include = np.asarray(range(i, i + width + 1))

            naive.apply_include(D, include)
            core._apply_include(D, include)

            npt.assert_almost_equal(ref_D, comp_D)


@pytest.mark.parametrize("T_A, T_B", test_data)
def test_mpdist_custom_func(T_A, T_B):
    m = 3

    percentage = 0.05
    n_A = T_A.shape[0]
    n_B = T_B.shape[0]

    ref_mpdist = naive.mpdist(T_A, T_B, m)

    partial_stump = functools.partial(stump)
    partial_k_func = functools.partial(
        naive.mpdist_custom_func, m=m, percentage=percentage, n_A=n_A, n_B=n_B
    )
    comp_mpdist = core._mpdist(T_A, T_B, m, partial_stump, custom_func=partial_k_func)

    npt.assert_almost_equal(ref_mpdist, comp_mpdist)


@pytest.mark.parametrize("T_A, T_B", test_data)
def test_mpdist_with_isconstant(T_A, T_B):
    isconstant_custom_func = functools.partial(
        naive.isconstant_func_stddev_threshold, quantile_threshold=0.05
    )

    m = 3
    T_A_subseq_isconstant = isconstant_custom_func
    T_B_subseq_isconstant = isconstant_custom_func

    ref_mpdist = naive.mpdist(
        T_A,
        T_B,
        m,
        T_A_subseq_isconstant=T_A_subseq_isconstant,
        T_B_subseq_isconstant=T_B_subseq_isconstant,
    )

    partial_stump = functools.partial(
        stump,
        T_A_subseq_isconstant=T_A_subseq_isconstant,
        T_B_subseq_isconstant=T_B_subseq_isconstant,
    )
    comp_mpdist = core._mpdist(T_A, T_B, m, partial_stump)

    npt.assert_almost_equal(ref_mpdist, comp_mpdist)


@pytest.mark.parametrize("T_A, T_B", test_data)
def test_compute_P_ABBA(T_A, T_B):
    m = 3
    n_A = T_A.shape[0]
    n_B = T_B.shape[0]
    ref_P_ABBA = np.empty(n_A - m + 1 + n_B - m + 1, dtype=np.float64)
    comp_P_ABBA = np.empty(n_A - m + 1 + n_B - m + 1, dtype=np.float64)

    ref_P_ABBA[: n_A - m + 1] = naive.stump(T_A, m, T_B)[:, 0]
    ref_P_ABBA[n_A - m + 1 :] = naive.stump(T_B, m, T_A)[:, 0]

    partial_stump = functools.partial(stump)
    core._compute_P_ABBA(T_A, T_B, m, comp_P_ABBA, partial_stump)

    npt.assert_almost_equal(ref_P_ABBA, comp_P_ABBA)


@pytest.mark.parametrize("T_A, T_B", test_data)
def test_compute_P_ABBA_with_isconstant(T_A, T_B):
    isconstant_custom_func = functools.partial(
        naive.isconstant_func_stddev_threshold, quantile_threshold=0.05
    )

    m = 3
    n_A = T_A.shape[0]
    n_B = T_B.shape[0]

    T_A_subseq_isconstant = isconstant_custom_func
    T_B_subseq_isconstant = isconstant_custom_func

    ref_P_ABBA = np.empty(n_A - m + 1 + n_B - m + 1, dtype=np.float64)
    comp_P_ABBA = np.empty(n_A - m + 1 + n_B - m + 1, dtype=np.float64)

    ref_P_ABBA[: n_A - m + 1] = naive.stump(
        T_A,
        m,
        T_B,
        T_A_subseq_isconstant=T_A_subseq_isconstant,
        T_B_subseq_isconstant=T_B_subseq_isconstant,
    )[:, 0]
    ref_P_ABBA[n_A - m + 1 :] = naive.stump(
        T_B,
        m,
        T_A,
        T_A_subseq_isconstant=T_B_subseq_isconstant,
        T_B_subseq_isconstant=T_A_subseq_isconstant,
    )[:, 0]

    mp_func = functools.partial(
        stump,
        T_A_subseq_isconstant=T_A_subseq_isconstant,
        T_B_subseq_isconstant=T_B_subseq_isconstant,
    )
    core._compute_P_ABBA(
        T_A,
        T_B,
        m,
        comp_P_ABBA,
        mp_func,
    )

    npt.assert_almost_equal(ref_P_ABBA, comp_P_ABBA)


def test_process_isconstant_1d():
    isconstant_custom_func = functools.partial(
        naive.isconstant_func_stddev_threshold, quantile_threshold=0.05
    )

    n = 64
    m = 8

    # case 1: without nan
    T = np.random.rand(n)

    T_subseq_isconstant_ref = naive.rolling_isconstant(T, m, isconstant_custom_func)
    T_subseq_isconstant_comp = core.process_isconstant(T, m, isconstant_custom_func)

    npt.assert_array_equal(T_subseq_isconstant_ref, T_subseq_isconstant_comp)

    # case 2: with nan
    T = np.random.rand(n)
    idx = np.random.randint(n)
    T[idx] = np.nan
    T_subseq_isconstant = np.random.choice([True, False], n - m + 1, replace=True)

    T_subseq_isfinite = core.rolling_isfinite(T, m)

    T_subseq_isconstant_ref = T_subseq_isconstant & T_subseq_isfinite
    T_subseq_isconstant_comp = core.process_isconstant(T, m, T_subseq_isconstant)

    npt.assert_array_equal(T_subseq_isconstant_ref, T_subseq_isconstant_comp)


def test_process_isconstant_2d():
    isconstant_custom_func = functools.partial(
        naive.isconstant_func_stddev_threshold, quantile_threshold=0.05
    )

    n = 64
    m = 8
    d = 3

    # case 1: without nan
    T = np.random.rand(d, n)
    T_subseq_isconstant = [
        None,
        isconstant_custom_func,
        np.random.choice([True, False], n - m + 1, replace=True),
    ]

    T_subseq_isconstant_ref = np.array(
        [naive.rolling_isconstant(T[i], m, T_subseq_isconstant[i]) for i in range(d)]
    )

    T_subseq_isconstant_comp = core.process_isconstant(T, m, T_subseq_isconstant)

    npt.assert_array_equal(T_subseq_isconstant_ref, T_subseq_isconstant_comp)

    # case 2: with nan
    T = np.random.rand(d, n)
    i, j = np.random.choice(np.arange(n - m + 1), size=2, replace=False)
    T[-1, i : i + m] = 0.0
    T[-1, j : j + m] = 0.0
    T[-1, j] = np.nan

    T_subseq_isconstant = [
        None,
        isconstant_custom_func,
        np.full(n - m + 1, 0, dtype=bool),
    ]
    T_subseq_isconstant[-1][i] = True
    T_subseq_isconstant[-1][j] = True
    # Although T_subseq_isconstant[-1, j] should be set to False (since...
    # the subquence `T[-1, j:j+m]` is not finte), it is intentially set
    # to True to test the functionality of `process_isconstant` in handling
    # such conflict.

    T_subseq_isconstant_ref = np.array(
        [naive.rolling_isconstant(T[i], m, T_subseq_isconstant[i]) for i in range(d)]
    )
    T_subseq_isconstant_ref = T_subseq_isconstant_ref & core.rolling_isfinite(T, m)

    T_subseq_isconstant_comp = core.process_isconstant(T, m, T_subseq_isconstant)

    npt.assert_array_equal(T_subseq_isconstant_ref, T_subseq_isconstant_comp)


def test_process_isconstant_1d_default():
    # test the default value of `T_subseq_isconstant` in `process_isconstant`
    n = 64
    m = 8

    # case 1: without nan
    T = np.random.rand(n)
    T[:m] = 0.5  # constant subsequence

    T_subseq_isconstant_ref = naive.rolling_isconstant(T, m, a_subseq_isconstant=None)
    T_subseq_isconstant_comp = core.process_isconstant(T, m, T_subseq_isconstant=None)

    npt.assert_array_equal(T_subseq_isconstant_ref, T_subseq_isconstant_comp)

    # case 2: with nan
    T = np.random.rand(n)
    T[:m] = 0.5  # constant subsequence
    T[-m:] = np.nan  # non-finite subsequence

    T_subseq_isconstant_ref = naive.rolling_isconstant(T, m, a_subseq_isconstant=None)
    T_subseq_isconstant_comp = core.process_isconstant(T, m, T_subseq_isconstant=None)

    npt.assert_array_equal(T_subseq_isconstant_ref, T_subseq_isconstant_comp)


def test_update_incremental_PI_egressFalse():
    # This tests the function `core._update_incremental_PI`
    # when `egress` is False, meaning new data point is being
    # appended to the historical data.
    T = np.random.rand(64)
    t = np.random.rand()  # new datapoint
    T_new = np.append(T, t)

    m = 3
    excl_zone = int(np.ceil(m / config.STUMPY_EXCL_ZONE_DENOM))

    for k in range(1, 4):
        # ref
        mp_ref = naive.stump(T_new, m, row_wise=True, k=k)
        P_ref = mp_ref[:, :k].astype(np.float64)
        I_ref = mp_ref[:, k : 2 * k].astype(np.int64)

        # comp
        mp = naive.stump(T, m, row_wise=True, k=k)
        P_comp = mp[:, :k].astype(np.float64)
        I_comp = mp[:, k : 2 * k].astype(np.int64)

        # Because of the new data point, the length of matrix profile
        # and matrix profile indices should be increased by one.
        P_comp = np.pad(
            P_comp,
            [(0, 1), (0, 0)],
            mode="constant",
            constant_values=np.inf,
        )
        I_comp = np.pad(
            I_comp,
            [(0, 1), (0, 0)],
            mode="constant",
            constant_values=-1,
        )

        D = core.mass(T_new[-m:], T_new)
        core._update_incremental_PI(D, P_comp, I_comp, excl_zone, n_appended=0)

        # assertion
        npt.assert_almost_equal(P_ref, P_comp)
        npt.assert_almost_equal(I_ref, I_comp)


def test_update_incremental_PI_egressTrue():
    T = np.random.rand(64)
    t = np.random.rand()  # new data point
    m = 3
    excl_zone = int(np.ceil(m / config.STUMPY_EXCL_ZONE_DENOM))

    for k in range(1, 4):
        # ref
        # In egress=True mode, a new data point, t, is being appended
        # to the historical data, T, while the oldest data point is
        # being removed. Therefore, the first  subsequence in T
        # and the last subsequence does not get a chance to meet each
        # other. Therefore, we need to exclude that distance.

        T_with_t = np.append(T, t)
        D = naive.distance_matrix(T_with_t, T_with_t, m)
        D[-1, 0] = np.inf
        D[0, -1] = np.inf

        l = len(T_with_t) - m + 1
        P = np.empty((l, k), dtype=np.float64)
        I = np.empty((l, k), dtype=np.int64)
        for i in range(l):
            core.apply_exclusion_zone(D[i], i, excl_zone, np.inf)
            IDX = np.argsort(D[i], kind="mergesort")[:k]
            I[i] = IDX
            P[i] = D[i, IDX]

        P_ref = P[1:].copy()
        I_ref = I[1:].copy()

        # comp
        mp = naive.stump(T, m, row_wise=True, k=k)
        P_comp = mp[:, :k].astype(np.float64)
        I_comp = mp[:, k : 2 * k].astype(np.int64)

        P_comp[:-1] = P_comp[1:]
        P_comp[-1] = np.inf
        I_comp[:-1] = I_comp[1:]
        I_comp[-1] = -1

        T_new = np.append(T[1:], t)
        D = core.mass(T_new[-m:], T_new)
        core._update_incremental_PI(D, P_comp, I_comp, excl_zone, n_appended=1)

        # assertion
        npt.assert_almost_equal(P_ref, P_comp)
        npt.assert_almost_equal(I_ref, I_comp)


def test_update_incremental_PI_egressTrue_MemoryCheck():
    # This test function is to ensure that the function
    # `core._update_incremental_PI` does not forget the
    # nearest neighbors that were pointing to those old data
    # points that are removed in the `egress=True` mode.
    # This can be tested by inserting the same subsequence, s, in the beginning,
    # middle, and end of the time series. This is to allow us to know which
    # neighbor is the nearest neighbor to each of those three subsequences.

    # In the `egress=True` mode, the first element of the time series is removed and
    # a new data point is appended. However, the updated matrix profile index for the
    # middle subsequence `s` should still refer to  the first subsequence in
    # the historical data.
    seed = 0
    np.random.seed(seed)

    T = np.random.rand(64)
    m = 3
    excl_zone = int(np.ceil(m / config.STUMPY_EXCL_ZONE_DENOM))

    s = np.random.rand(m)
    T[:m] = s
    T[30 : 30 + m] = s
    T[-m:] = s

    t = np.random.rand()  # new data point
    T_with_t = np.append(T, t)

    # In egress=True mode, a new data point, t, is being appended
    # to the historical data, T, while the oldest data point is
    # being removed. Therefore, the first  subsequence in T
    # and the last subsequence does not get a chance to meet each
    # other. Therefore, their pairwise distances should be excluded
    # from the distance matrix.
    D = naive.distance_matrix(T_with_t, T_with_t, m)
    D[-1, 0] = np.inf
    D[0, -1] = np.inf

    l = len(T_with_t) - m + 1
    for i in range(l):
        core.apply_exclusion_zone(D[i], i, excl_zone, np.inf)

    T_new = np.append(T[1:], t)
    dist_profile = naive.distance_profile(T_new[-m:], T_new, m)
    core.apply_exclusion_zone(dist_profile, len(dist_profile) - 1, excl_zone, np.inf)

    for k in range(1, 4):
        # ref
        P = np.empty((l, k), dtype=np.float64)
        I = np.empty((l, k), dtype=np.int64)
        for i in range(l):
            IDX = np.argsort(D[i], kind="mergesort")[:k]
            I[i] = IDX
            P[i] = D[i, IDX]

        P_ref = P[1:].copy()
        I_ref = I[1:].copy()

        # comp
        mp = naive.stump(T, m, row_wise=True, k=k)
        P_comp = mp[:, :k].astype(np.float64)
        I_comp = mp[:, k : 2 * k].astype(np.int64)

        P_comp[:-1] = P_comp[1:]
        P_comp[-1] = np.inf
        I_comp[:-1] = I_comp[1:]
        I_comp[-1] = -1
        core._update_incremental_PI(
            dist_profile, P_comp, I_comp, excl_zone, n_appended=1
        )

        npt.assert_almost_equal(P_ref, P_comp)
        npt.assert_almost_equal(I_ref, I_comp)
