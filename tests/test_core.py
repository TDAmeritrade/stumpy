import numpy as np
import numpy.testing as npt
import pandas as pd
from stumpy import core, config
import pytest
import os

import naive


def naive_rolling_window_dot_product(Q, T):
    window = len(Q)
    result = np.zeros(len(T) - window + 1)
    for i in range(len(result)):
        result[i] = np.dot(T[i : i + window], Q)
    return result


def test_check_dtype_float32():
    assert core.check_dtype(np.random.rand(10).astype(np.float32))


def test_check_dtype_float64():
    assert core.check_dtype(np.random.rand(10))


def test_check_window_size():
    for m in range(-1, 3):
        with pytest.raises(ValueError):
            core.check_window_size(m)


def naive_compute_mean_std(T, m):
    n = T.shape[0]

    M_T = np.zeros(n - m + 1, dtype=float)
    Σ_T = np.zeros(n - m + 1, dtype=float)

    for i in range(n - m + 1):
        Q = T[i : i + m].copy()
        Q[np.isinf(Q)] = np.nan

        M_T[i] = np.mean(Q)
        Σ_T[i] = np.nanstd(Q)

    M_T[np.isnan(M_T)] = np.inf
    Σ_T[np.isnan(Σ_T)] = 0
    return M_T, Σ_T


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


test_data = [
    (np.array([-1, 1, 2], dtype=np.float64), np.array(range(5), dtype=np.float64)),
    (
        np.array([9, 8100, -60], dtype=np.float64),
        np.array([584, -11, 23, 79, 1001], dtype=np.float64),
    ),
    (np.random.uniform(-1000, 1000, [8]), np.random.uniform(-1000, 1000, [64])),
]


@pytest.mark.parametrize("Q, T", test_data)
def test_sliding_dot_product(Q, T):
    ref_mp = naive_rolling_window_dot_product(Q, T)
    comp_mp = core.sliding_dot_product(Q, T)
    npt.assert_almost_equal(ref_mp, comp_mp)


@pytest.mark.parametrize("Q, T", test_data)
def test_compute_mean_std(Q, T):
    m = Q.shape[0]

    ref_μ_Q, ref_σ_Q = naive_compute_mean_std(Q, m)
    ref_M_T, ref_Σ_T = naive_compute_mean_std(T, m)
    comp_μ_Q, comp_σ_Q = core.compute_mean_std(Q, m)
    comp_M_T, comp_Σ_T = core.compute_mean_std(T, m)

    npt.assert_almost_equal(ref_μ_Q, comp_μ_Q)
    npt.assert_almost_equal(ref_σ_Q, comp_σ_Q)
    npt.assert_almost_equal(ref_M_T, comp_M_T)
    npt.assert_almost_equal(ref_Σ_T, comp_Σ_T)


@pytest.mark.parametrize("Q, T", test_data)
def test_compute_mean_std_chunked(Q, T):
    m = Q.shape[0]

    config.STUMPY_MEAN_STD_NUM_CHUNKS = 2
    ref_μ_Q, ref_σ_Q = naive_compute_mean_std(Q, m)
    ref_M_T, ref_Σ_T = naive_compute_mean_std(T, m)
    comp_μ_Q, comp_σ_Q = core.compute_mean_std(Q, m)
    comp_M_T, comp_Σ_T = core.compute_mean_std(T, m)
    config.STUMPY_MEAN_STD_NUM_CHUNKS = 1

    npt.assert_almost_equal(ref_μ_Q, comp_μ_Q)
    npt.assert_almost_equal(ref_σ_Q, comp_σ_Q)
    npt.assert_almost_equal(ref_M_T, comp_M_T)
    npt.assert_almost_equal(ref_Σ_T, comp_Σ_T)


@pytest.mark.parametrize("Q, T", test_data)
def test_compute_mean_std_chunked_many(Q, T):
    m = Q.shape[0]

    config.STUMPY_MEAN_STD_NUM_CHUNKS = 128
    ref_μ_Q, ref_σ_Q = naive_compute_mean_std(Q, m)
    ref_M_T, ref_Σ_T = naive_compute_mean_std(T, m)
    comp_μ_Q, comp_σ_Q = core.compute_mean_std(Q, m)
    comp_M_T, comp_Σ_T = core.compute_mean_std(T, m)
    config.STUMPY_MEAN_STD_NUM_CHUNKS = 1

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

    config.STUMPY_MEAN_STD_NUM_CHUNKS = 2
    ref_μ_Q, ref_σ_Q = naive_compute_mean_std_multidimensional(Q, m)
    ref_M_T, ref_Σ_T = naive_compute_mean_std_multidimensional(T, m)
    comp_μ_Q, comp_σ_Q = core.compute_mean_std(Q, m)
    comp_M_T, comp_Σ_T = core.compute_mean_std(T, m)
    config.STUMPY_MEAN_STD_NUM_CHUNKS = 1

    npt.assert_almost_equal(ref_μ_Q, comp_μ_Q)
    npt.assert_almost_equal(ref_σ_Q, comp_σ_Q)
    npt.assert_almost_equal(ref_M_T, comp_M_T)
    npt.assert_almost_equal(ref_Σ_T, comp_Σ_T)


@pytest.mark.parametrize("Q, T", test_data)
def test_compute_mean_std_multidimensional_chunked_many(Q, T):
    m = Q.shape[0]

    Q = np.array([Q, np.random.uniform(-1000, 1000, [Q.shape[0]])])
    T = np.array([T, T, np.random.uniform(-1000, 1000, [T.shape[0]])])

    config.STUMPY_MEAN_STD_NUM_CHUNKS = 128
    ref_μ_Q, ref_σ_Q = naive_compute_mean_std_multidimensional(Q, m)
    ref_M_T, ref_Σ_T = naive_compute_mean_std_multidimensional(T, m)
    comp_μ_Q, comp_σ_Q = core.compute_mean_std(Q, m)
    comp_M_T, comp_Σ_T = core.compute_mean_std(T, m)
    config.STUMPY_MEAN_STD_NUM_CHUNKS = 1

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
    μ_Q, σ_Q = core.compute_mean_std(Q, m)
    M_T, Σ_T = core.compute_mean_std(T, m)
    comp = core._calculate_squared_distance_profile(
        m, QT, μ_Q.item(0), σ_Q.item(0), M_T, Σ_T
    )
    npt.assert_almost_equal(ref, comp)


@pytest.mark.parametrize("Q, T", test_data)
def test_calculate_distance_profile(Q, T):
    m = Q.shape[0]
    ref = np.linalg.norm(
        core.z_norm(core.rolling_window(T, m), 1) - core.z_norm(Q), axis=1
    )
    QT = core.sliding_dot_product(Q, T)
    μ_Q, σ_Q = core.compute_mean_std(Q, m)
    M_T, Σ_T = core.compute_mean_std(T, m)
    comp = core.calculate_distance_profile(m, QT, μ_Q.item(0), σ_Q.item(0), M_T, Σ_T)
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
def test_mass_asbolute(Q, T):
    Q = Q.copy()
    T = T.copy()
    m = Q.shape[0]
    ref = np.linalg.norm(core.rolling_window(T, m) - Q, axis=1)
    comp = core.mass_absolute(Q, T)
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


def test_apply_exclusion_zone():
    T = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9], dtype=float)
    ref = np.empty(T.shape)
    comp = np.empty(T.shape)
    exclusion_zone = 2

    for i in range(T.shape[0]):
        ref[:] = T[:]
        naive.apply_exclusion_zone(ref, i, exclusion_zone)

        comp[:] = T[:]
        core.apply_exclusion_zone(comp, i, exclusion_zone)

        naive.replace_inf(ref)
        naive.replace_inf(comp)
        npt.assert_array_equal(ref, comp)


def test_apply_exclusion_zone_multidimensional():
    T = np.array(
        [[0, 1, 2, 3, 4, 5, 6, 7, 8, 9], [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]], dtype=float
    )
    ref = np.empty(T.shape)
    comp = np.empty(T.shape)
    exclusion_zone = 2

    for i in range(T.shape[1]):
        ref[:, :] = T[:, :]
        naive.apply_exclusion_zone(ref, i, exclusion_zone)

        comp[:, :] = T[:, :]
        core.apply_exclusion_zone(comp, i, exclusion_zone)

        naive.replace_inf(ref)
        naive.replace_inf(comp)
        npt.assert_array_equal(ref, comp)


def test_preprocess():
    T = np.array([0, np.nan, 2, 3, 4, 5, 6, 7, np.inf, 9])
    m = 3

    ref_T = np.array([0, 0, 2, 3, 4, 5, 6, 7, 0, 9], dtype=float)
    ref_M, ref_Σ = naive_compute_mean_std(T, m)

    comp_T, comp_M, comp_Σ = core.preprocess(T, m)

    npt.assert_almost_equal(ref_T, comp_T)
    npt.assert_almost_equal(ref_M, comp_M)
    npt.assert_almost_equal(ref_Σ, comp_Σ)

    T = pd.Series(T)
    comp_T, comp_M, comp_Σ = core.preprocess(T, m)

    npt.assert_almost_equal(ref_T, comp_T)
    npt.assert_almost_equal(ref_M, comp_M)
    npt.assert_almost_equal(ref_Σ, comp_Σ)


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
    ref_M, ref_Σ = naive_compute_mean_std(ref_T, m)
    ref_Σ_inverse = 1.0 / ref_Σ
    ref_M_m_1, _ = naive_compute_mean_std(ref_T, m - 1)

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
                diags = np.random.permutation(range(-(n_A - m + 1) + 1, n_B - m + 1))
                ones_matrix = np.ones((n_A - m + 1, n_B - m + 1), dtype=np.int64)
                ref_ndist_counts = np.empty(len(diags))
                for i, diag in enumerate(diags):
                    ref_ndist_counts[i] = ones_matrix.diagonal(offset=diag).sum()

            comp_ndist_counts = core._count_diagonal_ndist(diags, m, n_A, n_B)

            npt.assert_almost_equal(ref_ndist_counts, comp_ndist_counts)


def test_get_array_ranges():
    x = np.array([3, 9, 2, 1, 5, 4, 7, 7, 8, 6])
    for n_chunks in range(2, 5):
        ref = naive.get_array_ranges(x, n_chunks)

        comp = core._get_array_ranges(x, n_chunks)
        npt.assert_almost_equal(ref, comp)


def test_get_array_ranges_exhausted():
    x = np.array([3, 3, 3, 11, 11, 11])
    n_chunks = 6

    ref = naive.get_array_ranges(x, n_chunks)

    comp = core._get_array_ranges(x, n_chunks)
    npt.assert_almost_equal(ref, comp)


def test_get_array_ranges_exhausted_truncated():
    x = np.array([3, 3, 3, 11, 11, 11])
    n_chunks = 6

    ref = naive.get_array_ranges(x, n_chunks, truncate=True)

    comp = core._get_array_ranges(x, n_chunks, truncate=True)
    npt.assert_almost_equal(ref, comp)
