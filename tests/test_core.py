import numpy as np
import numpy.testing as npt
from stumpy import core
import pytest
import os

import utils


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

    cumsum_T = np.empty(len(T) + 1)
    np.cumsum(T, out=cumsum_T[1:])  # store output in cumsum_T[1:]
    cumsum_T[0] = 0

    cumsum_T_squared = np.empty(len(T) + 1)
    np.cumsum(np.square(T), out=cumsum_T_squared[1:])
    cumsum_T_squared[0] = 0

    subseq_sum_T = cumsum_T[m:] - cumsum_T[: n - m + 1]
    subseq_sum_T_squared = cumsum_T_squared[m:] - cumsum_T_squared[: n - m + 1]
    M_T = subseq_sum_T / m
    Σ_T = np.abs((subseq_sum_T_squared / m) - np.square(M_T))
    Σ_T = np.sqrt(Σ_T)

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
    left = naive_rolling_window_dot_product(Q, T)
    right = core.sliding_dot_product(Q, T)
    npt.assert_almost_equal(left, right)


@pytest.mark.parametrize("Q, T", test_data)
def test_compute_mean_std(Q, T):
    m = Q.shape[0]

    left_μ_Q, left_σ_Q = naive_compute_mean_std(Q, m)
    left_M_T, left_Σ_T = naive_compute_mean_std(T, m)
    right_μ_Q, right_σ_Q = core.compute_mean_std(Q, m)
    right_M_T, right_Σ_T = core.compute_mean_std(T, m)

    npt.assert_almost_equal(left_μ_Q, right_μ_Q)
    npt.assert_almost_equal(left_σ_Q, right_σ_Q)
    npt.assert_almost_equal(left_M_T, right_M_T)
    npt.assert_almost_equal(left_Σ_T, right_Σ_T)


@pytest.mark.parametrize("Q, T", test_data)
def test_compute_mean_std_multidimensional(Q, T):
    m = Q.shape[0]

    Q = np.array([Q, np.random.uniform(-1000, 1000, [Q.shape[0]])])
    T = np.array([T, T, np.random.uniform(-1000, 1000, [T.shape[0]])])

    left_μ_Q, left_σ_Q = naive_compute_mean_std_multidimensional(Q, m)
    left_M_T, left_Σ_T = naive_compute_mean_std_multidimensional(T, m)
    right_μ_Q, right_σ_Q = core.compute_mean_std(Q, m)
    right_M_T, right_Σ_T = core.compute_mean_std(T, m)

    npt.assert_almost_equal(left_μ_Q, right_μ_Q)
    npt.assert_almost_equal(left_σ_Q, right_σ_Q)
    npt.assert_almost_equal(left_M_T, right_M_T)
    npt.assert_almost_equal(left_Σ_T, right_Σ_T)


@pytest.mark.parametrize("Q, T", test_data)
def test_calculate_squared_distance_profile(Q, T):
    m = Q.shape[0]
    left = (
        np.linalg.norm(
            core.z_norm(core.rolling_window(T, m), 1) - core.z_norm(Q), axis=1
        )
        ** 2
    )
    QT = core.sliding_dot_product(Q, T)
    μ_Q, σ_Q = core.compute_mean_std(Q, m)
    M_T, Σ_T = core.compute_mean_std(T, m)
    right = core._calculate_squared_distance_profile(
        m, QT, μ_Q.item(0), σ_Q.item(0), M_T, Σ_T
    )
    npt.assert_almost_equal(left, right)


@pytest.mark.parametrize("Q, T", test_data)
def test_calculate_distance_profile(Q, T):
    m = Q.shape[0]
    left = np.linalg.norm(
        core.z_norm(core.rolling_window(T, m), 1) - core.z_norm(Q), axis=1
    )
    QT = core.sliding_dot_product(Q, T)
    μ_Q, σ_Q = core.compute_mean_std(Q, m)
    M_T, Σ_T = core.compute_mean_std(T, m)
    right = core.calculate_distance_profile(m, QT, μ_Q.item(0), σ_Q.item(0), M_T, Σ_T)
    npt.assert_almost_equal(left, right)


@pytest.mark.parametrize("Q, T", test_data)
def test_mueen_calculate_distance_profile(Q, T):
    m = Q.shape[0]
    left = np.linalg.norm(
        core.z_norm(core.rolling_window(T, m), 1) - core.z_norm(Q), axis=1
    )
    right = core.mueen_calculate_distance_profile(Q, T)
    npt.assert_almost_equal(left, right)


@pytest.mark.parametrize("Q, T", test_data)
def test_mass(Q, T):
    m = Q.shape[0]
    left = np.linalg.norm(
        core.z_norm(core.rolling_window(T, m), 1) - core.z_norm(Q), axis=1
    )
    right = core.mass(Q, T)
    npt.assert_almost_equal(left, right)


def test_array_to_temp_file():
    left = np.random.rand()
    fname = core.array_to_temp_file(left)
    right = np.load(fname, allow_pickle=False)
    os.remove(fname)

    npt.assert_almost_equal(left, right)
