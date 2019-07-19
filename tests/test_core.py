import numpy as np
import numpy.testing as npt
from stumpy import core
import pytest


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
    left_μ_Q = np.sum(Q) / m
    left_σ_Q = np.sqrt(np.sum(np.square(Q - left_μ_Q) / m))
    left_M_T = np.mean(core.rolling_window(T, m), axis=1)
    left_Σ_T = np.std(core.rolling_window(T, m), axis=1)
    right_μ_Q, right_σ_Q = core.compute_mean_std(Q, m)
    right_M_T, right_Σ_T = core.compute_mean_std(T, m)
    npt.assert_almost_equal(left_μ_Q, right_μ_Q)
    npt.assert_almost_equal(left_σ_Q, right_σ_Q)
    npt.assert_almost_equal(left_M_T, right_M_T)
    npt.assert_almost_equal(left_Σ_T, right_Σ_T)


@pytest.mark.parametrize("Q, T", test_data)
def test_calculate_distance_profile(Q, T):
    m = Q.shape[0]
    left = np.linalg.norm(
        core.z_norm(core.rolling_window(T, m), 1) - core.z_norm(Q), axis=1
    )
    QT = core.sliding_dot_product(Q, T)
    μ_Q, σ_Q = core.compute_mean_std(Q, m)
    M_T, Σ_T = core.compute_mean_std(T, m)
    right = core.calculate_distance_profile(m, QT, μ_Q, σ_Q, M_T, Σ_T)
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
