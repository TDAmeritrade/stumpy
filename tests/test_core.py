import numpy as np
import numpy.testing as npt
from matrix_profile import core

def naive_rolling_window_dot_product(Q, T):
    window = len(Q)
    result = np.zeros(len(T) - window + 1)
    for i in range(len(result)):
        result[i] = np.dot(T[i:i + window], Q)
    return result

def rolling_window(a, window):
    shape = a.shape[:-1] + (a.shape[-1] - window + 1, window)
    strides = a.strides + (a.strides[-1],)
    return np.lib.stride_tricks.as_strided(a, shape=shape, strides=strides)

def z_norm(x, axis=0):
    return (x - np.mean(x, axis, keepdims=True))/np.std(x, axis, keepdims=True)

class TestCore:
    def setUp(self):
        pass

    def tearDown(self):
        pass

    def test_sliding_dot_product_with_range_16(self):
        T = np.array(range(16))
        Q = np.array([3,4,5,6])
        left = naive_rolling_window_dot_product(Q, T)
        right = core.sliding_dot_product(Q, T)
        npt.assert_almost_equal(left, right)

    def test_sliding_dot_product_with_range_neg_16(self):
        T = np.negative(np.array(range(16)))
        Q = np.array([3,4,5,6])
        left = naive_rolling_window_dot_product(Q, T)
        right = core.sliding_dot_product(Q, T)
        npt.assert_almost_equal(left, right)

    def test_sliding_dot_product_with_random_T64_Q8(self):
        # Select 64 random floats in range [-1000, 1000]
        T = np.random.uniform(-1000, 1000, [64])
        # Select 8 random floats in range [-1000, 1000]
        Q = np.random.uniform(-1000, 1000, [8])
        left = naive_rolling_window_dot_product(Q, T)
        right = core.sliding_dot_product(Q, T)
        npt.assert_almost_equal(left, right)

    def test_compute_mean_std(self):
        # Select 64 random floats in range [-1000, 1000]
        T = np.random.uniform(-1000, 1000, [64])
        # Select 8 random floats in range [-1000, 1000]
        Q = np.random.uniform(-1000, 1000, [8])
        m = Q.shape[0]
        left_μ_Q = np.sum(Q)/m
        left_σ_Q = np.sqrt(np.sum(np.square(Q))/m - np.square(left_μ_Q))
        left_M_T = np.mean(rolling_window(T, m), 1)
        left_Σ_T = np.std(rolling_window(T,m), 1)
        right_μ_Q, right_σ_Q, right_M_T, right_Σ_T = core.compute_mean_std(Q, T)

        npt.assert_almost_equal(left_μ_Q, right_μ_Q)
        npt.assert_almost_equal(left_σ_Q, right_σ_Q)
        npt.assert_almost_equal(left_M_T, right_M_T)
        npt.assert_almost_equal(left_Σ_T, right_Σ_T)

    def test_calculate_distance_profile(self):
        T = np.array(range(6))
        Q = np.array(range(3))
        m = Q.shape[0]
        left = np.linalg.norm(z_norm(rolling_window(T,m), 1) - z_norm(Q), axis=1)
        QT = core.sliding_dot_product(Q, T)
        μ_Q, σ_Q, M_T, Σ_T = core.compute_mean_std(Q, T)
        right = core.calculate_distance_profile(m, QT, μ_Q, σ_Q, M_T, Σ_T)
        npt.assert_almost_equal(left, right)