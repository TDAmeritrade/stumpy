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

class TestCore:
    def setUp(self):
        pass

    def tearDown(self):
        pass

    def test_sliding_dot_product_with_range_10(self):
        T = np.array(range(10))
        Q = np.array([3,4,5])
        left = naive_rolling_window_dot_product(Q, T)
        right = core.numpy_sliding_dot_product(Q, T)
        npt.assert_almost_equal(left, right)

    def test_sliding_dot_product_with_range_neg_10(self):
        T = np.negative(np.array(range(10)))
        Q = np.array([3,4,5])
        left = naive_rolling_window_dot_product(Q, T)
        right = core.numpy_sliding_dot_product(Q, T)
        npt.assert_almost_equal(left, right)

    def test_sliding_dot_product_with_random_T50_Q5(self):
        # Select 50 random floats in range [-1000, 1000]
        T = np.random.uniform(-1000, 1000, [50])
        # Select 5 random floats in range [-1000, 1000]
        Q = np.random.uniform(-1000, 1000, [5])
        left = naive_rolling_window_dot_product(Q, T)
        right = core.numpy_sliding_dot_product(Q, T)
        npt.assert_almost_equal(left, right)

    def test_sliding_dot_product_with_range_10(self):
       T = np.array(range(10))
       Q = np.array([3,4,5])
       left = np.dot(rolling_window(T, len(Q)), Q)
       right = core.numpy_sliding_dot_product(Q, T)
       npt.assert_almost_equal(left, right)

    def test_scipy_fftconvolve_sliding_dot_product_with_random_T50_Q5(self):
        # Select 50 random floats in range [-1000, 1000]
        T = np.random.uniform(-1000, 1000, [50])
        # Select 5 random floats in range [-1000, 1000]
        Q = np.random.uniform(-1000, 1000, [5])
        left = naive_rolling_window_dot_product(Q, T)
        right = core.scipy_fftconvolve_sliding_dot_product(Q, T)
        npt.assert_almost_equal(left, right)

