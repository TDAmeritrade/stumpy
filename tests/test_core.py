import numpy as np
import numpy.testing as npt
from matrix_profile import core

def naive_rolling_window_dot_product(Q, T):
    window = len(Q)
    result = np.zeros(len(T) - window + 1)
    for i in range(len(result)):
        result[i] = np.dot(T[i:i + window], Q)
    return result

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
        # Select 50 random floats in range [-1000, 1000]
        T = np.random.uniform(-1000, 1000, [64])
        # Select 5 random floats in range [-1000, 1000]
        Q = np.random.uniform(-1000, 1000, [8])
        left = naive_rolling_window_dot_product(Q, T)
        right = core.sliding_dot_product(Q, T)
        npt.assert_almost_equal(left, right)

