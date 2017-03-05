import numpy as np
import numpy.testing as npt
from matrix_profile import core

def naive_rolling_window_dot_product(Q, T):
    window = len(Q)
    result = np.zeros(len(T) - window + 1)
    for i in range(len(result)):
        result[i] = np.dot(T[i:i + window], Q)
    return result

def z_norm(x, axis=0):
    return (x - np.mean(x, axis, keepdims=True))/np.std(x, axis, keepdims=True)

class TestCore:
    def test_generator(self):
        """
        To add a new test function below, the function
        name must start with 'run' instead of 'test' 
        """
        arrays = [
                  (np.array([-1,1,2], dtype=np.float64),
                   np.array(range(5), dtype=np.float64)),
                  (np.random.uniform(-1000, 1000, [8]),
                   np.random.uniform(-1000, 1000, [64])),
                 ]
        funcs = [f for f in dir(self) if f.startswith('run')]
        for func in funcs:
            for Q, T in arrays:
                yield func, Q, T

    def setUp(self):
        pass

    def tearDown(self):
        pass

    def run_sliding_dot_product(self, Q, T):
        left = naive_rolling_window_dot_product(Q, T)
        right = core.sliding_dot_product(Q, T)
        npt.assert_almost_equal(left, right)

    def run_compute_mean_std(self, Q, T):
        m = Q.shape[0]
        left_μ_Q = np.sum(Q)/m
        left_σ_Q = np.sqrt(np.sum(np.square(Q-left_μ_Q)/m))
        left_M_T = np.mean(core.rolling_window(T, m), axis=1)
        left_Σ_T = np.std(core.rolling_window(T, m), axis=1)
        right_μ_Q, right_σ_Q, right_M_T, right_Σ_T = core.compute_mean_std(Q, T)
        npt.assert_almost_equal(left_μ_Q, right_μ_Q)
        npt.assert_almost_equal(left_σ_Q, right_σ_Q)
        npt.assert_almost_equal(left_M_T, right_M_T)
        npt.assert_almost_equal(left_Σ_T, right_Σ_T)

    def run_calculate_distance_profile(self, Q, T):
        m = Q.shape[0]
        left = np.linalg.norm(z_norm(core.rolling_window(T, m), 1) - z_norm(Q), axis=1)
        QT = core.sliding_dot_product(Q, T)
        μ_Q, σ_Q, M_T, Σ_T = core.compute_mean_std(Q, T)
        right = core.calculate_distance_profile(m, QT, μ_Q, σ_Q, M_T, Σ_T)
        #right = core.mueen_calculate_distance_profile(Q,T)
        npt.assert_almost_equal(left, right)

    def run_calculate_distance_profile(self, Q, T):
        m = Q.shape[0]       
        left = np.linalg.norm(z_norm(core.rolling_window(T, m), 1) - z_norm(Q), axis=1)
        QT = core.sliding_dot_product(Q, T)
        μ_Q, σ_Q, M_T, Σ_T = core.compute_mean_std(Q, T)
        right = core.calculate_distance_profile(m, QT, μ_Q, σ_Q, M_T, Σ_T)
        #right = core.mueen_calculate_distance_profile(Q,T)
        npt.assert_almost_equal(left, right)

    def run_mass(self, Q, T):
        m = Q.shape[0]
        QT = core.sliding_dot_product(Q, T)
        μ_Q, σ_Q, M_T, Σ_T = core.compute_mean_std(Q, T)
        #right = core.calculate_distance_profile(m, QT, μ_Q, σ_Q, M_T, Σ_T)
        #print(left)

