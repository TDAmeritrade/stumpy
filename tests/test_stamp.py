import os
import numpy as np
import numpy.testing as npt
from matrix_profile import stamp, core
from functools import partial
import os

# From Matrix Profile Authors
TEST_DATA = os.path.join(os.path.dirname(__file__), 'testData.txt')
TEST_INP_INDICES = os.path.join(os.path.dirname(__file__), 'testIdx.txt')
TEST_MATRIX_PROFILE = os.path.join(os.path.dirname(__file__), 'testMP.txt')
TEST_OUT_INDICES = os.path.join(os.path.dirname(__file__), 'testMPI.txt')

def naive_mass(Q, T, m, trivial_idx=None, excl_zone=0):
    D = np.linalg.norm(core.z_norm(core.rolling_window(T, m), 1) - core.z_norm(Q), axis=1)
    if trivial_idx is not None:
            start = max(0, trivial_idx - excl_zone)
            stop = trivial_idx + excl_zone+1
            D[start:stop] = np.inf
    I = np.argmin(D)
    P = D[I]
    return P, I

def replace_inf(x, value=0):
    x[x == np.inf] = value
    x[x == -np.inf] = value
    return

class TestStamp:
    def get_description(self):
        test_dir = os.path.basename(os.path.dirname(__file__))
        cls = self.__class__.__name__
        base = os.path.basename(__file__)
        name = os.path.splitext(base)[0]
        return '{}.{}.{}'.format(test_dir, name, cls)

    def test_generator(self):
        test_dir = os.path.basename(os.path.dirname(__file__))
        cls = self.__class__.__name__
        base = os.path.basename(__file__)
        name = os.path.splitext(base)[0]
        arrays = [
                  (np.array([9,8100,-60,7], dtype=np.float64),
                   np.array([584,-11,23,79,1001,0,-19], dtype=np.float64)),
                  (np.random.uniform(-1000, 1000, [8]),
                   np.random.uniform(-1000, 1000, [64])),
                 ]
        funcs = [f for f in dir(self) if f.startswith('run')]
        for func in funcs:
            for i, (T_A, T_B) in enumerate(arrays):
                f = partial(getattr(self, func), T_A, T_B)
                descr = self.get_description()
                f.description = '{}.{} {}'.format(descr, func, i+1) 
                yield f

    def setUp(self):
        self.test_data = np.loadtxt(TEST_DATA, dtype=np.float64, skiprows=2)
        self.test_inp_indices = np.loadtxt(TEST_INP_INDICES, dtype=np.int32, skiprows=1)
        self.test_inp_indices = self.test_inp_indices - 1  # Fix off-by-one index
        # Reorder test data with input indices from Matrix Profile Authors
        self.test_data = self.test_data[self.test_inp_indices]  # Reorder test data with input indices

        self.test_matrix_profile = np.loadtxt(TEST_MATRIX_PROFILE, dtype=np.float64, skiprows=0)
        self.test_out_indices = np.loadtxt(TEST_OUT_INDICES, dtype=np.int32, skiprows=1)
        self.test_out_indices = self.test_out_indices - 1  # Fix off-by-one index

    def tearDown(self):
        pass

    def run_stamp_self_join(self, T_A, T_B):
        m = 3
        zone = int(np.ceil(m/2))
        left = np.array([naive_mass(Q, T_B, m, i, zone) for i, Q in enumerate(core.rolling_window(T_B, m))], dtype=object)
        right = stamp.stamp(T_B, T_B, m, ignore_trivial=True)
        replace_inf(left)
        replace_inf(right)
        npt.assert_almost_equal(left, right)

    def run_stamp_A_B_join(self, T_A, T_B):
        m = 3
        left = np.array([naive_mass(Q, T_A, m) for Q in core.rolling_window(T_B, m)])
        right = stamp.stamp(T_A, T_B, m)
        replace_inf(left)
        replace_inf(right)
        npt.assert_almost_equal(left, right)