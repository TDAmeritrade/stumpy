import os
import numpy as np
import numpy.testing as npt
from matrix_profile import stamp, core
import pytest

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

test_data = [
    (np.array([9,8100,-60,7], dtype=np.float64), np.array([584,-11,23,79,1001,0,-19], dtype=np.float64)),
    (np.random.uniform(-1000, 1000, [8]), np.random.uniform(-1000, 1000, [64]))
]

@pytest.mark.parametrize("T_A, T_B", test_data)
def test_stamp_self_join(T_A, T_B):
    m = 3
    zone = int(np.ceil(m/2))
    left = np.array([naive_mass(Q, T_B, m, i, zone) for i, Q in enumerate(core.rolling_window(T_B, m))], dtype=object)
    right = stamp.stamp(T_B, T_B, m, ignore_trivial=True)
    replace_inf(left)
    replace_inf(right)
    npt.assert_almost_equal(left, right)

@pytest.mark.parametrize("T_A, T_B", test_data)
def test_stamp_A_B_join(T_A, T_B):
    m = 3
    left = np.array([naive_mass(Q, T_A, m) for Q in core.rolling_window(T_B, m)])
    right = stamp.stamp(T_A, T_B, m)
    replace_inf(left)
    replace_inf(right)
    npt.assert_almost_equal(left, right)

