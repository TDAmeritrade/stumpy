import numpy as np
import numpy.testing as npt
from stumpy import stamp, core
import pytest
import utils


def naive_mass(Q, T, m, trivial_idx=None, excl_zone=0):
    D = np.linalg.norm(
        utils.z_norm(core.rolling_window(T, m), 1) - utils.z_norm(Q), axis=1
    )
    if trivial_idx is not None:
        start = max(0, trivial_idx - excl_zone)
        stop = min(T.shape[0] - Q.shape[0] + 1, trivial_idx + excl_zone)
        D[start:stop] = np.inf
    I = np.argmin(D)
    P = D[I]

    if P == np.inf:
        I = -1

    return P, I


test_data = [
    (
        np.array([9, 8100, -60, 7], dtype=np.float64),
        np.array([584, -11, 23, 79, 1001, 0, -19], dtype=np.float64),
    ),
    (
        np.random.uniform(-1000, 1000, [8]).astype(np.float64),
        np.random.uniform(-1000, 1000, [64]).astype(np.float64),
    ),
]


@pytest.mark.parametrize("T_A, T_B", test_data)
def test_stamp_self_join(T_A, T_B):
    m = 3
    zone = int(np.ceil(m / 2))
    left = np.array(
        [
            utils.naive_mass(Q, T_B, m, i, zone, ignore_trivial=True)[0:2]
            for i, Q in enumerate(core.rolling_window(T_B, m))
        ],
        dtype=object,
    )
    right = stamp.stamp(T_B, T_B, m, ignore_trivial=True)
    utils.replace_inf(left)
    utils.replace_inf(right)
    npt.assert_almost_equal(left, right)


@pytest.mark.parametrize("T_A, T_B", test_data)
def test_stamp_A_B_join(T_A, T_B):
    m = 3
    left = np.array(
        [utils.naive_mass(Q, T_A, m)[0:2] for Q in core.rolling_window(T_B, m)], dtype=object
    )
    right = stamp.stamp(T_A, T_B, m)
    utils.replace_inf(left)
    utils.replace_inf(right)
    npt.assert_almost_equal(left, right)

def test_stamp_nan_selfjoin_beginning():
    m=3
    T = np.array([np.nan, 1, 0, 0, 1, 0, 0])

    zone = int(np.ceil(m / 2))
    left = np.array(
        [
            utils.naive_mass(Q, T, m, i, zone, ignore_trivial=True)[0:2]
            for i, Q in enumerate(core.rolling_window(T, m))
        ],
        dtype=object,
    )

    right = stamp.stamp(T, T, m, ignore_trivial=True)

    utils.replace_inf(left)
    utils.replace_inf(right)
    npt.assert_almost_equal(left, right)

def test_stamp_inf_selfjoin_beginning():
    m=3
    T = np.array([np.inf, 1, 0, 0, 1, 0, 0])

    zone = int(np.ceil(m / 2))
    left = np.array(
        [
            utils.naive_mass(Q, T, m, i, zone, ignore_trivial=True)[0:2]
            for i, Q in enumerate(core.rolling_window(T, m))
        ],
        dtype=object,
    )

    right = stamp.stamp(T, T, m, ignore_trivial=True)

    utils.replace_inf(left)
    utils.replace_inf(right)
    npt.assert_almost_equal(left, right)

@pytest.mark.parametrize("T_A, T_B", test_data)
def test_stamp_nan_inf_selfjoin(T_A, T_B):
    m=3
    T_B_nan_inf = np.random.uniform(size=len(T_B))
    T_B[T_B_nan_inf > 0.90] = np.nan
    T_B[T_B_nan_inf > 0.95] = np.inf

    zone = int(np.ceil(m / 2))
    left = np.array(
        [
            utils.naive_mass(Q, T_B, m, i, zone, ignore_trivial=True)[0:2]
            for i, Q in enumerate(core.rolling_window(T_B, m))
        ],
        dtype=object,
    )

    right = stamp.stamp(T_B, T_B, m, ignore_trivial=True)
    
    utils.replace_inf(left)
    utils.replace_inf(right)
    npt.assert_almost_equal(left, right)

@pytest.mark.parametrize("T_A, T_B", test_data)
def test_stamp_nan_inf_A_B_join(T_A, T_B):
    m=3
    T_A_nan_inf = np.random.uniform(size=len(T_A))
    T_B_nan_inf = np.random.uniform(size=len(T_B))
    T_A[T_A_nan_inf > 0.90] = np.nan
    T_A[T_A_nan_inf > 0.95] = np.inf
    T_B[T_B_nan_inf > 0.90] = np.nan
    T_B[T_B_nan_inf > 0.95] = np.inf

    left = np.array(
        [utils.naive_mass(Q, T_A, m)[0:2] for Q in core.rolling_window(T_B, m)], dtype=object
    )

    right = stamp.stamp(T_A, T_B, m)
    
    utils.replace_inf(left)
    utils.replace_inf(right)
    npt.assert_almost_equal(left, right)