import numpy as np
import numpy.testing as npt
import pandas as pd
from stumpy import core
from stumpy import (
    mstump,
    _mstump,
    _multi_mass,
    _get_first_mstump_profile,
    _get_multi_QT,
)
import pytest
import utils


def naive_rolling_window_dot_product(Q, T):
    window = len(Q)
    result = np.zeros(len(T) - window + 1)
    for i in range(len(result)):
        result[i] = np.dot(T[i : i + window], Q)
    return result


test_data = [
    (np.array([[584, -11, 23, 79, 1001, 0, -19]], dtype=np.float64), 3),
    (np.random.uniform(-1000, 1000, [3, 10]).astype(np.float64), 5),
]


@pytest.mark.parametrize("T, m", test_data)
def test_multi_mass(T, m):
    trivial_idx = 2
    Q = T[:, trivial_idx : trivial_idx + m]

    left = utils.naive_multi_mass(Q, T, m)

    M_T, Σ_T = core.compute_mean_std(T, m)
    right = _multi_mass(Q, T, m, M_T, Σ_T)

    npt.assert_almost_equal(left, right)


@pytest.mark.parametrize("T, m", test_data)
def test_get_first_mstump_profile(T, m):
    excl_zone = int(np.ceil(m / 4))
    start = 0

    left_P, left_I = utils.naive_mstump(T, m, excl_zone)
    left_P = left_P[start, :]
    left_I = left_I[start, :]

    M_T, Σ_T = core.compute_mean_std(T, m)
    right_P, right_I = _get_first_mstump_profile(start, T, m, excl_zone, M_T, Σ_T)

    npt.assert_almost_equal(left_P, right_P)
    npt.assert_equal(left_I, right_I)


@pytest.mark.parametrize("T, m", test_data)
def test_get_multi_QT(T, m):
    start = 0
    Q = core.rolling_window(T, m)
    left_QT = np.empty((Q.shape[0], Q.shape[1]), dtype="float64")
    left_QT_first = np.empty((Q.shape[0], Q.shape[1]), dtype="float64")

    for dim in range(T.shape[0]):
        left_QT[dim] = naive_rolling_window_dot_product(
            T[dim, start : start + m], T[dim]
        )
        left_QT_first[dim] = naive_rolling_window_dot_product(T[dim, :m], T[dim])

    right_QT, right_QT_first = _get_multi_QT(start, T, m)

    npt.assert_almost_equal(left_QT, right_QT)
    npt.assert_almost_equal(left_QT_first, right_QT_first)


def test_naive_mstump():
    T = np.random.uniform(-1000, 1000, [1, 1000]).astype(np.float64)
    m = 20

    excl_zone = int(np.ceil(m / 4))

    left = np.array(
        [
            utils.naive_mass(
                Q, T[0], m, trivial_idx=i, ignore_trivial=True, excl_zone=excl_zone
            )
            for i, Q in enumerate(core.rolling_window(T[0], m))
        ],
        dtype=object,
    )
    left_P = left[np.newaxis, :, 0].T
    left_I = left[np.newaxis, :, 1].T

    right_P, right_I = utils.naive_mstump(T, m, excl_zone)

    npt.assert_almost_equal(left_P, right_P)
    npt.assert_almost_equal(left_I, right_I)


@pytest.mark.parametrize("T, m", test_data)
def test_mstump(T, m):
    excl_zone = int(np.ceil(m / 4))

    left_P, left_I = utils.naive_mstump(T, m, excl_zone)
    right_P, right_I = mstump(T, m)

    npt.assert_almost_equal(left_P, right_P)
    npt.assert_almost_equal(left_I, right_I)


@pytest.mark.parametrize("T, m", test_data)
def test_mstump_wrapper(T, m):
    excl_zone = int(np.ceil(m / 4))

    left_P, left_I = utils.naive_mstump(T, m, excl_zone)
    right_P, right_I = mstump(T, m)

    npt.assert_almost_equal(left_P, right_P)
    npt.assert_almost_equal(left_I, right_I)

    df = pd.DataFrame(T.T)
    right_P, right_I = mstump(df, m)

    npt.assert_almost_equal(left_P, right_P)
    npt.assert_almost_equal(left_I, right_I)


def test_constant_subsequence_self_join():
    T_A = np.concatenate((np.zeros(20, dtype=np.float64), np.ones(5, dtype=np.float64)))
    T = np.array([T_A, T_A, np.random.rand(T_A.shape[0])])
    m = 3

    excl_zone = int(np.ceil(m / 4))

    left_P, left_I = utils.naive_mstump(T, m, excl_zone)
    right_P, right_I = mstump(T, m)

    npt.assert_almost_equal(left_P, right_P)  # ignore indices
