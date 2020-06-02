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
    _apply_include,
)
import pytest
import naive


def naive_rolling_window_dot_product(Q, T):
    window = len(Q)
    result = np.zeros(len(T) - window + 1)
    for i in range(len(result)):
        result[i] = np.dot(T[i : i + window], Q)
    return result


def naive_apply_include(D, include):
    tmp = []
    for i in range(include.shape[0]):
        tmp.append(D[i])

    for i in range(include.shape[0]):
        D[i] = D[include[i]]

    for i in range(include.shape[0]):
        if include[i] >= include.shape[0]:
            D[include[i]] = tmp[i]


test_data = [
    (np.array([[584, -11, 23, 79, 1001, 0, -19]], dtype=np.float64), 3),
    (np.random.uniform(-1000, 1000, [5, 20]).astype(np.float64), 5),
]

substitution_locations = [(slice(0, 0), 0, -1, slice(1, 3), [0, 3])]
substitution_values = [np.nan, np.inf]


def test_apply_include():
    D = np.random.uniform(-1000, 1000, [10, 20]).astype(np.float64)
    left_D = np.empty(D.shape)
    right_D = np.empty(D.shape)
    for width in range(D.shape[0]):
        for i in range(D.shape[0] - width):
            left_D[:, :] = D[:, :]
            right_D[:, :] = D[:, :]
            include = np.asarray(range(i, i + width + 1))

            naive_apply_include(D, include)
            _apply_include(D, include)

            npt.assert_almost_equal(left_D, right_D)


def test_multi_mass_seeded():
    np.random.seed(5)
    T = np.random.uniform(-1000, 1000, [3, 10]).astype(np.float64)
    m = 5

    trivial_idx = 2

    Q = T[:, trivial_idx : trivial_idx + m]

    left = naive.multi_mass(Q, T, m)

    M_T, Σ_T = core.compute_mean_std(T, m)
    right = _multi_mass(Q, T, m, M_T, Σ_T, M_T[:, trivial_idx], Σ_T[:, trivial_idx])

    npt.assert_almost_equal(left, right, decimal=6)


@pytest.mark.parametrize("T, m", test_data)
def test_multi_mass(T, m):
    trivial_idx = 2

    Q = T[:, trivial_idx : trivial_idx + m]

    left = naive.multi_mass(Q, T, m)

    M_T, Σ_T = core.compute_mean_std(T, m)
    right = _multi_mass(Q, T, m, M_T, Σ_T, M_T[:, trivial_idx], Σ_T[:, trivial_idx])

    npt.assert_almost_equal(left, right, decimal=6)


@pytest.mark.parametrize("T, m", test_data)
def test_get_first_mstump_profile(T, m):
    excl_zone = int(np.ceil(m / 4))
    start = 0

    left_P, left_I = naive.mstump(T, m, excl_zone)
    left_P = left_P[start, :]
    left_I = left_I[start, :]

    M_T, Σ_T = core.compute_mean_std(T, m)
    right_P, right_I = _get_first_mstump_profile(
        start, T, T, m, excl_zone, M_T, Σ_T, M_T, Σ_T
    )

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

    zone = int(np.ceil(m / 4))

    left = naive.stamp(T[0], m, exclusion_zone=zone)
    left_P = left[np.newaxis, :, 0].T
    left_I = left[np.newaxis, :, 1].T

    right_P, right_I = naive.mstump(T, m, zone)

    npt.assert_almost_equal(left_P, right_P)
    npt.assert_almost_equal(left_I, right_I)


@pytest.mark.parametrize("T, m", test_data)
def test_mstump(T, m):
    excl_zone = int(np.ceil(m / 4))

    left_P, left_I = naive.mstump(T, m, excl_zone)
    right_P, right_I = mstump(T, m)

    npt.assert_almost_equal(left_P, right_P)
    npt.assert_almost_equal(left_I, right_I)


@pytest.mark.parametrize("T, m", test_data)
def test_mstump_include(T, m):
    for width in range(T.shape[0]):
        for i in range(T.shape[0] - width):
            include = np.asarray(range(i, i + width + 1))
            excl_zone = int(np.ceil(m / 4))

            left_P, left_I = naive.mstump(T, m, excl_zone, include)
            right_P, right_I = mstump(T, m, include)

            npt.assert_almost_equal(left_P, right_P)
            npt.assert_almost_equal(left_I, right_I)


@pytest.mark.parametrize("T, m", test_data)
def test_mstump_discords(T, m):
    excl_zone = int(np.ceil(m / 4))

    left_P, left_I = naive.mstump(T, m, excl_zone, discords=True)
    right_P, right_I = mstump(T, m, discords=True)

    npt.assert_almost_equal(left_P, right_P)
    npt.assert_almost_equal(left_I, right_I)


@pytest.mark.parametrize("T, m", test_data)
def test_mstump_include_discords(T, m):
    for width in range(T.shape[0]):
        for i in range(T.shape[0] - width):
            include = np.asarray(range(i, i + width + 1))

            excl_zone = int(np.ceil(m / 4))

            left_P, left_I = naive.mstump(T, m, excl_zone, include, discords=True)
            right_P, right_I = mstump(T, m, include, discords=True)

            npt.assert_almost_equal(left_P, right_P)
            npt.assert_almost_equal(left_I, right_I)


@pytest.mark.parametrize("T, m", test_data)
def test_mstump_wrapper(T, m):
    excl_zone = int(np.ceil(m / 4))

    left_P, left_I = naive.mstump(T, m, excl_zone)
    right_P, right_I = mstump(T, m)

    npt.assert_almost_equal(left_P, right_P)
    npt.assert_almost_equal(left_I, right_I)

    df = pd.DataFrame(T.T)
    right_P, right_I = mstump(df, m)

    npt.assert_almost_equal(left_P, right_P)
    npt.assert_almost_equal(left_I, right_I)


@pytest.mark.parametrize("T, m", test_data)
def test_mstump_wrapper_include(T, m):
    for width in range(T.shape[0]):
        for i in range(T.shape[0] - width):
            include = np.asarray(range(i, i + width + 1))

            excl_zone = int(np.ceil(m / 4))

            left_P, left_I = naive.mstump(T, m, excl_zone, include)
            right_P, right_I = mstump(T, m, include)

            npt.assert_almost_equal(left_P, right_P)
            npt.assert_almost_equal(left_I, right_I)

            df = pd.DataFrame(T.T)
            right_P, right_I = mstump(df, m, include)

            npt.assert_almost_equal(left_P, right_P)
            npt.assert_almost_equal(left_I, right_I)


def test_constant_subsequence_self_join():
    T_A = np.concatenate((np.zeros(20, dtype=np.float64), np.ones(5, dtype=np.float64)))
    T = np.array([T_A, T_A, np.random.rand(T_A.shape[0])])
    m = 3

    excl_zone = int(np.ceil(m / 4))

    left_P, left_I = naive.mstump(T, m, excl_zone)
    right_P, right_I = mstump(T, m)

    npt.assert_almost_equal(left_P, right_P)  # ignore indices


@pytest.mark.parametrize("T, m", test_data)
@pytest.mark.parametrize("substitute", substitution_values)
@pytest.mark.parametrize("substitution_locations", substitution_locations)
def test_mstump_nan_inf_self_join_first_dimension(
    T, m, substitute, substitution_locations
):
    excl_zone = int(np.ceil(m / 4))

    T_sub = T.copy()

    for substitution_location in substitution_locations:
        T_sub[:] = T[:]
        T_sub[0, substitution_location] = substitute

        left_P, left_I = naive.mstump(T_sub, m, excl_zone)
        right_P, right_I = mstump(T_sub, m)

        npt.assert_almost_equal(left_P, right_P)
        npt.assert_almost_equal(left_I, right_I)


@pytest.mark.parametrize("T, m", test_data)
@pytest.mark.parametrize("substitute", substitution_values)
@pytest.mark.parametrize("substitution_locations", substitution_locations)
def test_mstump_nan_self_join_all_dimensions(T, m, substitute, substitution_locations):
    excl_zone = int(np.ceil(m / 4))

    T_sub = T.copy()

    for substitution_location in substitution_locations:
        T_sub[:] = T[:]
        T_sub[:, substitution_location] = substitute

        left_P, left_I = naive.mstump(T_sub, m, excl_zone)
        right_P, right_I = mstump(T_sub, m)

        npt.assert_almost_equal(left_P, right_P)
        npt.assert_almost_equal(left_I, right_I)
