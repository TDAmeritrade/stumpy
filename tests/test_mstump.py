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


def naive_mass(Q, T, m, trivial_idx, excl_zone):
    D = np.linalg.norm(
        utils.z_norm(core.rolling_window(T, m), 1) - utils.z_norm(Q), axis=1
    )
    start = max(0, trivial_idx - excl_zone)
    stop = min(T.shape[0] - Q.shape[0] + 1, trivial_idx + excl_zone)
    D[start : stop + 1] = np.inf

    return D


def naive_PI(D, trivial_idx):
    P = np.full((D.shape[0], D.shape[1]), np.inf)
    I = np.ones((D.shape[0], D.shape[1]), dtype="int64") * -1

    D = np.sort(D, axis=0)

    D_prime = np.zeros(D.shape[1])
    for i in range(D.shape[0]):
        D_prime = D_prime + D[i]
        D_prime_prime = D_prime / (i + 1)
        # Element-wise Min
        # col_idx = np.argmin([left_P[i, :], D_prime_prime], axis=0)
        # col_mask = col_idx > 0
        col_mask = P[i] > D_prime_prime
        P[i, col_mask] = D_prime_prime[col_mask]
        I[i, col_mask] = trivial_idx

    return P, I


def naive_rolling_window_dot_product(Q, T):
    window = len(Q)
    result = np.zeros(len(T) - window + 1)
    for i in range(len(result)):
        result[i] = np.dot(T[i : i + window], Q)
    return result


def naive_mstump(T, m):
    zone = int(np.ceil(m / 4))
    Q = core.rolling_window(T, m)
    D = np.empty((Q.shape[0], Q.shape[1]))
    P = np.full((Q.shape[0], Q.shape[1]), np.inf)
    I = np.ones((Q.shape[0], Q.shape[1]), dtype="int64") * -1

    # Left
    for i in range(Q.shape[1]):
        D[:] = 0.0
        for dim in range(T.shape[0]):
            D[dim] = naive_mass(Q[dim, i], T[dim], m, i, zone)

        P_i, I_i = naive_PI(D, i)

        for dim in range(T.shape[0]):
            col_mask = P[dim] > P_i[dim]
            P[dim, col_mask] = P_i[dim, col_mask]
            I[dim, col_mask] = I_i[dim, col_mask]

    return P, I


test_data = [
    (np.array([[584, -11, 23, 79, 1001, 0, -19]], dtype=np.float64), 3),
    (np.random.uniform(-1000, 1000, [3, 10]).astype(np.float64), 5),
]


@pytest.mark.parametrize("T, m", test_data)
def test_multi_mass(T, m):

    excl_zone = int(np.ceil(m / 4))
    trivial_idx = 2
    Q = core.rolling_window(T, m)
    # left
    D = np.empty((Q.shape[0], Q.shape[1]))

    for i in range(T.shape[0]):
        D[i] = naive_mass(Q[i, 0], T[i], m, trivial_idx, excl_zone)

    left_P, left_I = naive_PI(D, trivial_idx)

    # right
    M_T = np.empty((Q.shape[0], Q.shape[1]))
    Σ_T = np.empty((Q.shape[0], Q.shape[1]))
    for i in range(Q.shape[0]):
        M_T[i] = np.mean(Q[i], axis=1)
        Σ_T[i] = np.std(Q[i], axis=1)
    right_P, right_I = _multi_mass(Q[:, 0], T, m, M_T, Σ_T, trivial_idx, excl_zone)

    npt.assert_almost_equal(left_P, right_P)
    npt.assert_equal(left_I, right_I)


@pytest.mark.parametrize("T, m", test_data)
def test_get_first_mstump_profile(T, m):
    excl_zone = int(np.ceil(m / 4))
    start = 0
    Q = core.rolling_window(T, m)
    # left
    D = np.empty((Q.shape[0], Q.shape[1]))
    for i in range(T.shape[0]):
        D[i] = naive_mass(Q[i, 0], T[i], m, start, excl_zone)

    left_P, left_I = naive_PI(D, start)

    # right
    M_T = np.empty((Q.shape[0], Q.shape[1]))
    Σ_T = np.empty((Q.shape[0], Q.shape[1]))
    for i in range(Q.shape[0]):
        M_T[i] = np.mean(Q[i], axis=1)
        Σ_T[i] = np.std(Q[i], axis=1)
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


@pytest.mark.parametrize("T, m", test_data)
def test_mstump(T, m):
    left_P, left_I = naive_mstump(T, m)

    # Right
    d = T.shape[0]
    n = T.shape[1]
    k = n - m + 1
    excl_zone = int(np.ceil(m / 4))  # See Definition 3 and Figure 3

    M_T, Σ_T = core.compute_mean_std(T, m)
    μ_Q, σ_Q = core.compute_mean_std(T, m)

    P = np.empty((d, k), dtype="float64")
    D = np.zeros((d, k), dtype="float64")
    D_prime = np.zeros(k, dtype="float64")
    I = np.ones((d, k), dtype="int64") * -1

    start = 0
    stop = k

    P, I = _get_first_mstump_profile(start, T, m, excl_zone, M_T, Σ_T)

    QT, QT_first = _get_multi_QT(start, T, m)

    right_P, right_I = _mstump(
        T,
        m,
        P,
        I,
        D,
        D_prime,
        stop,
        excl_zone,
        M_T,
        Σ_T,
        QT,
        QT_first,
        μ_Q,
        σ_Q,
        k,
        start + 1,
    )

    npt.assert_almost_equal(left_P, right_P)
    npt.assert_almost_equal(left_I, right_I)


@pytest.mark.parametrize("T, m", test_data)
def test_mstump_wrapper(T, m):
    left_P, left_I = naive_mstump(T, m)
    right_P, right_I = mstump(T, m)

    npt.assert_almost_equal(left_P.T, right_P)
    npt.assert_almost_equal(left_I.T, right_I)

    df = pd.DataFrame(T.T)
    right_P, right_I = mstump(df, m)

    npt.assert_almost_equal(left_P.T, right_P)
    npt.assert_almost_equal(left_I.T, right_I)


def test_constant_subsequence_self_join():
    T_A = np.concatenate((np.zeros(20, dtype=np.float64), np.ones(5, dtype=np.float64)))
    T = np.array([T_A, T_A, np.random.rand(T_A.shape[0])])
    m = 3

    left_P, left_I = naive_mstump(T, m)
    right_P, right_I = mstump(T, m)

    npt.assert_almost_equal(left_P.T, right_P)  # ignore indices
