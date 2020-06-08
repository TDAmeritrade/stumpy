import numpy as np
import numpy.testing as npt
import pandas as pd
from stumpy import stumpi, core
import pytest
import naive


def test_stump_self_join():
    m = 3
    zone = int(np.ceil(m / 4))

    seed = np.random.randint(100000)
    np.random.seed(seed)

    T = np.random.rand(30)
    T, M_T, Σ_T = core.preprocess(T, m)
    mp = naive.stamp(T, m, exclusion_zone=zone)
    right_P = mp[:, 0].copy()
    right_I = mp[:, 1].copy()
    Q = T[-m:]
    QT = core.sliding_dot_product(Q, T)
    for i in range(34):
        t = np.random.rand()
        T, right_P, right_I, QT, M_T, Σ_T = stumpi(
            t, T, m, right_P, right_I, QT, M_T, Σ_T
        )

    left = naive.stamp(T, m, exclusion_zone=zone)
    left_P = left[:, 0]
    left_I = left[:, 1]

    naive.replace_inf(left_P)
    naive.replace_inf(right_P)

    npt.assert_almost_equal(left_P, right_P)
    npt.assert_almost_equal(left_I, right_I)

    np.random.seed(seed)
    T = np.random.rand(30)
    T, M_T, Σ_T = core.preprocess(T, m)
    T = pd.Series(T)
    mp = naive.stamp(T, m, exclusion_zone=zone)
    right_P = mp[:, 0].copy()
    right_I = mp[:, 1].copy()
    Q = T[-m:]
    QT = core.sliding_dot_product(Q, T)
    for i in range(34):
        t = np.random.rand()
        T, right_P, right_I, QT, M_T, Σ_T = stumpi(
            t, T, m, right_P, right_I, QT, M_T, Σ_T
        )

    naive.replace_inf(right_P)

    npt.assert_almost_equal(left_P, right_P)
    npt.assert_almost_equal(left_I, right_I)


def test_stump_constant_subsequence_self_join():
    m = 3
    zone = int(np.ceil(m / 4))

    seed = np.random.randint(100000)
    np.random.seed(seed)

    T = np.concatenate((np.zeros(20, dtype=np.float64), np.ones(10, dtype=np.float64)))
    T, M_T, Σ_T = core.preprocess(T, m)
    mp = naive.stamp(T, m, exclusion_zone=zone)
    right_P = mp[:, 0].copy()
    right_I = mp[:, 1].copy()
    Q = T[-m:]
    QT = core.sliding_dot_product(Q, T)
    for i in range(34):
        t = np.random.rand()
        T, right_P, right_I, QT, M_T, Σ_T = stumpi(
            t, T, m, right_P, right_I, QT, M_T, Σ_T
        )

    left = naive.stamp(T, m, exclusion_zone=zone)
    left_P = left[:, 0]
    left_I = left[:, 1]

    naive.replace_inf(left_P)
    naive.replace_inf(right_P)

    npt.assert_almost_equal(left_P, right_P)
    npt.assert_almost_equal(left_I, right_I)

    np.random.seed(seed)
    T = np.concatenate((np.zeros(20, dtype=np.float64), np.ones(10, dtype=np.float64)))
    T, M_T, Σ_T = core.preprocess(T, m)
    mp = naive.stamp(T, m, exclusion_zone=zone)
    right_P = mp[:, 0].copy()
    right_I = mp[:, 1].copy()
    Q = T[-m:]
    QT = core.sliding_dot_product(Q, T)
    T = pd.Series(T)
    for i in range(34):
        t = np.random.rand()
        T, right_P, right_I, QT, M_T, Σ_T = stumpi(
            t, T, m, right_P, right_I, QT, M_T, Σ_T
        )

    naive.replace_inf(right_P)

    npt.assert_almost_equal(left_P, right_P)
    npt.assert_almost_equal(left_I, right_I)
