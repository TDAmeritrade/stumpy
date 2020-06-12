import numpy as np
import numpy.testing as npt
import pandas as pd
from stumpy import stumpi, core
import pytest
import naive


def test_stumpi_self_join():
    m = 3
    zone = int(np.ceil(m / 4))

    seed = np.random.randint(100000)
    np.random.seed(seed)

    T = np.random.rand(30)
    stream = stumpi(T, m)
    for i in range(34):
        t = np.random.rand()
        stream.add(t)

    right_P = stream.P_
    right_I = stream.I_
    right_left_P = stream.left_P_
    right_left_I = stream.left_I_

    left = naive.stamp(stream.T_, m, exclusion_zone=zone)
    left_P = left[:, 0]
    left_I = left[:, 1]
    left_left_P = np.empty(left_P.shape)
    left_left_P[:] = np.inf
    left_left_I = left[:, 2]
    for i, j in enumerate(left_left_I):
        if j >= 0:
            D = core.mass(stream.T_[i : i + m], stream.T_[j : j + m])
            left_left_P[i] = D[0]

    naive.replace_inf(left_P)
    naive.replace_inf(left_left_P)
    naive.replace_inf(right_P)
    naive.replace_inf(right_left_P)

    npt.assert_almost_equal(left_P, right_P)
    npt.assert_almost_equal(left_I, right_I)
    npt.assert_almost_equal(left_left_P, right_left_P)
    npt.assert_almost_equal(left_left_I, right_left_I)

    np.random.seed(seed)
    T = np.random.rand(30)
    T = pd.Series(T)
    stream = stumpi(T, m)
    for i in range(34):
        t = np.random.rand()
        stream.add(t)

    right_P = stream.P_
    right_I = stream.I_
    right_left_P = stream.left_P_
    right_left_I = stream.left_I_

    naive.replace_inf(right_P)
    naive.replace_inf(right_left_P)

    npt.assert_almost_equal(left_P, right_P)
    npt.assert_almost_equal(left_I, right_I)
    npt.assert_almost_equal(left_left_P, right_left_P)
    npt.assert_almost_equal(left_left_I, right_left_I)


def test_stumpi_constant_subsequence_self_join():
    m = 3
    zone = int(np.ceil(m / 4))

    seed = np.random.randint(100000)
    np.random.seed(seed)

    T = np.concatenate((np.zeros(20, dtype=np.float64), np.ones(10, dtype=np.float64)))
    stream = stumpi(T, m)
    for i in range(34):
        t = np.random.rand()
        stream.add(t)

    right_P = stream.P_
    right_I = stream.I_

    left = naive.stamp(stream.T_, m, exclusion_zone=zone)
    left_P = left[:, 0]
    left_I = left[:, 1]

    naive.replace_inf(left_P)
    naive.replace_inf(right_P)

    npt.assert_almost_equal(left_P, right_P)
    npt.assert_almost_equal(left_I, right_I)

    np.random.seed(seed)
    T = np.concatenate((np.zeros(20, dtype=np.float64), np.ones(10, dtype=np.float64)))
    T = pd.Series(T)
    stream = stumpi(T, m)
    for i in range(34):
        t = np.random.rand()
        stream.add(t)

    right_P = stream.P_
    right_I = stream.I_

    naive.replace_inf(right_P)

    npt.assert_almost_equal(left_P, right_P)
    npt.assert_almost_equal(left_I, right_I)
