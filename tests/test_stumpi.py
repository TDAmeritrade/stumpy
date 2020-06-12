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

    right_P = stream.P
    right_I = stream.I

    left = naive.stamp(stream.T, m, exclusion_zone=zone)
    left_P = left[:, 0]
    left_I = left[:, 1]

    naive.replace_inf(left_P)
    naive.replace_inf(right_P)

    npt.assert_almost_equal(left_P, right_P)
    npt.assert_almost_equal(left_I, right_I)

    np.random.seed(seed)
    T = np.random.rand(30)
    T = pd.Series(T)
    stream = stumpi(T, m)
    for i in range(34):
        t = np.random.rand()
        stream.add(t)

    right_P = stream.P
    right_I = stream.I

    naive.replace_inf(right_P)

    npt.assert_almost_equal(left_P, right_P)
    npt.assert_almost_equal(left_I, right_I)


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

    right_P = stream.P
    right_I = stream.I

    left = naive.stamp(stream.T, m, exclusion_zone=zone)
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

    right_P = stream.P
    right_I = stream.I

    naive.replace_inf(right_P)

    npt.assert_almost_equal(left_P, right_P)
    npt.assert_almost_equal(left_I, right_I)
