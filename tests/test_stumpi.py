import numpy as np
import numpy.testing as npt
import pandas as pd
from stumpy import stumpi, core, config
import pytest
import naive

substitution_locations = [(slice(0, 0), 0, -1, slice(1, 3), [0, 3])]
substitution_values = [np.nan, np.inf]


def test_stumpi_self_join():
    m = 3
    zone = int(np.ceil(m / 4))

    seed = np.random.randint(100000)
    np.random.seed(seed)

    T = np.random.rand(30)
    stream = stumpi(T, m, egress=False)
    for i in range(34):
        t = np.random.rand()
        stream.update(t)

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
    stream = stumpi(T, m, egress=False)
    for i in range(34):
        t = np.random.rand()
        stream.update(t)

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


def test_stumpi_self_join_egress():
    m = 3
    zone = int(np.ceil(m / 4))

    seed = np.random.randint(100000)
    np.random.seed(seed)
    n = 30
    T = np.random.rand(n)

    left = naive.stumpi_egress(T, m)
    left_P = left.P_.copy()
    left_I = left.I_
    left_left_P = left.left_P_.copy()
    left_left_I = left.left_I_

    stream = stumpi(T, m, egress=True)

    right_P = stream.P_.copy()
    right_I = stream.I_
    right_left_P = stream.left_P_.copy()
    right_left_I = stream.left_I_

    naive.replace_inf(left_P)
    naive.replace_inf(left_left_P)
    naive.replace_inf(right_P)
    naive.replace_inf(right_left_P)

    npt.assert_almost_equal(left_P, right_P)
    npt.assert_almost_equal(left_I, right_I)
    npt.assert_almost_equal(left_left_P, right_left_P)
    npt.assert_almost_equal(left_left_I, right_left_I)

    for i in range(34):
        t = np.random.rand()
        left.update(t)
        stream.update(t)

        right_P = stream.P_.copy()
        right_I = stream.I_
        right_left_P = stream.left_P_.copy()
        right_left_I = stream.left_I_

        left_P = left.P_.copy()
        left_I = left.I_
        left_left_P = left.left_P_.copy()
        left_left_I = left.left_I_

        naive.replace_inf(left_P)
        naive.replace_inf(left_left_P)
        naive.replace_inf(right_P)
        naive.replace_inf(right_left_P)

        npt.assert_almost_equal(left_P, right_P)
        npt.assert_almost_equal(left_I, right_I)
        npt.assert_almost_equal(left_left_P, right_left_P)
        npt.assert_almost_equal(left_left_I, right_left_I)

    np.random.seed(seed)
    T = np.random.rand(n)
    T = pd.Series(T)

    left = naive.stumpi_egress(T, m)
    left_P = left.P_.copy()
    left_I = left.I_
    left_left_P = left.left_P_.copy()
    left_left_I = left.left_I_

    stream = stumpi(T, m, egress=True)

    right_P = stream.P_.copy()
    right_I = stream.I_
    right_left_P = stream.left_P_.copy()
    right_left_I = stream.left_I_

    naive.replace_inf(left_P)
    naive.replace_inf(left_left_P)
    naive.replace_inf(right_P)
    naive.replace_inf(right_left_P)

    npt.assert_almost_equal(left_P, right_P)
    npt.assert_almost_equal(left_I, right_I)
    npt.assert_almost_equal(left_left_P, right_left_P)
    npt.assert_almost_equal(left_left_I, right_left_I)

    for i in range(34):
        t = np.random.rand()
        t = np.random.rand()
        left.update(t)
        stream.update(t)

        right_P = stream.P_.copy()
        right_I = stream.I_
        right_left_P = stream.left_P_.copy()
        right_left_I = stream.left_I_

        left_P = left.P_.copy()
        left_I = left.I_
        left_left_P = left.left_P_.copy()
        left_left_I = left.left_I_

        naive.replace_inf(left_P)
        naive.replace_inf(left_left_P)
        naive.replace_inf(right_P)
        naive.replace_inf(right_left_P)

        npt.assert_almost_equal(left_P, right_P)
        npt.assert_almost_equal(left_I, right_I)
        npt.assert_almost_equal(left_left_P, right_left_P)
        npt.assert_almost_equal(left_left_I, right_left_I)


@pytest.mark.parametrize("substitute", substitution_values)
@pytest.mark.parametrize("substitution_locations", substitution_locations)
def test_stumpi_init_nan_inf_self_join(substitute, substitution_locations):
    m = 3
    zone = int(np.ceil(m / 4))

    seed = np.random.randint(100000)
    # seed = 58638

    for substitution_location in substitution_locations:
        np.random.seed(seed)
        T = np.random.rand(30)

        if substitution_location == -1:
            substitution_location = T.shape[0] - 1
        T[substitution_location] = substitute
        stream = stumpi(T, m, egress=False)
        for i in range(34):
            t = np.random.rand()
            stream.update(t)

        right_P = stream.P_
        right_I = stream.I_

        stream.T_[substitution_location] = substitute
        left = naive.stamp(stream.T_, m, exclusion_zone=zone)
        left_P = left[:, 0]
        left_I = left[:, 1]

        naive.replace_inf(left_P)
        naive.replace_inf(right_P)
        npt.assert_almost_equal(left_P, right_P)
        npt.assert_almost_equal(left_I, right_I)

        np.random.seed(seed)
        T = np.random.rand(30)

        if substitution_location == -1:
            substitution_location = T.shape[0] - 1
        T[substitution_location] = substitute
        T = pd.Series(T)
        stream = stumpi(T, m, egress=False)
        for i in range(34):
            t = np.random.rand()
            stream.update(t)

        right_P = stream.P_
        right_I = stream.I_

        naive.replace_inf(right_P)

        npt.assert_almost_equal(left_P, right_P)
        npt.assert_almost_equal(left_I, right_I)


@pytest.mark.parametrize("substitute", substitution_values)
@pytest.mark.parametrize("substitution_locations", substitution_locations)
def test_stumpi_init_nan_inf_self_join_egress(substitute, substitution_locations):
    m = 3
    zone = int(np.ceil(m / 4))

    seed = np.random.randint(100000)
    # seed = 58638

    for substitution_location in substitution_locations:
        np.random.seed(seed)
        n = 30
        T = np.random.rand(n)

        if substitution_location == -1:
            substitution_location = T.shape[0] - 1
        T[substitution_location] = substitute

        left = naive.stumpi_egress(T, m)
        left_P = left.P_.copy()
        left_I = left.I_
        left_left_P = left.left_P_.copy()
        left_left_I = left.left_I_

        stream = stumpi(T, m, egress=True)

        right_P = stream.P_.copy()
        right_I = stream.I_
        right_left_P = stream.left_P_.copy()
        right_left_I = stream.left_I_

        naive.replace_inf(left_P)
        naive.replace_inf(left_left_P)
        naive.replace_inf(right_P)
        naive.replace_inf(right_left_P)

        npt.assert_almost_equal(left_P, right_P)
        npt.assert_almost_equal(left_I, right_I)
        npt.assert_almost_equal(left_left_P, right_left_P)
        npt.assert_almost_equal(left_left_I, right_left_I)

        for i in range(34):
            t = np.random.rand()
            left.update(t)
            stream.update(t)

            right_P = stream.P_.copy()
            right_I = stream.I_
            right_left_P = stream.left_P_.copy()
            right_left_I = stream.left_I_

            left_P = left.P_.copy()
            left_I = left.I_
            left_left_P = left.left_P_.copy()
            left_left_I = left.left_I_

            naive.replace_inf(left_P)
            naive.replace_inf(left_left_P)
            naive.replace_inf(right_P)
            naive.replace_inf(right_left_P)

            npt.assert_almost_equal(left_P, right_P)
            npt.assert_almost_equal(left_I, right_I)
            npt.assert_almost_equal(left_left_P, right_left_P)
            npt.assert_almost_equal(left_left_I, right_left_I)

        np.random.seed(seed)
        T = np.random.rand(n)

        if substitution_location == -1:
            substitution_location = T.shape[0] - 1
        T[substitution_location] = substitute
        T = pd.Series(T)

        left = naive.stumpi_egress(T, m)
        left_P = left.P_.copy()
        left_I = left.I_
        left_left_P = left.left_P_.copy()
        left_left_I = left.left_I_

        stream = stumpi(T, m, egress=True)

        right_P = stream.P_.copy()
        right_I = stream.I_
        right_left_P = stream.left_P_.copy()
        right_left_I = stream.left_I_

        naive.replace_inf(left_P)
        naive.replace_inf(left_left_P)
        naive.replace_inf(right_P)
        naive.replace_inf(right_left_P)

        npt.assert_almost_equal(left_P, right_P)
        npt.assert_almost_equal(left_I, right_I)
        npt.assert_almost_equal(left_left_P, right_left_P)
        npt.assert_almost_equal(left_left_I, right_left_I)

        for i in range(34):
            t = np.random.rand()
            left.update(t)
            stream.update(t)

            right_P = stream.P_.copy()
            right_I = stream.I_
            right_left_P = stream.left_P_.copy()
            right_left_I = stream.left_I_

            left_P = left.P_.copy()
            left_I = left.I_
            left_left_P = left.left_P_.copy()
            left_left_I = left.left_I_

            naive.replace_inf(left_P)
            naive.replace_inf(left_left_P)
            naive.replace_inf(right_P)
            naive.replace_inf(right_left_P)

            npt.assert_almost_equal(left_P, right_P)
            npt.assert_almost_equal(left_I, right_I)
            npt.assert_almost_equal(left_left_P, right_left_P)
            npt.assert_almost_equal(left_left_I, right_left_I)


@pytest.mark.parametrize("substitute", substitution_values)
@pytest.mark.parametrize("substitution_locations", substitution_locations)
def test_stumpi_stream_nan_inf_self_join(substitute, substitution_locations):
    m = 3
    zone = int(np.ceil(m / 4))

    seed = np.random.randint(100000)

    for substitution_location in substitution_locations:
        np.random.seed(seed)
        T = np.random.rand(64)

        stream = stumpi(T[:30], m, egress=False)
        if substitution_location == -1:
            substitution_location = T[30:].shape[0] - 1
        T[30:][substitution_location] = substitute
        for t in T[30:]:
            stream.update(t)

        right_P = stream.P_
        right_I = stream.I_

        stream.T_[30:][substitution_location] = substitute
        left = naive.stamp(stream.T_, m, exclusion_zone=zone)
        left_P = left[:, 0]
        left_I = left[:, 1]

        naive.replace_inf(left_P)
        naive.replace_inf(right_P)

        npt.assert_almost_equal(left_P, right_P)
        npt.assert_almost_equal(left_I, right_I)

        np.random.seed(seed)
        T = np.random.rand(64)

        stream = stumpi(pd.Series(T[:30]), m, egress=False)
        if substitution_location == -1:
            substitution_location = T[30:].shape[0] - 1
        T[30:][substitution_location] = substitute
        for t in T[30:]:
            stream.update(t)

        right_P = stream.P_
        right_I = stream.I_

        naive.replace_inf(right_P)

        npt.assert_almost_equal(left_P, right_P)
        npt.assert_almost_equal(left_I, right_I)


@pytest.mark.parametrize("substitute", substitution_values)
@pytest.mark.parametrize("substitution_locations", substitution_locations)
def test_stumpi_stream_nan_inf_self_join_egress(substitute, substitution_locations):
    m = 3
    zone = int(np.ceil(m / 4))

    seed = np.random.randint(100000)

    for substitution_location in substitution_locations:
        np.random.seed(seed)
        T = np.random.rand(64)
        n = 30

        left = naive.stumpi_egress(T[:n], m)
        left_P = left.P_.copy()
        left_I = left.I_
        left_left_P = left.left_P_.copy()
        left_left_I = left.left_I_

        stream = stumpi(T[:n], m, egress=True)

        right_P = stream.P_.copy()
        right_I = stream.I_
        right_left_P = stream.left_P_.copy()
        right_left_I = stream.left_I_

        naive.replace_inf(left_P)
        naive.replace_inf(left_left_P)
        naive.replace_inf(right_P)
        naive.replace_inf(right_left_P)

        npt.assert_almost_equal(left_P, right_P)
        npt.assert_almost_equal(left_I, right_I)
        npt.assert_almost_equal(left_left_P, right_left_P)
        npt.assert_almost_equal(left_left_I, right_left_I)

        if substitution_location == -1:
            substitution_location = T[30:].shape[0] - 1
        T[n:][substitution_location] = substitute
        for t in T[n:]:
            left.update(t)
            stream.update(t)

            right_P = stream.P_.copy()
            right_I = stream.I_
            right_left_P = stream.left_P_.copy()
            right_left_I = stream.left_I_

            left_P = left.P_.copy()
            left_I = left.I_
            left_left_P = left.left_P_.copy()
            left_left_I = left.left_I_

            naive.replace_inf(left_P)
            naive.replace_inf(left_left_P)
            naive.replace_inf(right_P)
            naive.replace_inf(right_left_P)

            npt.assert_almost_equal(left_P, right_P)
            npt.assert_almost_equal(left_I, right_I)
            npt.assert_almost_equal(left_left_P, right_left_P)
            npt.assert_almost_equal(left_left_I, right_left_I)

        np.random.seed(seed)
        T = np.random.rand(64)

        left = naive.stumpi_egress(T[:n], m)
        left_P = left.P_.copy()
        left_I = left.I_
        left_left_P = left.left_P_.copy()
        left_left_I = left.left_I_

        stream = stumpi(T[:n], m, egress=True)

        right_P = stream.P_.copy()
        right_I = stream.I_
        right_left_P = stream.left_P_.copy()
        right_left_I = stream.left_I_

        naive.replace_inf(left_P)
        naive.replace_inf(left_left_P)
        naive.replace_inf(right_P)
        naive.replace_inf(right_left_P)

        npt.assert_almost_equal(left_P, right_P)
        npt.assert_almost_equal(left_I, right_I)
        npt.assert_almost_equal(left_left_P, right_left_P)
        npt.assert_almost_equal(left_left_I, right_left_I)

        if substitution_location == -1:
            substitution_location = T[n:].shape[0] - 1
        T[n:][substitution_location] = substitute
        for t in T[n:]:
            left.update(t)
            stream.update(t)

            right_P = stream.P_.copy()
            right_I = stream.I_
            right_left_P = stream.left_P_.copy()
            right_left_I = stream.left_I_

            left_P = left.P_.copy()
            left_I = left.I_
            left_left_P = left.left_P_.copy()
            left_left_I = left.left_I_

            naive.replace_inf(left_P)
            naive.replace_inf(left_left_P)
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
    stream = stumpi(T, m, egress=False)
    for i in range(34):
        t = np.random.rand()
        stream.update(t)

    right_P = stream.P_
    right_I = stream.I_

    left = naive.stamp(stream.T_, m, exclusion_zone=zone)
    left_P = left[:, 0]
    left_I = left[:, 1]

    naive.replace_inf(left_P)
    naive.replace_inf(right_P)

    npt.assert_almost_equal(left_P, right_P)
    # npt.assert_almost_equal(left_I, right_I)

    np.random.seed(seed)
    T = np.concatenate((np.zeros(20, dtype=np.float64), np.ones(10, dtype=np.float64)))
    T = pd.Series(T)
    stream = stumpi(T, m, egress=False)
    for i in range(34):
        t = np.random.rand()
        stream.update(t)

    right_P = stream.P_
    right_I = stream.I_

    naive.replace_inf(right_P)

    npt.assert_almost_equal(left_P, right_P)
    # npt.assert_almost_equal(left_I, right_I)


def test_stumpi_constant_subsequence_self_join_egress():
    m = 3
    zone = int(np.ceil(m / 4))

    seed = np.random.randint(100000)
    np.random.seed(seed)

    T = np.concatenate((np.zeros(20, dtype=np.float64), np.ones(10, dtype=np.float64)))

    left = naive.stumpi_egress(T, m)
    left_P = left.P_.copy()
    left_I = left.I_
    left_left_P = left.left_P_.copy()
    left_left_I = left.left_I_

    stream = stumpi(T, m, egress=True)

    right_P = stream.P_.copy()
    right_I = stream.I_
    right_left_P = stream.left_P_.copy()
    right_left_I = stream.left_I_

    naive.replace_inf(left_P)
    naive.replace_inf(left_left_P)
    naive.replace_inf(right_P)
    naive.replace_inf(right_left_P)

    npt.assert_almost_equal(left_P, right_P)
    # npt.assert_almost_equal(left_I, right_I)
    npt.assert_almost_equal(left_left_P, right_left_P)
    # npt.assert_almost_equal(left_left_I, right_left_I)

    for i in range(34):
        t = np.random.rand()
        left.update(t)
        stream.update(t)

        right_P = stream.P_.copy()
        right_I = stream.I_
        right_left_P = stream.left_P_.copy()
        right_left_I = stream.left_I_

        left_P = left.P_.copy()
        left_I = left.I_
        left_left_P = left.left_P_.copy()
        left_left_I = left.left_I_

        naive.replace_inf(left_P)
        naive.replace_inf(left_left_P)
        naive.replace_inf(right_P)
        naive.replace_inf(right_left_P)

        npt.assert_almost_equal(left_P, right_P)
        # npt.assert_almost_equal(left_I, right_I)
        npt.assert_almost_equal(left_left_P, right_left_P)
        # npt.assert_almost_equal(left_left_I, right_left_I)

    np.random.seed(seed)
    T = np.concatenate((np.zeros(20, dtype=np.float64), np.ones(10, dtype=np.float64)))
    T = pd.Series(T)

    left = naive.stumpi_egress(T, m)
    left_P = left.P_.copy()
    left_I = left.I_
    left_left_P = left.left_P_.copy()
    left_left_I = left.left_I_

    stream = stumpi(T, m, egress=True)

    right_P = stream.P_.copy()
    right_I = stream.I_
    right_left_P = stream.left_P_.copy()
    right_left_I = stream.left_I_

    naive.replace_inf(left_P)
    naive.replace_inf(left_left_P)
    naive.replace_inf(right_P)
    naive.replace_inf(right_left_P)

    npt.assert_almost_equal(left_P, right_P)
    # npt.assert_almost_equal(left_I, right_I)
    npt.assert_almost_equal(left_left_P, right_left_P)
    # npt.assert_almost_equal(left_left_I, right_left_I)

    for i in range(34):
        t = np.random.rand()
        left.update(t)
        stream.update(t)

        right_P = stream.P_.copy()
        right_I = stream.I_
        right_left_P = stream.left_P_.copy()
        right_left_I = stream.left_I_

        left_P = left.P_.copy()
        left_I = left.I_
        left_left_P = left.left_P_.copy()
        left_left_I = left.left_I_

        naive.replace_inf(left_P)
        naive.replace_inf(left_left_P)
        naive.replace_inf(right_P)
        naive.replace_inf(right_left_P)

        npt.assert_almost_equal(left_P, right_P)
        # npt.assert_almost_equal(left_I, right_I)
        npt.assert_almost_equal(left_left_P, right_left_P)
        # npt.assert_almost_equal(left_left_I, right_left_I)


def test_stumpi_identical_subsequence_self_join():
    m = 3
    zone = int(np.ceil(m / 4))

    seed = np.random.randint(100000)
    np.random.seed(seed)

    identical = np.random.rand(8)
    T = np.random.rand(20)
    T[1 : 1 + identical.shape[0]] = identical
    T[11 : 11 + identical.shape[0]] = identical
    stream = stumpi(T, m, egress=False)
    for i in range(34):
        t = np.random.rand()
        stream.update(t)

    right_P = stream.P_
    right_I = stream.I_

    left = naive.stamp(stream.T_, m, exclusion_zone=zone)
    left_P = left[:, 0]
    left_I = left[:, 1]

    naive.replace_inf(left_P)
    naive.replace_inf(right_P)

    npt.assert_almost_equal(left_P, right_P, decimal=config.STUMPY_TEST_PRECISION)
    # npt.assert_almost_equal(left_I, right_I)

    np.random.seed(seed)
    identical = np.random.rand(8)
    T = np.random.rand(20)
    T[1 : 1 + identical.shape[0]] = identical
    T[11 : 11 + identical.shape[0]] = identical
    T = pd.Series(T)
    stream = stumpi(T, m, egress=False)
    for i in range(34):
        t = np.random.rand()
        stream.update(t)

    right_P = stream.P_
    right_I = stream.I_

    naive.replace_inf(right_P)

    npt.assert_almost_equal(left_P, right_P, decimal=config.STUMPY_TEST_PRECISION)
    # npt.assert_almost_equal(left_I, right_I)


def test_stumpi_identical_subsequence_self_join_egress():
    m = 3
    zone = int(np.ceil(m / 4))

    seed = np.random.randint(100000)
    np.random.seed(seed)

    identical = np.random.rand(8)
    T = np.random.rand(20)
    T[1 : 1 + identical.shape[0]] = identical
    T[11 : 11 + identical.shape[0]] = identical

    left = naive.stumpi_egress(T, m)
    left_P = left.P_.copy()
    left_I = left.I_
    left_left_P = left.left_P_.copy()
    left_left_I = left.left_I_

    stream = stumpi(T, m, egress=True)

    right_P = stream.P_.copy()
    right_I = stream.I_
    right_left_P = stream.left_P_.copy()
    right_left_I = stream.left_I_

    naive.replace_inf(left_P)
    naive.replace_inf(left_left_P)
    naive.replace_inf(right_P)
    naive.replace_inf(right_left_P)

    npt.assert_almost_equal(left_P, right_P, decimal=config.STUMPY_TEST_PRECISION)
    # npt.assert_almost_equal(left_I, right_I)
    npt.assert_almost_equal(
        left_left_P, right_left_P, decimal=config.STUMPY_TEST_PRECISION
    )
    # npt.assert_almost_equal(left_left_I, right_left_I)

    for i in range(34):
        t = np.random.rand()
        left.update(t)
        stream.update(t)

        right_P = stream.P_.copy()
        right_I = stream.I_
        right_left_P = stream.left_P_.copy()
        right_left_I = stream.left_I_

        left_P = left.P_.copy()
        left_I = left.I_
        left_left_P = left.left_P_.copy()
        left_left_I = left.left_I_

        naive.replace_inf(left_P)
        naive.replace_inf(left_left_P)
        naive.replace_inf(right_P)
        naive.replace_inf(right_left_P)

        npt.assert_almost_equal(left_P, right_P, decimal=config.STUMPY_TEST_PRECISION)
        # npt.assert_almost_equal(left_I, right_I)
        npt.assert_almost_equal(
            left_left_P, right_left_P, decimal=config.STUMPY_TEST_PRECISION
        )
        # npt.assert_almost_equal(left_left_I, right_left_I)

    np.random.seed(seed)
    identical = np.random.rand(8)
    T = np.random.rand(20)
    T[1 : 1 + identical.shape[0]] = identical
    T[11 : 11 + identical.shape[0]] = identical
    T = pd.Series(T)
    left = naive.stumpi_egress(T, m)
    left_P = left.P_.copy()
    left_I = left.I_
    left_left_P = left.left_P_.copy()
    left_left_I = left.left_I_

    stream = stumpi(T, m, egress=True)

    right_P = stream.P_.copy()
    right_I = stream.I_
    right_left_P = stream.left_P_.copy()
    right_left_I = stream.left_I_

    naive.replace_inf(left_P)
    naive.replace_inf(left_left_P)
    naive.replace_inf(right_P)
    naive.replace_inf(right_left_P)

    npt.assert_almost_equal(left_P, right_P, decimal=config.STUMPY_TEST_PRECISION)
    # npt.assert_almost_equal(left_I, right_I)
    npt.assert_almost_equal(
        left_left_P, right_left_P, decimal=config.STUMPY_TEST_PRECISION
    )
    # npt.assert_almost_equal(left_left_I, right_left_I)

    for i in range(34):
        t = np.random.rand()
        left.update(t)
        stream.update(t)

        right_P = stream.P_.copy()
        right_I = stream.I_
        right_left_P = stream.left_P_.copy()
        right_left_I = stream.left_I_

        left_P = left.P_.copy()
        left_I = left.I_
        left_left_P = left.left_P_.copy()
        left_left_I = left.left_I_

        naive.replace_inf(left_P)
        naive.replace_inf(left_left_P)
        naive.replace_inf(right_P)
        naive.replace_inf(right_left_P)

        npt.assert_almost_equal(left_P, right_P, decimal=config.STUMPY_TEST_PRECISION)
        # npt.assert_almost_equal(left_I, right_I)
        npt.assert_almost_equal(
            left_left_P, right_left_P, decimal=config.STUMPY_TEST_PRECISION
        )
        # npt.assert_almost_equal(left_left_I, right_left_I)


def test_stumpi_profile_index_match():
    T_full = np.random.rand(64)
    m = 3
    warm_start = 8

    T_stream = T_full[:warm_start].copy()
    stream = stumpi(T_stream, m, egress=True)
    P = np.full(stream.P_.shape, np.inf)
    left_P = np.full(stream.left_P_.shape, np.inf)

    n = 0
    for i in range(len(T_stream), len(T_full)):
        t = T_full[i]
        stream.update(t)

        for j in range(stream.I_.shape[0]):
            I = stream.I_[j]
            left_I = stream.left_I_[j]

            if I < 0:
                P[j] = np.inf
            else:
                P[j] = naive.distance(
                    naive.z_norm(T_full[j + n + 1 : j + n + 1 + m]),
                    naive.z_norm(T_full[I : I + m]),
                )

            if left_I < 0:
                left_P[j] = np.inf
            else:
                left_P[j] = naive.distance(
                    naive.z_norm(T_full[j + n + 1 : j + n + 1 + m]),
                    naive.z_norm(T_full[left_I : left_I + m]),
                )

        npt.assert_almost_equal(stream.P_, P)
        npt.assert_almost_equal(stream.left_P_, left_P)

        n += 1
