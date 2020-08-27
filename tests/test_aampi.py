import numpy as np
import numpy.testing as npt
import pandas as pd
from stumpy import aampi, core, config
import pytest
import naive

substitution_locations = [(slice(0, 0), 0, -1, slice(1, 3), [0, 3])]
substitution_values = [np.nan, np.inf]


def test_aampi_self_join():
    m = 3
    zone = int(np.ceil(m / 4))

    seed = np.random.randint(100000)
    np.random.seed(seed)

    n = 30
    T = np.random.rand(n)
    stream = aampi(T, m, egress=False)
    for i in range(34):
        t = np.random.rand()
        stream.update(t)

    right_P = stream.P_
    right_I = stream.I_
    right_left_P = stream.left_P_
    right_left_I = stream.left_I_

    left = naive.aamp(stream.T_, m)
    left_P = left[:, 0]
    left_I = left[:, 1]
    left_left_P = np.full(left_P.shape, np.inf)
    left_left_I = left[:, 2]
    for i, j in enumerate(left_left_I):
        if j >= 0:
            D = core.mass_absolute(stream.T_[i : i + m], stream.T_[j : j + m])
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
    n = 30
    T = np.random.rand(n)
    T = pd.Series(T)
    stream = aampi(T, m, egress=False)
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


def test_aampi_self_join_egress():
    m = 3
    zone = int(np.ceil(m / 4))

    seed = np.random.randint(100000)
    np.random.seed(seed)

    n = 30
    T = np.random.rand(n)

    left = naive.aampi_egress(T, m)
    left_P = left.P_.copy()
    left_I = left.I_
    left_left_P = left.left_P_.copy()
    left_left_I = left.left_I_

    stream = aampi(T, m, egress=True)

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

    left = naive.aampi_egress(T, m)
    left_P = left.P_.copy()
    left_I = left.I_

    stream = aampi(T, m, egress=True)

    right_P = stream.P_.copy()
    right_I = stream.I_

    naive.replace_inf(left_P)
    naive.replace_inf(right_P)

    npt.assert_almost_equal(left_P, right_P)
    npt.assert_almost_equal(left_I, right_I)

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
def test_aampi_init_nan_inf_self_join(substitute, substitution_locations):
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
        stream = aampi(T, m, egress=False)
        for i in range(34):
            t = np.random.rand()
            stream.update(t)

        right_P = stream.P_
        right_I = stream.I_

        stream.T_[substitution_location] = substitute
        left = naive.aamp(stream.T_, m)
        left_P = left[:, 0]
        left_I = left[:, 1]

        naive.replace_inf(left_P)
        naive.replace_inf(right_P)
        npt.assert_almost_equal(left_P, right_P)
        npt.assert_almost_equal(left_I, right_I)

        np.random.seed(seed)
        n = 30
        T = np.random.rand(n)

        if substitution_location == -1:
            substitution_location = T.shape[0] - 1
        T[substitution_location] = substitute
        T = pd.Series(T)
        stream = aampi(T, m, egress=False)
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
def test_aampi_init_nan_inf_self_join_egress(substitute, substitution_locations):
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

        left = naive.aampi_egress(T, m)
        left_P = left.P_.copy()
        left_I = left.I_
        left_left_P = left.left_P_.copy()
        left_left_I = left.left_I_

        stream = aampi(T, m, egress=True)

        right_P = stream.P_.copy()
        right_I = stream.I_
        right_left_P = stream.left_P_.copy()
        right_left_I = stream.left_I_

        naive.replace_inf(left_P)
        naive.replace_inf(right_P)
        naive.replace_inf(left_left_P)
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
        n = 30
        T = np.random.rand(n)
        T = pd.Series(T)

        left = naive.aampi_egress(T, m)
        left_P = left.P_.copy()
        left_I = left.I_
        left_left_P = left.left_P_.copy()
        left_left_I = left.left_I_

        stream = aampi(T, m, egress=True)

        right_P = stream.P_.copy()
        right_I = stream.I_
        right_left_P = stream.left_P_.copy()
        right_left_I = stream.left_I_

        npt.assert_almost_equal(left_P, right_P)
        npt.assert_almost_equal(left_I, right_I)
        naive.replace_inf(left_left_P)
        naive.replace_inf(right_left_P)

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
def test_aampi_stream_nan_inf_self_join(substitute, substitution_locations):
    m = 3
    zone = int(np.ceil(m / 4))

    seed = np.random.randint(100000)

    for substitution_location in substitution_locations:
        np.random.seed(seed)
        n = 30
        T = np.random.rand(64)

        stream = aampi(T[:n], m, egress=False)
        if substitution_location == -1:
            substitution_location = T[n:].shape[0] - 1
        T[n:][substitution_location] = substitute
        for t in T[n:]:
            stream.update(t)

        right_P = stream.P_
        right_I = stream.I_

        stream.T_[n:][substitution_location] = substitute
        left = naive.aamp(stream.T_, m)
        left_P = left[:, 0]
        left_I = left[:, 1]

        naive.replace_inf(left_P)
        naive.replace_inf(right_P)

        npt.assert_almost_equal(left_P, right_P)
        npt.assert_almost_equal(left_I, right_I)

        np.random.seed(seed)
        T = np.random.rand(64)

        stream = aampi(pd.Series(T[:n]), m, egress=False)
        if substitution_location == -1:
            substitution_location = T[n:].shape[0] - 1
        T[n:][substitution_location] = substitute
        for t in T[n:]:
            stream.update(t)

        right_P = stream.P_
        right_I = stream.I_

        naive.replace_inf(right_P)

        npt.assert_almost_equal(left_P, right_P)
        npt.assert_almost_equal(left_I, right_I)


@pytest.mark.parametrize("substitute", substitution_values)
@pytest.mark.parametrize("substitution_locations", substitution_locations)
def test_aampi_stream_nan_inf_self_join_egress(substitute, substitution_locations):
    m = 3
    zone = int(np.ceil(m / 4))

    seed = np.random.randint(100000)

    for substitution_location in substitution_locations:
        np.random.seed(seed)
        n = 30
        T = np.random.rand(64)

        left = naive.aampi_egress(T[:n], m)
        left_P = left.P_.copy()
        left_I = left.I_
        left_left_P = left.left_P_.copy()
        left_left_I = left.left_I_

        stream = aampi(T[:n], m, egress=True)

        right_P = stream.P_.copy()
        right_I = stream.I_
        right_left_P = stream.left_P_.copy()
        right_left_I = stream.left_I_

        naive.replace_inf(left_P)
        naive.replace_inf(right_P)
        naive.replace_inf(left_left_P)
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

        np.random.seed(seed)
        T = np.random.rand(64)

        left = naive.aampi_egress(T[:n], m)
        left_P = left.P_.copy()
        left_I = left.I_
        left_left_P = left.left_P_.copy()
        left_left_I = left.left_I_

        stream = aampi(pd.Series(T[:n]), m, egress=True)

        right_P = stream.P_.copy()
        right_I = stream.I_
        right_left_P = stream.left_P_.copy()
        right_left_I = stream.left_I_

        naive.replace_inf(left_P)
        naive.replace_inf(right_P)
        naive.replace_inf(left_left_P)
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


def test_aampi_constant_subsequence_self_join():
    m = 3
    zone = int(np.ceil(m / 4))

    seed = np.random.randint(100000)
    np.random.seed(seed)

    T = np.concatenate((np.zeros(20, dtype=np.float64), np.ones(10, dtype=np.float64)))
    stream = aampi(T, m, egress=False)
    for i in range(34):
        t = np.random.rand()
        stream.update(t)

    right_P = stream.P_
    right_I = stream.I_

    left = naive.aamp(stream.T_, m)
    left_P = left[:, 0]
    left_I = left[:, 1]

    naive.replace_inf(left_P)
    naive.replace_inf(right_P)

    npt.assert_almost_equal(left_P, right_P)
    # npt.assert_almost_equal(left_I, right_I)

    np.random.seed(seed)
    T = np.concatenate((np.zeros(20, dtype=np.float64), np.ones(10, dtype=np.float64)))
    T = pd.Series(T)
    stream = aampi(T, m, egress=False)
    for i in range(34):
        t = np.random.rand()
        stream.update(t)

    right_P = stream.P_
    right_I = stream.I_

    naive.replace_inf(right_P)

    npt.assert_almost_equal(left_P, right_P)
    # npt.assert_almost_equal(left_I, right_I)


def test_aampi_constant_subsequence_self_join_egress():
    m = 3
    zone = int(np.ceil(m / 4))

    seed = np.random.randint(100000)
    np.random.seed(seed)

    T = np.concatenate((np.zeros(20, dtype=np.float64), np.ones(10, dtype=np.float64)))

    left = naive.aampi_egress(T, m)
    left_P = left.P_.copy()
    left_I = left.I_
    left_left_P = left.left_P_.copy()
    left_left_I = left.left_I_

    stream = aampi(T, m, egress=True)

    right_P = stream.P_.copy()
    right_I = stream.I_
    right_left_P = stream.left_P_.copy()
    right_left_I = stream.left_I_

    naive.replace_inf(left_P)
    naive.replace_inf(right_P)
    naive.replace_inf(left_left_P)
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

    left = naive.aampi_egress(T, m)
    left_P = left.P_.copy()
    left_I = left.I_
    left_left_P = left.left_P_.copy()
    left_left_I = left.left_I_

    stream = aampi(T, m, egress=True)

    right_P = stream.P_.copy()
    right_I = stream.I_
    right_left_P = stream.left_P_.copy()
    right_left_I = stream.left_I_

    naive.replace_inf(left_P)
    naive.replace_inf(right_P)
    naive.replace_inf(left_left_P)
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


def test_aampi_identical_subsequence_self_join():
    m = 3
    zone = int(np.ceil(m / 4))

    seed = np.random.randint(100000)
    np.random.seed(seed)

    identical = np.random.rand(8)
    T = np.random.rand(20)
    T[1 : 1 + identical.shape[0]] = identical
    T[11 : 11 + identical.shape[0]] = identical
    stream = aampi(T, m, egress=False)
    for i in range(34):
        t = np.random.rand()
        stream.update(t)

    right_P = stream.P_
    right_I = stream.I_

    left = naive.aamp(stream.T_, m)
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
    stream = aampi(T, m, egress=False)
    for i in range(34):
        t = np.random.rand()
        stream.update(t)

    right_P = stream.P_
    right_I = stream.I_

    naive.replace_inf(right_P)

    npt.assert_almost_equal(left_P, right_P, decimal=config.STUMPY_TEST_PRECISION)
    # npt.assert_almost_equal(left_I, right_I)


def test_aampi_identical_subsequence_self_join_egress():
    m = 3
    zone = int(np.ceil(m / 4))

    seed = np.random.randint(100000)
    np.random.seed(seed)

    identical = np.random.rand(8)
    T = np.random.rand(20)
    T[1 : 1 + identical.shape[0]] = identical
    T[11 : 11 + identical.shape[0]] = identical

    left = naive.aampi_egress(T, m)
    left_P = left.P_.copy()
    left_I = left.I_
    left_left_P = left.left_P_.copy()
    left_left_I = left.left_I_

    stream = aampi(T, m, egress=True)

    right_P = stream.P_.copy()
    right_I = stream.I_
    right_left_P = stream.left_P_.copy()
    right_left_I = stream.left_I_

    naive.replace_inf(left_P)
    naive.replace_inf(right_P)
    naive.replace_inf(left_left_P)
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

    left = naive.aampi_egress(T, m)
    left_P = left.P_.copy()
    left_I = left.I_
    left_left_P = left.left_P_.copy()
    left_left_I = left.left_I_

    stream = aampi(T, m, egress=True)

    right_P = stream.P_.copy()
    right_I = stream.I_
    right_left_P = stream.left_P_.copy()
    right_left_I = stream.left_I_

    naive.replace_inf(left_P)
    naive.replace_inf(right_P)
    naive.replace_inf(left_left_P)
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
