import naive
import numpy as np
import numpy.testing as npt
import pandas as pd
import pytest

from stumpy import aampi, config, core

substitution_locations = [(slice(0, 0), 0, -1, slice(1, 3), [0, 3])]
substitution_values = [np.nan, np.inf]


def test_aampi_int_input():
    with pytest.raises(TypeError):
        aampi(np.arange(10), 5)


def test_aampi_self_join():
    m = 3

    for p in [1.0, 2.0, 3.0]:
        seed = np.random.randint(100000)
        np.random.seed(seed)

        n = 30
        T = np.random.rand(n)
        stream = aampi(T, m, egress=False, p=p)
        for i in range(34):
            t = np.random.rand()
            stream.update(t)

        comp_P = stream.P_
        comp_I = stream.I_
        comp_left_P = stream.left_P_
        comp_left_I = stream.left_I_

        ref_mp = naive.aamp(stream.T_, m, p=p)
        ref_P = ref_mp[:, 0]
        ref_I = ref_mp[:, 1]
        ref_left_P = np.full(ref_P.shape, np.inf)
        ref_left_I = ref_mp[:, 2]
        for i, j in enumerate(ref_left_I):
            if j >= 0:
                ref_left_P[i] = np.linalg.norm(
                    stream.T_[i : i + m] - stream.T_[j : j + m], ord=p
                )

        naive.replace_inf(ref_P)
        naive.replace_inf(ref_left_P)
        naive.replace_inf(comp_P)
        naive.replace_inf(comp_left_P)

        npt.assert_almost_equal(ref_P, comp_P)
        npt.assert_almost_equal(ref_I, comp_I)
        npt.assert_almost_equal(ref_left_P, comp_left_P)
        npt.assert_almost_equal(ref_left_I, comp_left_I)

        np.random.seed(seed)
        n = 30
        T = np.random.rand(n)
        T = pd.Series(T)
        stream = aampi(T, m, egress=False, p=p)
        for i in range(34):
            t = np.random.rand()
            stream.update(t)

        comp_P = stream.P_
        comp_I = stream.I_
        comp_left_P = stream.left_P_
        comp_left_I = stream.left_I_

        naive.replace_inf(comp_P)
        naive.replace_inf(comp_left_P)

        npt.assert_almost_equal(ref_P, comp_P)
        npt.assert_almost_equal(ref_I, comp_I)
        npt.assert_almost_equal(ref_left_P, comp_left_P)
        npt.assert_almost_equal(ref_left_I, comp_left_I)


def test_aampi_self_join_egress():
    m = 3

    for p in [1.0, 2.0, 3.0]:
        seed = np.random.randint(100000)
        np.random.seed(seed)

        n = 30
        T = np.random.rand(n)

        ref_mp = naive.aampi_egress(T, m, p=p)
        ref_P = ref_mp.P_.copy()
        ref_I = ref_mp.I_
        ref_left_P = ref_mp.left_P_.copy()
        ref_left_I = ref_mp.left_I_

        stream = aampi(T, m, egress=True, p=p)

        comp_P = stream.P_.copy()
        comp_I = stream.I_
        comp_left_P = stream.left_P_.copy()
        comp_left_I = stream.left_I_

        naive.replace_inf(ref_P)
        naive.replace_inf(ref_left_P)
        naive.replace_inf(comp_P)
        naive.replace_inf(comp_left_P)

        npt.assert_almost_equal(ref_P, comp_P)
        npt.assert_almost_equal(ref_I, comp_I)
        npt.assert_almost_equal(ref_left_P, comp_left_P)
        npt.assert_almost_equal(ref_left_I, comp_left_I)

        for i in range(34):
            t = np.random.rand()

            ref_mp.update(t)
            stream.update(t)

            comp_P = stream.P_.copy()
            comp_I = stream.I_
            comp_left_P = stream.left_P_.copy()
            comp_left_I = stream.left_I_

            ref_P = ref_mp.P_.copy()
            ref_I = ref_mp.I_
            ref_left_P = ref_mp.left_P_.copy()
            ref_left_I = ref_mp.left_I_

            naive.replace_inf(ref_P)
            naive.replace_inf(ref_left_P)
            naive.replace_inf(comp_P)
            naive.replace_inf(comp_left_P)

            npt.assert_almost_equal(ref_P, comp_P)
            npt.assert_almost_equal(ref_I, comp_I)
            npt.assert_almost_equal(ref_left_P, comp_left_P)
            npt.assert_almost_equal(ref_left_I, comp_left_I)

        np.random.seed(seed)
        T = np.random.rand(n)
        T = pd.Series(T)

        ref_mp = naive.aampi_egress(T, m, p=p)
        ref_P = ref_mp.P_.copy()
        ref_I = ref_mp.I_

        stream = aampi(T, m, egress=True, p=p)

        comp_P = stream.P_.copy()
        comp_I = stream.I_

        naive.replace_inf(ref_P)
        naive.replace_inf(comp_P)

        npt.assert_almost_equal(ref_P, comp_P)
        npt.assert_almost_equal(ref_I, comp_I)

        for i in range(34):
            t = np.random.rand()

            ref_mp.update(t)
            stream.update(t)

            comp_P = stream.P_.copy()
            comp_I = stream.I_
            comp_left_P = stream.left_P_.copy()
            comp_left_I = stream.left_I_

            ref_P = ref_mp.P_.copy()
            ref_I = ref_mp.I_
            ref_left_P = ref_mp.left_P_.copy()
            ref_left_I = ref_mp.left_I_

            naive.replace_inf(ref_P)
            naive.replace_inf(ref_left_P)
            naive.replace_inf(comp_P)
            naive.replace_inf(comp_left_P)

            npt.assert_almost_equal(ref_P, comp_P)
            npt.assert_almost_equal(ref_I, comp_I)
            npt.assert_almost_equal(ref_left_P, comp_left_P)
            npt.assert_almost_equal(ref_left_I, comp_left_I)


@pytest.mark.parametrize("substitute", substitution_values)
@pytest.mark.parametrize("substitution_locations", substitution_locations)
def test_aampi_init_nan_inf_self_join(substitute, substitution_locations):
    m = 3

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

        comp_P = stream.P_
        comp_I = stream.I_

        stream.T_[substitution_location] = substitute
        ref_mp = naive.aamp(stream.T_, m)
        ref_P = ref_mp[:, 0]
        ref_I = ref_mp[:, 1]

        naive.replace_inf(ref_P)
        naive.replace_inf(comp_P)
        npt.assert_almost_equal(ref_P, comp_P)
        npt.assert_almost_equal(ref_I, comp_I)

        np.random.seed(seed)
        n = 30
        T = np.random.rand(n)

        if substitution_location == -1:  # pragma: no cover
            substitution_location = T.shape[0] - 1
        T[substitution_location] = substitute
        T = pd.Series(T)
        stream = aampi(T, m, egress=False)
        for i in range(34):
            t = np.random.rand()
            stream.update(t)

        comp_P = stream.P_
        comp_I = stream.I_

        naive.replace_inf(comp_P)

        npt.assert_almost_equal(ref_P, comp_P)
        npt.assert_almost_equal(ref_I, comp_I)


@pytest.mark.parametrize("substitute", substitution_values)
@pytest.mark.parametrize("substitution_locations", substitution_locations)
def test_aampi_init_nan_inf_self_join_egress(substitute, substitution_locations):
    m = 3

    seed = np.random.randint(100000)
    # seed = 58638

    for substitution_location in substitution_locations:
        np.random.seed(seed)
        n = 30
        T = np.random.rand(n)

        if substitution_location == -1:
            substitution_location = T.shape[0] - 1
        T[substitution_location] = substitute

        ref_mp = naive.aampi_egress(T, m)
        ref_P = ref_mp.P_.copy()
        ref_I = ref_mp.I_
        ref_left_P = ref_mp.left_P_.copy()
        ref_left_I = ref_mp.left_I_

        stream = aampi(T, m, egress=True)

        comp_P = stream.P_.copy()
        comp_I = stream.I_
        comp_left_P = stream.left_P_.copy()
        comp_left_I = stream.left_I_

        naive.replace_inf(ref_P)
        naive.replace_inf(comp_P)
        naive.replace_inf(ref_left_P)
        naive.replace_inf(comp_left_P)

        npt.assert_almost_equal(ref_P, comp_P)
        npt.assert_almost_equal(ref_I, comp_I)
        npt.assert_almost_equal(ref_left_P, comp_left_P)
        npt.assert_almost_equal(ref_left_I, comp_left_I)

        for i in range(34):
            t = np.random.rand()

            ref_mp.update(t)
            stream.update(t)

            comp_P = stream.P_.copy()
            comp_I = stream.I_
            comp_left_P = stream.left_P_.copy()
            comp_left_I = stream.left_I_

            ref_P = ref_mp.P_.copy()
            ref_I = ref_mp.I_
            ref_left_P = ref_mp.left_P_.copy()
            ref_left_I = ref_mp.left_I_

            naive.replace_inf(ref_P)
            naive.replace_inf(ref_left_P)
            naive.replace_inf(comp_P)
            naive.replace_inf(comp_left_P)

            npt.assert_almost_equal(ref_P, comp_P)
            npt.assert_almost_equal(ref_I, comp_I)
            npt.assert_almost_equal(ref_left_P, comp_left_P)
            npt.assert_almost_equal(ref_left_I, comp_left_I)

        np.random.seed(seed)
        n = 30
        T = np.random.rand(n)
        T = pd.Series(T)

        ref_mp = naive.aampi_egress(T, m)
        ref_P = ref_mp.P_.copy()
        ref_I = ref_mp.I_
        ref_left_P = ref_mp.left_P_.copy()
        ref_left_I = ref_mp.left_I_

        stream = aampi(T, m, egress=True)

        comp_P = stream.P_.copy()
        comp_I = stream.I_
        comp_left_P = stream.left_P_.copy()
        comp_left_I = stream.left_I_

        npt.assert_almost_equal(ref_P, comp_P)
        npt.assert_almost_equal(ref_I, comp_I)
        naive.replace_inf(ref_left_P)
        naive.replace_inf(comp_left_P)

        for i in range(34):
            t = np.random.rand()

            ref_mp.update(t)
            stream.update(t)

            comp_P = stream.P_.copy()
            comp_I = stream.I_
            comp_left_P = stream.left_P_.copy()
            comp_left_I = stream.left_I_

            ref_P = ref_mp.P_.copy()
            ref_I = ref_mp.I_
            ref_left_P = ref_mp.left_P_.copy()
            ref_left_I = ref_mp.left_I_

            naive.replace_inf(ref_P)
            naive.replace_inf(ref_left_P)
            naive.replace_inf(comp_P)
            naive.replace_inf(comp_left_P)

            npt.assert_almost_equal(ref_P, comp_P)
            npt.assert_almost_equal(ref_I, comp_I)
            npt.assert_almost_equal(ref_left_P, comp_left_P)
            npt.assert_almost_equal(ref_left_I, comp_left_I)


@pytest.mark.parametrize("substitute", substitution_values)
@pytest.mark.parametrize("substitution_locations", substitution_locations)
def test_aampi_stream_nan_inf_self_join(substitute, substitution_locations):
    m = 3

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

        comp_P = stream.P_
        comp_I = stream.I_

        stream.T_[n:][substitution_location] = substitute
        ref_mp = naive.aamp(stream.T_, m)
        ref_P = ref_mp[:, 0]
        ref_I = ref_mp[:, 1]

        naive.replace_inf(ref_P)
        naive.replace_inf(comp_P)

        npt.assert_almost_equal(ref_P, comp_P)
        npt.assert_almost_equal(ref_I, comp_I)

        np.random.seed(seed)
        T = np.random.rand(64)

        stream = aampi(pd.Series(T[:n]), m, egress=False)
        if substitution_location == -1:  # pragma: no cover
            substitution_location = T[n:].shape[0] - 1
        T[n:][substitution_location] = substitute
        for t in T[n:]:
            stream.update(t)

        comp_P = stream.P_
        comp_I = stream.I_

        naive.replace_inf(comp_P)

        npt.assert_almost_equal(ref_P, comp_P)
        npt.assert_almost_equal(ref_I, comp_I)


@pytest.mark.parametrize("substitute", substitution_values)
@pytest.mark.parametrize("substitution_locations", substitution_locations)
def test_aampi_stream_nan_inf_self_join_egress(substitute, substitution_locations):
    m = 3

    seed = np.random.randint(100000)

    for substitution_location in substitution_locations:
        np.random.seed(seed)
        n = 30
        T = np.random.rand(64)

        ref_mp = naive.aampi_egress(T[:n], m)
        ref_P = ref_mp.P_.copy()
        ref_I = ref_mp.I_
        ref_left_P = ref_mp.left_P_.copy()
        ref_left_I = ref_mp.left_I_

        stream = aampi(T[:n], m, egress=True)

        comp_P = stream.P_.copy()
        comp_I = stream.I_
        comp_left_P = stream.left_P_.copy()
        comp_left_I = stream.left_I_

        naive.replace_inf(ref_P)
        naive.replace_inf(comp_P)
        naive.replace_inf(ref_left_P)
        naive.replace_inf(comp_left_P)

        npt.assert_almost_equal(ref_P, comp_P)
        npt.assert_almost_equal(ref_I, comp_I)
        npt.assert_almost_equal(ref_left_P, comp_left_P)
        npt.assert_almost_equal(ref_left_I, comp_left_I)

        if substitution_location == -1:
            substitution_location = T[n:].shape[0] - 1
        T[n:][substitution_location] = substitute
        for t in T[n:]:
            ref_mp.update(t)
            stream.update(t)

            comp_P = stream.P_.copy()
            comp_I = stream.I_
            comp_left_P = stream.left_P_.copy()
            comp_left_I = stream.left_I_

            ref_P = ref_mp.P_.copy()
            ref_I = ref_mp.I_
            ref_left_P = ref_mp.left_P_.copy()
            ref_left_I = ref_mp.left_I_

            naive.replace_inf(ref_P)
            naive.replace_inf(ref_left_P)
            naive.replace_inf(comp_P)
            naive.replace_inf(comp_left_P)

            npt.assert_almost_equal(ref_P, comp_P)
            npt.assert_almost_equal(ref_I, comp_I)
            npt.assert_almost_equal(ref_left_P, comp_left_P)
            npt.assert_almost_equal(ref_left_I, comp_left_I)

        np.random.seed(seed)
        T = np.random.rand(64)

        ref_mp = naive.aampi_egress(T[:n], m)
        ref_P = ref_mp.P_.copy()
        ref_I = ref_mp.I_
        ref_left_P = ref_mp.left_P_.copy()
        ref_left_I = ref_mp.left_I_

        stream = aampi(pd.Series(T[:n]), m, egress=True)

        comp_P = stream.P_.copy()
        comp_I = stream.I_
        comp_left_P = stream.left_P_.copy()
        comp_left_I = stream.left_I_

        naive.replace_inf(ref_P)
        naive.replace_inf(comp_P)
        naive.replace_inf(ref_left_P)
        naive.replace_inf(comp_left_P)

        npt.assert_almost_equal(ref_P, comp_P)
        npt.assert_almost_equal(ref_I, comp_I)
        npt.assert_almost_equal(ref_left_P, comp_left_P)
        npt.assert_almost_equal(ref_left_I, comp_left_I)
        if substitution_location == -1:  # pragma: no cover
            substitution_location = T[n:].shape[0] - 1
        T[n:][substitution_location] = substitute
        for t in T[n:]:
            ref_mp.update(t)
            stream.update(t)

            comp_P = stream.P_.copy()
            comp_I = stream.I_
            comp_left_P = stream.left_P_.copy()
            comp_left_I = stream.left_I_

            ref_P = ref_mp.P_.copy()
            ref_I = ref_mp.I_
            ref_left_P = ref_mp.left_P_.copy()
            ref_left_I = ref_mp.left_I_

            naive.replace_inf(ref_P)
            naive.replace_inf(ref_left_P)
            naive.replace_inf(comp_P)
            naive.replace_inf(comp_left_P)

            npt.assert_almost_equal(ref_P, comp_P)
            npt.assert_almost_equal(ref_I, comp_I)
            npt.assert_almost_equal(ref_left_P, comp_left_P)
            npt.assert_almost_equal(ref_left_I, comp_left_I)


def test_aampi_constant_subsequence_self_join():
    m = 3

    seed = np.random.randint(100000)
    np.random.seed(seed)

    T = np.concatenate((np.zeros(20, dtype=np.float64), np.ones(10, dtype=np.float64)))
    stream = aampi(T, m, egress=False)
    for i in range(34):
        t = np.random.rand()
        stream.update(t)

    comp_P = stream.P_
    # comp_I = stream.I_

    ref_mp = naive.aamp(stream.T_, m)
    ref_P = ref_mp[:, 0]
    # ref_I = ref_mp[:, 1]

    naive.replace_inf(ref_P)
    naive.replace_inf(comp_P)

    npt.assert_almost_equal(ref_P, comp_P)
    # npt.assert_almost_equal(ref_I, comp_I)

    np.random.seed(seed)
    T = np.concatenate((np.zeros(20, dtype=np.float64), np.ones(10, dtype=np.float64)))
    T = pd.Series(T)
    stream = aampi(T, m, egress=False)
    for i in range(34):
        t = np.random.rand()
        stream.update(t)

    comp_P = stream.P_
    # comp_I = stream.I_

    naive.replace_inf(comp_P)

    npt.assert_almost_equal(ref_P, comp_P)
    # npt.assert_almost_equal(ref_I, comp_I)


def test_aampi_constant_subsequence_self_join_egress():
    m = 3

    seed = np.random.randint(100000)
    np.random.seed(seed)

    T = np.concatenate((np.zeros(20, dtype=np.float64), np.ones(10, dtype=np.float64)))

    ref_mp = naive.aampi_egress(T, m)
    ref_P = ref_mp.P_.copy()
    # ref_I = ref_mp.I_
    ref_left_P = ref_mp.left_P_.copy()
    # ref_left_I = ref_mp.left_I_

    stream = aampi(T, m, egress=True)

    comp_P = stream.P_.copy()
    # comp_I = stream.I_
    comp_left_P = stream.left_P_.copy()
    # comp_left_I = stream.left_I_

    naive.replace_inf(ref_P)
    naive.replace_inf(comp_P)
    naive.replace_inf(ref_left_P)
    naive.replace_inf(comp_left_P)

    npt.assert_almost_equal(ref_P, comp_P)
    # npt.assert_almost_equal(ref_I, comp_I)
    npt.assert_almost_equal(ref_left_P, comp_left_P)
    # npt.assert_almost_equal(ref_left_I, comp_left_I)

    for i in range(34):
        t = np.random.rand()
        ref_mp.update(t)
        stream.update(t)

        comp_P = stream.P_.copy()
        # comp_I = stream.I_
        comp_left_P = stream.left_P_.copy()
        # comp_left_I = stream.left_I_

        ref_P = ref_mp.P_.copy()
        # ref_I = ref_mp.I_
        ref_left_P = ref_mp.left_P_.copy()
        # ref_left_I = ref_mp.left_I_

        naive.replace_inf(ref_P)
        naive.replace_inf(ref_left_P)
        naive.replace_inf(comp_P)
        naive.replace_inf(comp_left_P)

        npt.assert_almost_equal(ref_P, comp_P)
        # npt.assert_almost_equal(ref_I, comp_I)
        npt.assert_almost_equal(ref_left_P, comp_left_P)
        # npt.assert_almost_equal(ref_left_I, comp_left_I)

    np.random.seed(seed)
    T = np.concatenate((np.zeros(20, dtype=np.float64), np.ones(10, dtype=np.float64)))
    T = pd.Series(T)

    ref_mp = naive.aampi_egress(T, m)
    ref_P = ref_mp.P_.copy()
    # ref_I = ref_mp.I_
    ref_left_P = ref_mp.left_P_.copy()
    # ref_left_I = ref_mp.left_I_

    stream = aampi(T, m, egress=True)

    comp_P = stream.P_.copy()
    # comp_I = stream.I_
    comp_left_P = stream.left_P_.copy()
    # comp_left_I = stream.left_I_

    naive.replace_inf(ref_P)
    naive.replace_inf(comp_P)
    naive.replace_inf(ref_left_P)
    naive.replace_inf(comp_left_P)

    npt.assert_almost_equal(ref_P, comp_P)
    # npt.assert_almost_equal(ref_I, comp_I)
    npt.assert_almost_equal(ref_left_P, comp_left_P)
    # npt.assert_almost_equal(ref_left_I, comp_left_I)

    for i in range(34):
        t = np.random.rand()
        ref_mp.update(t)
        stream.update(t)

        comp_P = stream.P_.copy()
        # comp_I = stream.I_
        comp_left_P = stream.left_P_.copy()
        # comp_left_I = stream.left_I_

        ref_P = ref_mp.P_.copy()
        # ref_I = ref_mp.I_
        ref_left_P = ref_mp.left_P_.copy()
        # ref_left_I = ref_mp.left_I_

        naive.replace_inf(ref_P)
        naive.replace_inf(ref_left_P)
        naive.replace_inf(comp_P)
        naive.replace_inf(comp_left_P)

        npt.assert_almost_equal(ref_P, comp_P)
        # npt.assert_almost_equal(ref_I, comp_I)
        npt.assert_almost_equal(ref_left_P, comp_left_P)
        # npt.assert_almost_equal(ref_left_I, comp_left_I)


def test_aampi_update_constant_subsequence_self_join():
    m = 3

    seed = np.random.randint(100000)
    np.random.seed(seed)
    T_full = np.random.rand(64)  # generate random data
    T_full[40:55] = 3  # add constant level interval

    T_stream = T_full[:10].copy()
    stream = aampi(T_stream, m, egress=False)

    for i in range(len(T_stream), len(T_full)):
        t = T_full[i]
        stream.update(t)

    comp_P = stream.P_

    ref_mp = naive.aamp(stream.T_, m)
    ref_P = ref_mp[:, 0]

    naive.replace_inf(ref_P)
    naive.replace_inf(comp_P)
    npt.assert_almost_equal(ref_P, comp_P)

    T_full = pd.Series(T_full)
    T_stream = T_full[:10].copy()
    stream = aampi(T_stream, m, egress=False)

    for i in range(len(T_stream), len(T_full)):
        t = T_full[i]
        stream.update(t)

    comp_P = stream.P_

    naive.replace_inf(comp_P)
    npt.assert_almost_equal(ref_P, comp_P)


def test_aampi_update_constant_subsequence_self_join_egress():
    m = 3

    seed = np.random.randint(100000)
    np.random.seed(seed)

    T_full = np.random.rand(64)  # generate random data
    T_full[40:55] = 3  # add constant level interval
    T_stream = T_full[:10].copy()

    ref_mp = naive.aampi_egress(T_stream, m)
    ref_P = ref_mp.P_.copy()
    ref_left_P = ref_mp.left_P_.copy()

    stream = aampi(T_stream, m, egress=True)

    for i in range(len(T_stream), len(T_full)):
        t = T_full[i]
        ref_mp.update(t)
        stream.update(t)

        comp_P = stream.P_.copy()
        comp_left_P = stream.left_P_.copy()

        ref_P = ref_mp.P_.copy()
        ref_left_P = ref_mp.left_P_.copy()

        naive.replace_inf(ref_P)
        naive.replace_inf(ref_left_P)
        naive.replace_inf(comp_P)
        naive.replace_inf(comp_left_P)

        npt.assert_almost_equal(ref_P, comp_P)
        npt.assert_almost_equal(ref_left_P, comp_left_P)

    T_full = pd.Series(T_full)
    T_stream = T_full[:10].copy()

    ref_mp = naive.aampi_egress(T_stream, m)
    ref_P = ref_mp.P_.copy()
    ref_left_P = ref_mp.left_P_.copy()

    stream = aampi(T_stream, m, egress=True)

    for i in range(len(T_stream), len(T_full)):
        t = T_full[i]
        ref_mp.update(t)
        stream.update(t)

        comp_P = stream.P_.copy()
        comp_left_P = stream.left_P_.copy()

        ref_P = ref_mp.P_.copy()
        ref_left_P = ref_mp.left_P_.copy()

        naive.replace_inf(ref_P)
        naive.replace_inf(ref_left_P)
        naive.replace_inf(comp_P)
        naive.replace_inf(comp_left_P)

        npt.assert_almost_equal(ref_P, comp_P)
        npt.assert_almost_equal(ref_left_P, comp_left_P)


def test_aampi_identical_subsequence_self_join():
    m = 3
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

    comp_P = stream.P_
    # comp_I = stream.I_

    ref_mp = naive.aamp(stream.T_, m)
    ref_P = ref_mp[:, 0]
    # ref_I = ref_mp[:, 1]

    naive.replace_inf(ref_P)
    naive.replace_inf(comp_P)

    npt.assert_almost_equal(ref_P, comp_P, decimal=config.STUMPY_TEST_PRECISION)
    # npt.assert_almost_equal(ref_I, comp_I)

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

    comp_P = stream.P_
    # comp_I = stream.I_

    naive.replace_inf(comp_P)

    npt.assert_almost_equal(ref_P, comp_P, decimal=config.STUMPY_TEST_PRECISION)
    # npt.assert_almost_equal(ref_I, comp_I)


def test_aampi_identical_subsequence_self_join_egress():
    m = 3

    seed = np.random.randint(100000)
    np.random.seed(seed)

    identical = np.random.rand(8)
    T = np.random.rand(20)
    T[1 : 1 + identical.shape[0]] = identical
    T[11 : 11 + identical.shape[0]] = identical

    ref_mp = naive.aampi_egress(T, m)
    ref_P = ref_mp.P_.copy()
    # ref_I = ref_mp.I_
    ref_left_P = ref_mp.left_P_.copy()
    # ref_left_I = ref_mp.left_I_

    stream = aampi(T, m, egress=True)

    comp_P = stream.P_.copy()
    # comp_I = stream.I_
    comp_left_P = stream.left_P_.copy()
    # comp_left_I = stream.left_I_

    naive.replace_inf(ref_P)
    naive.replace_inf(comp_P)
    naive.replace_inf(ref_left_P)
    naive.replace_inf(comp_left_P)

    npt.assert_almost_equal(ref_P, comp_P, decimal=config.STUMPY_TEST_PRECISION)
    # npt.assert_almost_equal(ref_I, comp_I)
    npt.assert_almost_equal(
        ref_left_P, comp_left_P, decimal=config.STUMPY_TEST_PRECISION
    )
    # npt.assert_almost_equal(ref_left_I, comp_left_I)

    for i in range(34):
        t = np.random.rand()
        ref_mp.update(t)
        stream.update(t)

        comp_P = stream.P_.copy()
        # comp_I = stream.I_
        comp_left_P = stream.left_P_.copy()
        # comp_left_I = stream.left_I_

        ref_P = ref_mp.P_.copy()
        # ref_I = ref_mp.I_
        ref_left_P = ref_mp.left_P_.copy()
        # ref_left_I = ref_mp.left_I_

        naive.replace_inf(ref_P)
        naive.replace_inf(ref_left_P)
        naive.replace_inf(comp_P)
        naive.replace_inf(comp_left_P)

        npt.assert_almost_equal(ref_P, comp_P, decimal=config.STUMPY_TEST_PRECISION)
        # npt.assert_almost_equal(ref_I, comp_I)
        npt.assert_almost_equal(
            ref_left_P, comp_left_P, decimal=config.STUMPY_TEST_PRECISION
        )
        # npt.assert_almost_equal(ref_left_I, comp_left_I)

    np.random.seed(seed)
    identical = np.random.rand(8)
    T = np.random.rand(20)
    T[1 : 1 + identical.shape[0]] = identical
    T[11 : 11 + identical.shape[0]] = identical
    T = pd.Series(T)

    ref_mp = naive.aampi_egress(T, m)
    ref_P = ref_mp.P_.copy()
    # ref_I = ref_mp.I_
    ref_left_P = ref_mp.left_P_.copy()
    # ref_left_I = ref_mp.left_I_

    stream = aampi(T, m, egress=True)

    comp_P = stream.P_.copy()
    # comp_I = stream.I_
    comp_left_P = stream.left_P_.copy()
    # comp_left_I = stream.left_I_

    naive.replace_inf(ref_P)
    naive.replace_inf(comp_P)
    naive.replace_inf(ref_left_P)
    naive.replace_inf(comp_left_P)

    npt.assert_almost_equal(ref_P, comp_P, decimal=config.STUMPY_TEST_PRECISION)
    # npt.assert_almost_equal(ref_I, comp_I)
    npt.assert_almost_equal(
        ref_left_P, comp_left_P, decimal=config.STUMPY_TEST_PRECISION
    )
    # npt.assert_almost_equal(ref_left_I, comp_left_I)

    for i in range(34):
        t = np.random.rand()
        ref_mp.update(t)
        stream.update(t)

        comp_P = stream.P_.copy()
        # comp_I = stream.I_
        comp_left_P = stream.left_P_.copy()
        # comp_left_I = stream.left_I_

        ref_P = ref_mp.P_.copy()
        # ref_I = ref_mp.I_
        ref_left_P = ref_mp.left_P_.copy()
        # ref_left_I = ref_mp.left_I_

        naive.replace_inf(ref_P)
        naive.replace_inf(ref_left_P)
        naive.replace_inf(comp_P)
        naive.replace_inf(comp_left_P)

        npt.assert_almost_equal(ref_P, comp_P, decimal=config.STUMPY_TEST_PRECISION)
        # npt.assert_almost_equal(ref_I, comp_I)
        npt.assert_almost_equal(
            ref_left_P, comp_left_P, decimal=config.STUMPY_TEST_PRECISION
        )
        # npt.assert_almost_equal(ref_left_I, comp_left_I)


def test_aampi_profile_index_match():
    T_full = np.random.rand(64)
    m = 3
    T_full_subseq = core.rolling_window(T_full, m)
    warm_start = 8

    T_stream = T_full[:warm_start].copy()
    stream = aampi(T_stream, m, egress=True)
    P = np.full(stream.P_.shape, np.inf)
    left_P = np.full(stream.left_P_.shape, np.inf)

    n = 0
    for i in range(len(T_stream), len(T_full)):
        t = T_full[i]
        stream.update(t)

        P[:] = np.inf
        idx = np.argwhere(stream.I_ >= 0).flatten()
        P[idx] = naive.distance(
            T_full_subseq[idx + n + 1], T_full_subseq[stream.I_[idx]], axis=1
        )

        left_P[:] = np.inf
        idx = np.argwhere(stream.left_I_ >= 0).flatten()
        left_P[idx] = naive.distance(
            T_full_subseq[idx + n + 1], T_full_subseq[stream.left_I_[idx]], axis=1
        )

        npt.assert_almost_equal(stream.P_, P)
        npt.assert_almost_equal(stream.left_P_, left_P)

        n += 1


def test_aampi_self_join_KNN():
    m = 3
    for k in range(2, 4):
        for p in [1.0, 2.0, 3.0]:
            seed = np.random.randint(100000)
            np.random.seed(seed)

            n = 30
            T = np.random.rand(n)
            stream = aampi(T, m, egress=False, p=p, k=k)
            for i in range(34):
                t = np.random.rand()
                stream.update(t)

            comp_P = stream.P_
            comp_I = stream.I_
            comp_left_P = stream.left_P_
            comp_left_I = stream.left_I_

            ref_mp = naive.aamp(stream.T_, m, p=p, k=k)
            ref_P = ref_mp[:, :k]
            ref_I = ref_mp[:, k : 2 * k]

            ref_left_I = ref_mp[:, 2 * k]
            ref_left_P = np.full_like(ref_left_I, np.inf, dtype=np.float64)
            for i, j in enumerate(ref_left_I):
                if j >= 0:
                    ref_left_P[i] = np.linalg.norm(
                        stream.T_[i : i + m] - stream.T_[j : j + m], ord=p
                    )

            naive.replace_inf(ref_P)
            naive.replace_inf(ref_left_P)
            naive.replace_inf(comp_P)
            naive.replace_inf(comp_left_P)

            npt.assert_almost_equal(ref_P, comp_P)
            npt.assert_almost_equal(ref_I, comp_I)
            npt.assert_almost_equal(ref_left_P, comp_left_P)
            npt.assert_almost_equal(ref_left_I, comp_left_I)

            np.random.seed(seed)
            n = 30
            T = np.random.rand(n)
            T = pd.Series(T)
            stream = aampi(T, m, egress=False, p=p, k=k)
            for i in range(34):
                t = np.random.rand()
                stream.update(t)

            comp_P = stream.P_
            comp_I = stream.I_
            comp_left_P = stream.left_P_
            comp_left_I = stream.left_I_

            naive.replace_inf(comp_P)
            naive.replace_inf(comp_left_P)

            npt.assert_almost_equal(ref_P, comp_P)
            npt.assert_almost_equal(ref_I, comp_I)
            npt.assert_almost_equal(ref_left_P, comp_left_P)
            npt.assert_almost_equal(ref_left_I, comp_left_I)


def test_aampi_self_join_egress_KNN():
    m = 3
    for k in range(2, 4):
        for p in [1.0, 2.0, 3.0]:
            seed = np.random.randint(100000)
            np.random.seed(seed)

            n = 30
            T = np.random.rand(n)

            ref_mp = naive.aampi_egress(T, m, p=p, k=k)
            ref_P = ref_mp.P_.copy()
            ref_I = ref_mp.I_
            ref_left_P = ref_mp.left_P_.copy()
            ref_left_I = ref_mp.left_I_

            stream = aampi(T, m, egress=True, p=p, k=k)

            comp_P = stream.P_.copy()
            comp_I = stream.I_
            comp_left_P = stream.left_P_.copy()
            comp_left_I = stream.left_I_

            naive.replace_inf(ref_P)
            naive.replace_inf(ref_left_P)
            naive.replace_inf(comp_P)
            naive.replace_inf(comp_left_P)

            npt.assert_almost_equal(ref_P, comp_P)
            npt.assert_almost_equal(ref_I, comp_I)
            npt.assert_almost_equal(ref_left_P, comp_left_P)
            npt.assert_almost_equal(ref_left_I, comp_left_I)

            for i in range(34):
                t = np.random.rand()

                ref_mp.update(t)
                stream.update(t)

                comp_P = stream.P_.copy()
                comp_I = stream.I_
                comp_left_P = stream.left_P_.copy()
                comp_left_I = stream.left_I_

                ref_P = ref_mp.P_.copy()
                ref_I = ref_mp.I_
                ref_left_P = ref_mp.left_P_.copy()
                ref_left_I = ref_mp.left_I_

                naive.replace_inf(ref_P)
                naive.replace_inf(ref_left_P)
                naive.replace_inf(comp_P)
                naive.replace_inf(comp_left_P)

                npt.assert_almost_equal(ref_P, comp_P)
                npt.assert_almost_equal(ref_I, comp_I)
                npt.assert_almost_equal(ref_left_P, comp_left_P)
                npt.assert_almost_equal(ref_left_I, comp_left_I)

            np.random.seed(seed)
            T = np.random.rand(n)
            T = pd.Series(T)

            ref_mp = naive.aampi_egress(T, m, p=p, k=k)
            ref_P = ref_mp.P_.copy()
            ref_I = ref_mp.I_

            stream = aampi(T, m, egress=True, p=p, k=k)

            comp_P = stream.P_.copy()
            comp_I = stream.I_

            naive.replace_inf(ref_P)
            naive.replace_inf(comp_P)

            npt.assert_almost_equal(ref_P, comp_P)
            npt.assert_almost_equal(ref_I, comp_I)

            for i in range(34):
                t = np.random.rand()

                ref_mp.update(t)
                stream.update(t)

                comp_P = stream.P_.copy()
                comp_I = stream.I_
                comp_left_P = stream.left_P_.copy()
                comp_left_I = stream.left_I_

                ref_P = ref_mp.P_.copy()
                ref_I = ref_mp.I_
                ref_left_P = ref_mp.left_P_.copy()
                ref_left_I = ref_mp.left_I_

                naive.replace_inf(ref_P)
                naive.replace_inf(ref_left_P)
                naive.replace_inf(comp_P)
                naive.replace_inf(comp_left_P)

                npt.assert_almost_equal(ref_P, comp_P)
                npt.assert_almost_equal(ref_I, comp_I)
                npt.assert_almost_equal(ref_left_P, comp_left_P)
                npt.assert_almost_equal(ref_left_I, comp_left_I)


def test_aampi_self_join_egress_passing_mp():
    m = 3

    for p in [1.0, 2.0, 3.0]:
        seed = np.random.randint(100000)
        np.random.seed(seed)

        n = 30
        T = np.random.rand(n)
        mp = naive.aamp(T, m, p=p)

        ref_mp = naive.aampi_egress(T, m, p=p, mp=mp)
        ref_P = ref_mp.P_.copy()
        ref_I = ref_mp.I_
        ref_left_P = ref_mp.left_P_.copy()
        ref_left_I = ref_mp.left_I_

        stream = aampi(T, m, egress=True, p=p, mp=mp)

        comp_P = stream.P_.copy()
        comp_I = stream.I_
        comp_left_P = stream.left_P_.copy()
        comp_left_I = stream.left_I_

        naive.replace_inf(ref_P)
        naive.replace_inf(ref_left_P)
        naive.replace_inf(comp_P)
        naive.replace_inf(comp_left_P)

        npt.assert_almost_equal(ref_P, comp_P)
        npt.assert_almost_equal(ref_I, comp_I)
        npt.assert_almost_equal(ref_left_P, comp_left_P)
        npt.assert_almost_equal(ref_left_I, comp_left_I)

        for i in range(34):
            t = np.random.rand()

            ref_mp.update(t)
            stream.update(t)

            comp_P = stream.P_.copy()
            comp_I = stream.I_
            comp_left_P = stream.left_P_.copy()
            comp_left_I = stream.left_I_

            ref_P = ref_mp.P_.copy()
            ref_I = ref_mp.I_
            ref_left_P = ref_mp.left_P_.copy()
            ref_left_I = ref_mp.left_I_

            naive.replace_inf(ref_P)
            naive.replace_inf(ref_left_P)
            naive.replace_inf(comp_P)
            naive.replace_inf(comp_left_P)

            npt.assert_almost_equal(ref_P, comp_P)
            npt.assert_almost_equal(ref_I, comp_I)
            npt.assert_almost_equal(ref_left_P, comp_left_P)
            npt.assert_almost_equal(ref_left_I, comp_left_I)
