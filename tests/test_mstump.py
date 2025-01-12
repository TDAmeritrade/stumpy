import functools

import naive
import numpy as np
import numpy.testing as npt
import pandas as pd
import polars as pl
import pytest

from stumpy import config, core, mdl, mstump, subspace
from stumpy.mstump import (
    _get_first_mstump_profile,
    _get_multi_QT,
    _multi_mass,
    multi_distance_profile,
)


def naive_rolling_window_dot_product(Q, T):
    window = len(Q)
    result = np.zeros(len(T) - window + 1)
    for i in range(len(result)):
        result[i] = np.dot(T[i : i + window], Q)
    return result


test_data = [
    (np.array([[584, -11, 23, 79, 1001, 0, -19]], dtype=np.float64), 3),
    (np.random.uniform(-1000, 1000, [5, 20]).astype(np.float64), 5),
]

substitution_locations = [(slice(0, 0), 0, -1, slice(1, 3), [0, 3])]
substitution_values = [np.nan, np.inf]


def test_multi_mass_seeded():
    np.random.seed(5)
    T = np.random.uniform(-1000, 1000, [3, 10]).astype(np.float64)
    m = 5

    trivial_idx = 2

    Q = T[:, trivial_idx : trivial_idx + m]

    ref = naive.multi_mass(Q, T, m)

    T_subseq_isconstant = core.rolling_isconstant(T, m)
    M_T, Σ_T = core.compute_mean_std(T, m)

    Q_subseq_isconstant = np.expand_dims(T_subseq_isconstant[:, trivial_idx], 1)

    comp = _multi_mass(
        Q,
        T,
        m,
        M_T,
        Σ_T,
        M_T[:, trivial_idx],
        Σ_T[:, trivial_idx],
        T_subseq_isconstant=T_subseq_isconstant,
        Q_subseq_isconstant=Q_subseq_isconstant,
        query_idx=trivial_idx,
    )

    npt.assert_almost_equal(ref, comp, decimal=config.STUMPY_TEST_PRECISION)


@pytest.mark.parametrize("T, m", test_data)
def test_multi_mass(T, m):
    trivial_idx = 2

    Q = T[:, trivial_idx : trivial_idx + m]

    ref = naive.multi_mass(Q, T, m)

    T_subseq_isconstant = core.rolling_isconstant(T, m)
    M_T, Σ_T = core.compute_mean_std(T, m)

    Q_subseq_isconstant = np.expand_dims(T_subseq_isconstant[:, trivial_idx], 1)

    comp = _multi_mass(
        Q,
        T,
        m,
        M_T,
        Σ_T,
        M_T[:, trivial_idx],
        Σ_T[:, trivial_idx],
        T_subseq_isconstant=T_subseq_isconstant,
        Q_subseq_isconstant=Q_subseq_isconstant,
        query_idx=trivial_idx,
    )

    npt.assert_almost_equal(ref, comp, decimal=config.STUMPY_TEST_PRECISION)


@pytest.mark.parametrize("T, m", test_data)
def test_multi_distance_profile(T, m):
    for query_idx in range(T.shape[0] - m + 1):
        ref_D = naive.multi_distance_profile(query_idx, T, m)

        M_T, Σ_T = core.compute_mean_std(T, m)
        comp_D = multi_distance_profile(query_idx, T, m)

        npt.assert_almost_equal(ref_D, comp_D)


@pytest.mark.parametrize("T, m", test_data)
def test_get_first_mstump_profile(T, m):
    excl_zone = int(np.ceil(m / 4))
    start = 0

    ref_P, ref_I = naive.mstump(T, m, excl_zone)
    ref_P = ref_P[:, start]
    ref_I = ref_I[:, start]

    T_subseq_isconstant = core.rolling_isconstant(T, m)
    M_T, Σ_T = core.compute_mean_std(T, m)
    comp_P, comp_I = _get_first_mstump_profile(
        start,
        T,
        T,
        m,
        excl_zone,
        M_T,
        Σ_T,
        M_T,
        Σ_T,
        T_subseq_isconstant,
        T_subseq_isconstant,
    )

    npt.assert_almost_equal(ref_P, comp_P)
    npt.assert_equal(ref_I, comp_I)


@pytest.mark.parametrize("T, m", test_data)
def test_get_multi_QT(T, m):
    start = 0
    Q = core.rolling_window(T, m)
    ref_QT = np.empty((Q.shape[0], Q.shape[1]), dtype="float64")
    ref_QT_first = np.empty((Q.shape[0], Q.shape[1]), dtype="float64")

    for dim in range(T.shape[0]):
        ref_QT[dim] = naive_rolling_window_dot_product(
            T[dim, start : start + m], T[dim]
        )
        ref_QT_first[dim] = naive_rolling_window_dot_product(T[dim, :m], T[dim])

    comp_QT, comp_QT_first = _get_multi_QT(start, T, m)

    npt.assert_almost_equal(ref_QT, comp_QT)
    npt.assert_almost_equal(ref_QT_first, comp_QT_first)


@pytest.mark.parametrize("T, m", test_data)
def test_subspace(T, m):
    motif_idx = 1
    nn_idx = 4

    for k in range(T.shape[0]):
        ref_S = naive.subspace(T, m, motif_idx, nn_idx, k)
        comp_S = subspace(T, m, motif_idx, nn_idx, k)
        npt.assert_almost_equal(ref_S, comp_S)


@pytest.mark.parametrize("T, m", test_data)
def test_subspace_include(T, m):
    motif_idx = 1
    nn_idx = 4
    for width in range(T.shape[0]):
        for i in range(T.shape[0] - width):
            include = np.asarray(range(i, i + width + 1))

            for k in range(T.shape[0]):
                ref_S = naive.subspace(T, m, motif_idx, nn_idx, k, include)
                comp_S = subspace(T, m, motif_idx, nn_idx, k, include)
                npt.assert_almost_equal(ref_S, comp_S)


@pytest.mark.parametrize("T, m", test_data)
def test_subspace_discords(T, m):
    discord_idx = 1
    nn_idx = 4

    for k in range(T.shape[0]):
        ref_S = naive.subspace(T, m, discord_idx, nn_idx, k, discords=True)
        comp_S = subspace(T, m, discord_idx, nn_idx, k, discords=True)
        npt.assert_almost_equal(ref_S, comp_S)


@pytest.mark.parametrize("T, m", test_data)
def test_subspace_include_discords(T, m):
    discord_idx = 1
    nn_idx = 4
    for width in range(T.shape[0]):
        for i in range(T.shape[0] - width):
            include = np.asarray(range(i, i + width + 1))

            for k in range(T.shape[0]):
                ref_S = naive.subspace(
                    T, m, discord_idx, nn_idx, k, include, discords=True
                )
                comp_S = subspace(T, m, discord_idx, nn_idx, k, include, discords=True)
                npt.assert_almost_equal(ref_S, comp_S)


@pytest.mark.parametrize("T, m", test_data)
def test_mdl(T, m):
    subseq_idx = np.full(T.shape[0], 1)
    nn_idx = np.full(T.shape[0], 4)

    ref_MDL, ref_S = naive.mdl(T, m, subseq_idx, nn_idx)
    comp_MDL, comp_S = mdl(T, m, subseq_idx, nn_idx)
    npt.assert_almost_equal(ref_MDL, comp_MDL)

    for ref, cmp in zip(ref_S, comp_S):
        npt.assert_almost_equal(ref, cmp)


def test_naive_mstump():
    T = np.random.uniform(-1000, 1000, [1, 1000]).astype(np.float64)
    m = 20

    zone = int(np.ceil(m / 4))

    ref_mp = naive.stump(T[0], m, exclusion_zone=zone, row_wise=True)
    ref_P = ref_mp[np.newaxis, :, 0]
    ref_I = ref_mp[np.newaxis, :, 1]

    comp_P, comp_I = naive.mstump(T, m, zone)

    npt.assert_almost_equal(ref_P, comp_P)
    npt.assert_almost_equal(ref_I, comp_I)


def test_mstump_int_input():
    with pytest.raises(TypeError):
        mstump(np.arange(20).reshape(2, 10), 5)


@pytest.mark.parametrize("T, m", test_data)
def test_mstump(T, m):
    excl_zone = int(np.ceil(m / 4))

    ref_P, ref_I = naive.mstump(T, m, excl_zone)
    comp_P, comp_I = mstump(T, m)

    npt.assert_almost_equal(ref_P, comp_P)
    npt.assert_almost_equal(ref_I, comp_I)


@pytest.mark.parametrize("T, m", test_data)
def test_mstump_include(T, m):
    for width in range(T.shape[0]):
        for i in range(T.shape[0] - width):
            include = np.asarray(range(i, i + width + 1))
            excl_zone = int(np.ceil(m / 4))

            ref_P, ref_I = naive.mstump(T, m, excl_zone, include)
            comp_P, comp_I = mstump(T, m, include)

            npt.assert_almost_equal(ref_P, comp_P)
            npt.assert_almost_equal(ref_I, comp_I)


@pytest.mark.parametrize("T, m", test_data)
def test_mstump_discords(T, m):
    excl_zone = int(np.ceil(m / 4))

    ref_P, ref_I = naive.mstump(T, m, excl_zone, discords=True)
    comp_P, comp_I = mstump(T, m, discords=True)

    npt.assert_almost_equal(ref_P, comp_P)
    npt.assert_almost_equal(ref_I, comp_I)


@pytest.mark.parametrize("T, m", test_data)
def test_mstump_include_discords(T, m):
    for width in range(T.shape[0]):
        for i in range(T.shape[0] - width):
            include = np.asarray(range(i, i + width + 1))

            excl_zone = int(np.ceil(m / 4))

            ref_P, ref_I = naive.mstump(T, m, excl_zone, include, discords=True)
            comp_P, comp_I = mstump(T, m, include, discords=True)

            npt.assert_almost_equal(ref_P, comp_P)
            npt.assert_almost_equal(ref_I, comp_I)


@pytest.mark.parametrize("T, m", test_data)
def test_mstump_wrapper(T, m):
    excl_zone = int(np.ceil(m / 4))

    ref_P, ref_I = naive.mstump(T, m, excl_zone)
    comp_P, comp_I = mstump(T, m)

    npt.assert_almost_equal(ref_P, comp_P)
    npt.assert_almost_equal(ref_I, comp_I)

    df = pd.DataFrame(T.T)
    comp_P, comp_I = mstump(df, m)

    npt.assert_almost_equal(ref_P, comp_P)
    npt.assert_almost_equal(ref_I, comp_I)

    df = pl.DataFrame(T.T)
    comp_P, comp_I = mstump(df, m)

    npt.assert_almost_equal(ref_P, comp_P)
    npt.assert_almost_equal(ref_I, comp_I)


@pytest.mark.parametrize("T, m", test_data)
def test_mstump_wrapper_include(T, m):
    for width in range(T.shape[0]):
        for i in range(T.shape[0] - width):
            include = np.asarray(range(i, i + width + 1))

            excl_zone = int(np.ceil(m / 4))

            ref_P, ref_I = naive.mstump(T, m, excl_zone, include)
            comp_P, comp_I = mstump(T, m, include)

            npt.assert_almost_equal(ref_P, comp_P)
            npt.assert_almost_equal(ref_I, comp_I)

            df = pd.DataFrame(T.T)
            comp_P, comp_I = mstump(df, m, include)

            npt.assert_almost_equal(ref_P, comp_P)
            npt.assert_almost_equal(ref_I, comp_I)


def test_constant_subsequence_self_join():
    T_A = np.concatenate((np.zeros(20, dtype=np.float64), np.ones(5, dtype=np.float64)))
    T = np.array([T_A, T_A, np.random.rand(T_A.shape[0])])
    m = 3

    excl_zone = int(np.ceil(m / 4))

    ref_P, ref_I = naive.mstump(T, m, excl_zone)
    comp_P, comp_I = mstump(T, m)

    npt.assert_almost_equal(ref_P, comp_P)  # ignore indices


def test_identical_subsequence_self_join():
    identical = np.random.rand(8)
    T_A = np.random.rand(20)
    T_A[1 : 1 + identical.shape[0]] = identical
    T_A[11 : 11 + identical.shape[0]] = identical
    T = np.array([T_A, T_A, np.random.rand(T_A.shape[0])])
    m = 3

    excl_zone = int(np.ceil(m / 4))

    ref_P, ref_I = naive.mstump(T, m, excl_zone)
    comp_P, comp_I = mstump(T, m)

    npt.assert_almost_equal(
        ref_P, comp_P, decimal=config.STUMPY_TEST_PRECISION
    )  # ignore indices


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

        ref_P, ref_I = naive.mstump(T_sub, m, excl_zone)
        comp_P, comp_I = mstump(T_sub, m)

        npt.assert_almost_equal(ref_P, comp_P)
        npt.assert_almost_equal(ref_I, comp_I)


@pytest.mark.parametrize("T, m", test_data)
@pytest.mark.parametrize("substitute", substitution_values)
@pytest.mark.parametrize("substitution_locations", substitution_locations)
def test_mstump_nan_self_join_all_dimensions(T, m, substitute, substitution_locations):
    excl_zone = int(np.ceil(m / 4))

    T_sub = T.copy()

    for substitution_location in substitution_locations:
        T_sub[:] = T[:]
        T_sub[:, substitution_location] = substitute

        ref_P, ref_I = naive.mstump(T_sub, m, excl_zone)
        comp_P, comp_I = mstump(T_sub, m)

        npt.assert_almost_equal(ref_P, comp_P)
        npt.assert_almost_equal(ref_I, comp_I)


def test_multi_mass_with_isconstant():
    d = 3
    n = 64
    m = 8

    # case 1: Q is not multi-subseq of T
    T = np.random.uniform(-1000, 1000, size=[d, n])
    T_subseq_isconstant = np.random.choice(
        [True, False], size=(d, n - m + 1), replace=True
    )

    Q = np.random.uniform(-1000, 1000, size=[d, m])
    Q_subseq_isconstant = np.random.choice([True, False], size=(d, 1), replace=True)

    ref = naive.multi_mass(
        Q,
        T,
        m,
        T_subseq_isconstant=T_subseq_isconstant,
        Q_subseq_isconstant=Q_subseq_isconstant,
    )

    T_subseq_isconstant = core.rolling_isconstant(T, m, T_subseq_isconstant)
    M_T, Σ_T = core.compute_mean_std(T, m)

    Q_subseq_isconstant = core.rolling_isconstant(Q, m, Q_subseq_isconstant)
    μ_Q, σ_Q = core.compute_mean_std(Q, m)

    comp = _multi_mass(
        Q,
        T,
        m,
        M_T,
        Σ_T,
        μ_Q,
        σ_Q,
        T_subseq_isconstant=T_subseq_isconstant,
        Q_subseq_isconstant=Q_subseq_isconstant,
    )

    npt.assert_almost_equal(ref, comp, decimal=config.STUMPY_TEST_PRECISION)

    # case 2: Q is a multi-subseq of T
    T = np.random.uniform(-1000, 1000, size=[d, n])
    T_subseq_isconstant = np.random.choice(
        [True, False], size=(d, n - m + 1), replace=True
    )

    query_idx = np.random.randint(0, n - m + 1)
    Q = T[:, query_idx : query_idx + m]
    Q_subseq_isconstant = np.expand_dims(T_subseq_isconstant[:, query_idx], 1)

    ref = naive.multi_mass(
        Q,
        T,
        m,
        T_subseq_isconstant=T_subseq_isconstant,
        Q_subseq_isconstant=Q_subseq_isconstant,
    )

    T_subseq_isconstant = core.rolling_isconstant(T, m, T_subseq_isconstant)
    M_T, Σ_T = core.compute_mean_std(T, m)

    Q_subseq_isconstant = core.rolling_isconstant(Q, m, Q_subseq_isconstant)
    μ_Q, σ_Q = core.compute_mean_std(Q, m)

    comp = _multi_mass(
        Q,
        T,
        m,
        M_T,
        Σ_T,
        μ_Q,
        σ_Q,
        T_subseq_isconstant=T_subseq_isconstant,
        Q_subseq_isconstant=Q_subseq_isconstant,
        query_idx=query_idx,
    )

    npt.assert_almost_equal(ref, comp, decimal=config.STUMPY_TEST_PRECISION)


def test_multi_distance_profile_with_isconstant_case1():
    # case1: `T_subseq_isconstant` is `np.ndarray`
    d = 3
    n = 64
    m = 8

    T = np.random.uniform(-1000, 1000, size=[d, n])
    T_subseq_isconstant = np.random.choice(
        [True, False], size=(d, n - m + 1), replace=True
    )

    for query_idx in range(n - m + 1):
        ref_D = naive.multi_distance_profile(
            query_idx, T, m, T_subseq_isconstant=T_subseq_isconstant
        )

        comp_D = multi_distance_profile(
            query_idx, T, m, T_subseq_isconstant=T_subseq_isconstant
        )

        npt.assert_almost_equal(ref_D, comp_D)


def test_multi_distance_profile_with_isconstant_case2():
    # case2: `T_subseq_isconstant` is `function`
    d = 3
    n = 64
    m = 8

    T = np.random.uniform(-1000, 1000, size=[d, n])
    T_subseq_isconstant = functools.partial(
        naive.isconstant_func_stddev_threshold, quantile_threshold=0.05
    )

    query_idx = np.random.randint(0, n - m + 1)

    ref_D = naive.multi_distance_profile(
        query_idx, T, m, T_subseq_isconstant=T_subseq_isconstant
    )

    comp_D = multi_distance_profile(
        query_idx, T, m, T_subseq_isconstant=T_subseq_isconstant
    )

    npt.assert_almost_equal(ref_D, comp_D)


def test_multi_distance_profile_with_isconstant_case3():
    # case3: `T_subseq_isconstant` is of type `list`
    d = 3
    n = 64
    m = 8

    T = np.random.uniform(-1000, 1000, size=[d, n])
    T_subseq_isconstant = [
        None,
        np.random.choice([True, False], n - m + 1, replace=True),
        functools.partial(
            naive.isconstant_func_stddev_threshold, quantile_threshold=0.05
        ),
    ]

    query_idx = np.random.randint(0, n - m + 1)

    ref_D = naive.multi_distance_profile(
        query_idx, T, m, T_subseq_isconstant=T_subseq_isconstant
    )
    comp_D = multi_distance_profile(
        query_idx, T, m, T_subseq_isconstant=T_subseq_isconstant
    )

    npt.assert_almost_equal(ref_D, comp_D)


def test_mstump_with_isconstant_case1():
    # case1: `T_subseq_isconstant` is a (partial) function
    d = 3
    n = 64
    m = 8

    T = np.random.uniform(-1000, 1000, size=[d, n])
    T_subseq_isconstant = functools.partial(
        naive.isconstant_func_stddev_threshold, quantile_threshold=0.05
    )

    excl_zone = int(np.ceil(m / 4))

    ref_P, ref_I = naive.mstump(
        T, m, excl_zone, T_subseq_isconstant=T_subseq_isconstant
    )
    comp_P, comp_I = mstump(T, m, T_subseq_isconstant=T_subseq_isconstant)

    npt.assert_almost_equal(ref_P, comp_P)
    npt.assert_almost_equal(ref_I, comp_I)


def test_mstump_with_isconstant_case2():
    # case2: `T_subseq_isconstant` is a list
    d = 3
    n = 64
    m = 8

    T = np.random.uniform(-1000, 1000, size=[d, n])
    T_subseq_isconstant = [
        None,
        np.random.choice([True, False], n - m + 1, replace=True),
        functools.partial(
            naive.isconstant_func_stddev_threshold, quantile_threshold=0.05
        ),
    ]

    excl_zone = int(np.ceil(m / 4))

    ref_P, ref_I = naive.mstump(
        T, m, excl_zone, T_subseq_isconstant=T_subseq_isconstant
    )
    comp_P, comp_I = mstump(T, m, T_subseq_isconstant=T_subseq_isconstant)

    npt.assert_almost_equal(ref_P, comp_P)
    npt.assert_almost_equal(ref_I, comp_I)
