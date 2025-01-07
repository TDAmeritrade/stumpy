import functools

import naive
import numpy as np
import numpy.testing as npt
import pandas as pd
import polars as pl
import pytest

from stumpy import config, stump

test_data = [
    (
        np.array([9, 8100, -60, 7], dtype=np.float64),
        np.array([584, -11, 23, 79, 1001, 0, -19], dtype=np.float64),
    ),
    (
        np.random.uniform(-1000, 1000, [8]).astype(np.float64),
        np.random.uniform(-1000, 1000, [64]).astype(np.float64),
    ),
]

substitution_locations = [(slice(0, 0), 0, -1, slice(1, 3), [0, 3])]
substitution_values = [np.nan, np.inf]


def test_stump_int_input():
    with pytest.raises(TypeError):
        stump(np.arange(10), 5)


@pytest.mark.parametrize("T_A, T_B", test_data)
def test_stump_self_join(T_A, T_B):
    m = 3
    zone = int(np.ceil(m / 4))
    ref_mp = naive.stump(T_B, m, exclusion_zone=zone)
    comp_mp = stump(T_B, m, ignore_trivial=True)
    naive.replace_inf(ref_mp)
    naive.replace_inf(comp_mp)
    npt.assert_almost_equal(ref_mp, comp_mp)

    comp_mp = stump(pd.Series(T_B), m, ignore_trivial=True)
    naive.replace_inf(comp_mp)
    npt.assert_almost_equal(ref_mp, comp_mp)

    comp_mp = stump(pl.Series(T_B), m, ignore_trivial=True)
    naive.replace_inf(comp_mp)
    npt.assert_almost_equal(ref_mp, comp_mp)


@pytest.mark.parametrize("T_A, T_B", test_data)
def test_stump_A_B_join(T_A, T_B):
    m = 3
    ref_mp = naive.stump(T_A, m, T_B=T_B)
    comp_mp = stump(T_A, m, T_B, ignore_trivial=False)
    naive.replace_inf(ref_mp)
    naive.replace_inf(comp_mp)
    npt.assert_almost_equal(ref_mp, comp_mp)

    comp_mp = stump(pd.Series(T_A), m, pd.Series(T_B), ignore_trivial=False)
    naive.replace_inf(comp_mp)
    npt.assert_almost_equal(ref_mp, comp_mp)

    comp_mp = stump(pl.Series(T_A), m, pl.Series(T_B), ignore_trivial=False)
    naive.replace_inf(comp_mp)
    npt.assert_almost_equal(ref_mp, comp_mp)


def test_stump_constant_subsequence_self_join():
    T_A = np.concatenate((np.zeros(20, dtype=np.float64), np.ones(5, dtype=np.float64)))
    m = 3
    zone = int(np.ceil(m / 4))
    ref_mp = naive.stump(T_A, m, exclusion_zone=zone)
    comp_mp = stump(T_A, m, ignore_trivial=True)
    naive.replace_inf(ref_mp)
    naive.replace_inf(comp_mp)
    npt.assert_almost_equal(ref_mp[:, 0], comp_mp[:, 0])  # ignore indices

    comp_mp = stump(pd.Series(T_A), m, ignore_trivial=True)
    naive.replace_inf(comp_mp)
    npt.assert_almost_equal(ref_mp[:, 0], comp_mp[:, 0])  # ignore indices


def test_stump_one_constant_subsequence_A_B_join():
    T_A = np.random.rand(20)
    T_B = np.concatenate((np.zeros(20, dtype=np.float64), np.ones(5, dtype=np.float64)))
    m = 3
    ref_mp = naive.stump(T_A, m, T_B=T_B, row_wise=True)
    comp_mp = stump(T_A, m, T_B, ignore_trivial=False)
    naive.replace_inf(ref_mp)
    naive.replace_inf(comp_mp)
    npt.assert_almost_equal(ref_mp[:, 0], comp_mp[:, 0])  # ignore indices

    comp_mp = stump(pd.Series(T_A), m, pd.Series(T_B), ignore_trivial=False)
    naive.replace_inf(comp_mp)
    npt.assert_almost_equal(ref_mp[:, 0], comp_mp[:, 0])  # ignore indices

    # Swap inputs
    ref_mp = naive.stump(T_B, m, T_B=T_A, row_wise=True)
    comp_mp = stump(T_B, m, T_A, ignore_trivial=False)
    naive.replace_inf(ref_mp)
    naive.replace_inf(comp_mp)
    npt.assert_almost_equal(ref_mp[:, 0], comp_mp[:, 0])  # ignore indices


def test_stump_two_constant_subsequences_A_B_join():
    T_A = np.concatenate(
        (np.zeros(10, dtype=np.float64), np.ones(10, dtype=np.float64))
    )
    T_B = np.concatenate((np.zeros(20, dtype=np.float64), np.ones(5, dtype=np.float64)))
    m = 3
    ref_mp = naive.stump(T_A, m, T_B=T_B, row_wise=True)
    comp_mp = stump(T_A, m, T_B, ignore_trivial=False)
    naive.replace_inf(ref_mp)
    naive.replace_inf(comp_mp)
    npt.assert_almost_equal(ref_mp[:, 0], comp_mp[:, 0])  # ignore indices

    comp_mp = stump(pd.Series(T_A), m, pd.Series(T_B), ignore_trivial=False)
    naive.replace_inf(comp_mp)
    npt.assert_almost_equal(ref_mp[:, 0], comp_mp[:, 0])  # ignore indices

    # Swap inputs
    ref_mp = naive.stump(T_B, m, T_B=T_A, row_wise=True)
    comp_mp = stump(T_B, m, T_A, ignore_trivial=False)
    naive.replace_inf(ref_mp)
    naive.replace_inf(comp_mp)
    npt.assert_almost_equal(ref_mp[:, 0], comp_mp[:, 0])  # ignore indices

    comp_mp = stump(pd.Series(T_B), m, pd.Series(T_A), ignore_trivial=False)
    naive.replace_inf(comp_mp)
    npt.assert_almost_equal(ref_mp[:, 0], comp_mp[:, 0])  # ignore indices


def test_stump_identical_subsequence_self_join():
    identical = np.random.rand(8)
    T_A = np.random.rand(20)
    T_A[1 : 1 + identical.shape[0]] = identical
    T_A[11 : 11 + identical.shape[0]] = identical
    m = 3
    zone = int(np.ceil(m / 4))
    ref_mp = naive.stump(T_A, m, exclusion_zone=zone, row_wise=True)
    comp_mp = stump(T_A, m, ignore_trivial=True)
    naive.replace_inf(ref_mp)
    naive.replace_inf(comp_mp)
    npt.assert_almost_equal(
        ref_mp[:, 0], comp_mp[:, 0], decimal=config.STUMPY_TEST_PRECISION
    )  # ignore indices

    comp_mp = stump(pd.Series(T_A), m, ignore_trivial=True)
    naive.replace_inf(comp_mp)
    npt.assert_almost_equal(
        ref_mp[:, 0], comp_mp[:, 0], decimal=config.STUMPY_TEST_PRECISION
    )  # ignore indices


def test_stump_identical_subsequence_A_B_join():
    identical = np.random.rand(8)
    T_A = np.random.rand(20)
    T_B = np.random.rand(20)
    T_A[1 : 1 + identical.shape[0]] = identical
    T_B[11 : 11 + identical.shape[0]] = identical
    m = 3
    ref_mp = naive.stump(T_A, m, T_B=T_B, row_wise=True)
    comp_mp = stump(T_A, m, T_B, ignore_trivial=False)
    naive.replace_inf(ref_mp)
    naive.replace_inf(comp_mp)
    npt.assert_almost_equal(
        ref_mp[:, 0], comp_mp[:, 0], decimal=config.STUMPY_TEST_PRECISION
    )  # ignore indices

    comp_mp = stump(pd.Series(T_A), m, pd.Series(T_B), ignore_trivial=False)
    naive.replace_inf(comp_mp)
    npt.assert_almost_equal(
        ref_mp[:, 0], comp_mp[:, 0], decimal=config.STUMPY_TEST_PRECISION
    )  # ignore indices

    # Swap inputs
    ref_mp = naive.stump(T_B, m, T_B=T_A, row_wise=True)
    comp_mp = stump(T_B, m, T_A, ignore_trivial=False)
    naive.replace_inf(ref_mp)
    naive.replace_inf(comp_mp)
    npt.assert_almost_equal(
        ref_mp[:, 0], comp_mp[:, 0], decimal=config.STUMPY_TEST_PRECISION
    )  # ignore indices


@pytest.mark.parametrize("T_A, T_B", test_data)
@pytest.mark.parametrize("substitute_B", substitution_values)
@pytest.mark.parametrize("substitution_locations", substitution_locations)
def test_stump_nan_inf_self_join(T_A, T_B, substitute_B, substitution_locations):
    m = 3

    T_B_sub = T_B.copy()

    for substitution_location_B in substitution_locations:
        T_B_sub[:] = T_B[:]
        T_B_sub[substitution_location_B] = substitute_B

        zone = int(np.ceil(m / 4))
        ref_mp = naive.stump(T_B_sub, m, exclusion_zone=zone, row_wise=True)
        comp_mp = stump(T_B_sub, m, ignore_trivial=True)
        naive.replace_inf(ref_mp)
        naive.replace_inf(comp_mp)
        npt.assert_almost_equal(ref_mp, comp_mp)

        comp_mp = stump(pd.Series(T_B_sub), m, ignore_trivial=True)
        naive.replace_inf(comp_mp)
        npt.assert_almost_equal(ref_mp, comp_mp)


@pytest.mark.parametrize("T_A, T_B", test_data)
@pytest.mark.parametrize("substitute_A", substitution_values)
@pytest.mark.parametrize("substitute_B", substitution_values)
@pytest.mark.parametrize("substitution_locations", substitution_locations)
def test_stump_nan_inf_A_B_join(
    T_A, T_B, substitute_A, substitute_B, substitution_locations
):
    m = 3

    T_A_sub = T_A.copy()
    T_B_sub = T_B.copy()

    for substitution_location_B in substitution_locations:
        for substitution_location_A in substitution_locations:
            T_A_sub[:] = T_A[:]
            T_B_sub[:] = T_B[:]
            T_A_sub[substitution_location_A] = substitute_A
            T_B_sub[substitution_location_B] = substitute_B

            ref_mp = naive.stump(T_A_sub, m, T_B=T_B_sub, row_wise=True)
            comp_mp = stump(T_A_sub, m, T_B_sub, ignore_trivial=False)
            naive.replace_inf(ref_mp)
            naive.replace_inf(comp_mp)
            npt.assert_almost_equal(ref_mp, comp_mp)

            comp_mp = stump(
                pd.Series(T_A_sub), m, pd.Series(T_B_sub), ignore_trivial=False
            )
            naive.replace_inf(comp_mp)
            npt.assert_almost_equal(ref_mp, comp_mp)


def test_stump_nan_zero_mean_self_join():
    T = np.array([-1, 0, 1, np.inf, 1, 0, -1])
    m = 3

    zone = int(np.ceil(m / 4))
    ref_mp = naive.stump(T, m, exclusion_zone=zone, row_wise=True)
    comp_mp = stump(T, m, ignore_trivial=True)

    naive.replace_inf(ref_mp)
    naive.replace_inf(comp_mp)
    npt.assert_almost_equal(ref_mp, comp_mp)


@pytest.mark.parametrize("T_A, T_B", test_data)
def test_stump_self_join_KNN(T_A, T_B):
    m = 3
    zone = int(np.ceil(m / 4))
    for k in range(2, 4):
        ref_mp = naive.stump(T_B, m, exclusion_zone=zone, k=k)
        comp_mp = stump(T_B, m, ignore_trivial=True, k=k)
        naive.replace_inf(ref_mp)
        naive.replace_inf(comp_mp)
        npt.assert_almost_equal(ref_mp, comp_mp)

        comp_mp = stump(pd.Series(T_B), m, ignore_trivial=True, k=k)
        naive.replace_inf(comp_mp)
        npt.assert_almost_equal(ref_mp, comp_mp)


@pytest.mark.parametrize("T_A, T_B", test_data)
def test_stump_A_B_join_KNN(T_A, T_B):
    m = 3
    for k in range(2, 4):
        ref_mp = naive.stump(T_A, m, T_B=T_B, k=k)
        comp_mp = stump(T_A, m, T_B, ignore_trivial=False, k=k)
        naive.replace_inf(ref_mp)
        naive.replace_inf(comp_mp)
        npt.assert_almost_equal(ref_mp, comp_mp)

        comp_mp = stump(pd.Series(T_A), m, pd.Series(T_B), ignore_trivial=False, k=k)
        naive.replace_inf(comp_mp)
        npt.assert_almost_equal(ref_mp, comp_mp)


@pytest.mark.parametrize("T_A, T_B", test_data)
def test_stump_self_join_custom_isconstant(T_A, T_B):
    m = 3
    zone = int(np.ceil(m / 4))
    isconstant_custom_func = functools.partial(
        naive.isconstant_func_stddev_threshold, quantile_threshold=0.05
    )

    # case 1: custom isconstant is a boolean array
    T_B_subseq_isconstant = naive.rolling_isconstant(T_B, m, isconstant_custom_func)
    ref_mp = naive.stump(
        T_B, m, exclusion_zone=zone, T_A_subseq_isconstant=T_B_subseq_isconstant
    )
    comp_mp = stump(
        T_B, m, ignore_trivial=True, T_A_subseq_isconstant=T_B_subseq_isconstant
    )
    naive.replace_inf(ref_mp)
    naive.replace_inf(comp_mp)
    npt.assert_almost_equal(ref_mp, comp_mp)

    # case 2: custom isconstant is func
    ref_mp = naive.stump(
        T_B, m, exclusion_zone=zone, T_A_subseq_isconstant=isconstant_custom_func
    )
    comp_mp = stump(
        T_B, m, ignore_trivial=True, T_A_subseq_isconstant=isconstant_custom_func
    )
    naive.replace_inf(ref_mp)
    naive.replace_inf(comp_mp)
    npt.assert_almost_equal(ref_mp, comp_mp)
