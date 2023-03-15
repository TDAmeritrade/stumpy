import naive
import numpy as np
import numpy.testing as npt
import pandas as pd
import pytest

from stumpy import aamp, config

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


def test_aamp_int_input():
    with pytest.raises(TypeError):
        aamp(np.arange(10), 5)


@pytest.mark.parametrize("T_A, T_B", test_data)
def test_aamp_self_join(T_A, T_B):
    m = 3
    for p in [1.0, 2.0, 3.0]:
        ref_mp = naive.aamp(T_B, m, p=p)
        comp_mp = aamp(T_B, m, p=p)
        naive.replace_inf(ref_mp)
        naive.replace_inf(comp_mp)
        npt.assert_almost_equal(ref_mp, comp_mp)

        comp_mp = aamp(pd.Series(T_B), m, p=p)
        naive.replace_inf(comp_mp)
        npt.assert_almost_equal(ref_mp, comp_mp)


@pytest.mark.parametrize("T_A, T_B", test_data)
def test_aamp_A_B_join(T_A, T_B):
    m = 3
    for p in [1.0, 2.0, 3.0]:
        ref_mp = naive.aamp(T_A, m, T_B=T_B, p=p)
        comp_mp = aamp(T_A, m, T_B, ignore_trivial=False, p=p)
        naive.replace_inf(ref_mp)
        naive.replace_inf(comp_mp)
        npt.assert_almost_equal(ref_mp, comp_mp)

        comp_mp = aamp(pd.Series(T_A), m, pd.Series(T_B), ignore_trivial=False, p=p)
        naive.replace_inf(comp_mp)
        npt.assert_almost_equal(ref_mp, comp_mp)


def test_aamp_constant_subsequence_self_join():
    T_A = np.concatenate((np.zeros(20, dtype=np.float64), np.ones(5, dtype=np.float64)))
    m = 3
    ref_mp = naive.aamp(T_A, m)
    comp_mp = aamp(T_A, m, ignore_trivial=True)
    naive.replace_inf(ref_mp)
    naive.replace_inf(comp_mp)
    npt.assert_almost_equal(ref_mp[:, 0], comp_mp[:, 0])  # ignore indices

    comp_mp = aamp(pd.Series(T_A), m, ignore_trivial=True)
    naive.replace_inf(comp_mp)
    npt.assert_almost_equal(ref_mp[:, 0], comp_mp[:, 0])  # ignore indices


def test_aamp_one_constant_subsequence_A_B_join():
    T_A = np.random.rand(20)
    T_B = np.concatenate((np.zeros(20, dtype=np.float64), np.ones(5, dtype=np.float64)))
    m = 3
    ref_mp = naive.aamp(T_A, m, T_B=T_B)
    comp_mp = aamp(T_A, m, T_B, ignore_trivial=False)
    naive.replace_inf(ref_mp)
    naive.replace_inf(comp_mp)
    npt.assert_almost_equal(ref_mp[:, 0], comp_mp[:, 0])  # ignore indices

    comp_mp = aamp(pd.Series(T_A), m, pd.Series(T_B), ignore_trivial=False)
    naive.replace_inf(comp_mp)
    npt.assert_almost_equal(ref_mp[:, 0], comp_mp[:, 0])  # ignore indices

    # Swap inputs
    ref_mp = naive.aamp(T_B, m, T_B=T_A)
    comp_mp = aamp(T_B, m, T_A, ignore_trivial=False)
    naive.replace_inf(ref_mp)
    naive.replace_inf(comp_mp)
    npt.assert_almost_equal(ref_mp[:, 0], comp_mp[:, 0])  # ignore indices


def test_aamp_two_constant_subsequences_A_B_join():
    T_A = np.concatenate(
        (np.zeros(10, dtype=np.float64), np.ones(10, dtype=np.float64))
    )
    T_B = np.concatenate((np.zeros(20, dtype=np.float64), np.ones(5, dtype=np.float64)))
    m = 3
    ref_mp = naive.aamp(T_A, m, T_B=T_B)
    comp_mp = aamp(T_A, m, T_B, ignore_trivial=False)
    naive.replace_inf(ref_mp)
    naive.replace_inf(comp_mp)
    npt.assert_almost_equal(ref_mp[:, 0], comp_mp[:, 0])  # ignore indices

    comp_mp = aamp(pd.Series(T_A), m, pd.Series(T_B), ignore_trivial=False)
    naive.replace_inf(comp_mp)
    npt.assert_almost_equal(ref_mp[:, 0], comp_mp[:, 0])  # ignore indices

    # Swap inputs
    ref_mp = naive.aamp(T_B, m, T_B=T_A)
    comp_mp = aamp(T_B, m, T_A, ignore_trivial=False)
    naive.replace_inf(ref_mp)
    naive.replace_inf(comp_mp)
    npt.assert_almost_equal(ref_mp[:, 0], comp_mp[:, 0])  # ignore indices

    comp_mp = aamp(pd.Series(T_B), m, pd.Series(T_A), ignore_trivial=False)
    naive.replace_inf(comp_mp)
    npt.assert_almost_equal(ref_mp[:, 0], comp_mp[:, 0])  # ignore indices


def test_aamp_identical_subsequence_self_join():
    identical = np.random.rand(8)
    T_A = np.random.rand(20)
    T_A[1 : 1 + identical.shape[0]] = identical
    T_A[11 : 11 + identical.shape[0]] = identical
    m = 3
    ref_mp = naive.aamp(T_A, m)
    comp_mp = aamp(T_A, m, ignore_trivial=True)
    naive.replace_inf(ref_mp)
    naive.replace_inf(comp_mp)
    npt.assert_almost_equal(
        ref_mp[:, 0], comp_mp[:, 0], decimal=config.STUMPY_TEST_PRECISION
    )  # ignore indices

    comp_mp = aamp(pd.Series(T_A), m, ignore_trivial=True)
    naive.replace_inf(comp_mp)
    npt.assert_almost_equal(
        ref_mp[:, 0], comp_mp[:, 0], decimal=config.STUMPY_TEST_PRECISION
    )  # ignore indices


def test_aamp_identical_subsequence_A_B_join():
    identical = np.random.rand(8)
    T_A = np.random.rand(20)
    T_B = np.random.rand(20)
    T_A[1 : 1 + identical.shape[0]] = identical
    T_B[11 : 11 + identical.shape[0]] = identical
    m = 3
    ref_mp = naive.aamp(T_A, m, T_B=T_B)
    comp_mp = aamp(T_A, m, T_B, ignore_trivial=False)
    naive.replace_inf(ref_mp)
    naive.replace_inf(comp_mp)
    npt.assert_almost_equal(
        ref_mp[:, 0], comp_mp[:, 0], config.STUMPY_TEST_PRECISION
    )  # ignore indices

    comp_mp = aamp(pd.Series(T_A), m, pd.Series(T_B), ignore_trivial=False)
    naive.replace_inf(comp_mp)
    npt.assert_almost_equal(
        ref_mp[:, 0], comp_mp[:, 0], config.STUMPY_TEST_PRECISION
    )  # ignore indices

    # Swap inputs
    ref_mp = naive.aamp(T_B, m, T_B=T_A)
    comp_mp = aamp(T_B, m, T_A, ignore_trivial=False)
    naive.replace_inf(ref_mp)
    naive.replace_inf(comp_mp)
    npt.assert_almost_equal(
        ref_mp[:, 0], comp_mp[:, 0], config.STUMPY_TEST_PRECISION
    )  # ignore indices


@pytest.mark.parametrize("T_A, T_B", test_data)
@pytest.mark.parametrize("substitute_B", substitution_values)
@pytest.mark.parametrize("substitution_locations", substitution_locations)
def test_aamp_nan_inf_self_join(T_A, T_B, substitute_B, substitution_locations):
    m = 3

    T_B_sub = T_B.copy()

    for substitution_location_B in substitution_locations:
        T_B_sub[:] = T_B[:]
        T_B_sub[substitution_location_B] = substitute_B

        ref_mp = naive.aamp(T_B_sub, m)
        comp_mp = aamp(T_B_sub, m, ignore_trivial=True)
        naive.replace_inf(ref_mp)
        naive.replace_inf(comp_mp)
        npt.assert_almost_equal(ref_mp, comp_mp)

        comp_mp = aamp(pd.Series(T_B_sub), m, ignore_trivial=True)
        naive.replace_inf(comp_mp)
        npt.assert_almost_equal(ref_mp, comp_mp)


@pytest.mark.parametrize("T_A, T_B", test_data)
@pytest.mark.parametrize("substitute_A", substitution_values)
@pytest.mark.parametrize("substitute_B", substitution_values)
@pytest.mark.parametrize("substitution_locations", substitution_locations)
def test_aamp_nan_inf_A_B_join(
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

            ref_mp = naive.aamp(T_A_sub, m, T_B=T_B_sub)
            comp_mp = aamp(T_A_sub, m, T_B_sub, ignore_trivial=False)
            naive.replace_inf(ref_mp)
            naive.replace_inf(comp_mp)
            npt.assert_almost_equal(ref_mp, comp_mp)

            comp_mp = aamp(
                pd.Series(T_A_sub), m, pd.Series(T_B_sub), ignore_trivial=False
            )
            naive.replace_inf(comp_mp)
            npt.assert_almost_equal(ref_mp, comp_mp)


def test_aamp_nan_zero_mean_self_join():
    T = np.array([-1, 0, 1, np.inf, 1, 0, -1])
    m = 3

    ref_mp = naive.aamp(T, m)
    comp_mp = aamp(T, m, ignore_trivial=True)

    naive.replace_inf(ref_mp)
    naive.replace_inf(comp_mp)
    npt.assert_almost_equal(ref_mp, comp_mp)


@pytest.mark.parametrize("T_A, T_B", test_data)
def test_aamp_self_join_KNN(T_A, T_B):
    m = 3
    for k in range(2, 4):
        for p in [1.0, 2.0, 3.0]:
            ref_mp = naive.aamp(T_B, m, p=p, k=k)
            comp_mp = aamp(T_B, m, p=p, k=k)
            naive.replace_inf(ref_mp)
            naive.replace_inf(comp_mp)
            npt.assert_almost_equal(ref_mp, comp_mp)

            comp_mp = aamp(pd.Series(T_B), m, p=p, k=k)
            naive.replace_inf(comp_mp)
            npt.assert_almost_equal(ref_mp, comp_mp)


@pytest.mark.parametrize("T_A, T_B", test_data)
def test_aamp_A_B_join_KNN(T_A, T_B):
    m = 3
    for k in range(2, 4):
        for p in [1.0, 2.0, 3.0]:
            ref_mp = naive.aamp(T_A, m, T_B=T_B, p=p, k=k)
            comp_mp = aamp(T_A, m, T_B, ignore_trivial=False, p=p, k=k)
            naive.replace_inf(ref_mp)
            naive.replace_inf(comp_mp)
            npt.assert_almost_equal(ref_mp, comp_mp)

            comp_mp = aamp(
                pd.Series(T_A), m, pd.Series(T_B), ignore_trivial=False, p=p, k=k
            )
            naive.replace_inf(comp_mp)
            npt.assert_almost_equal(ref_mp, comp_mp)
