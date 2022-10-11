import numpy as np
import numpy.testing as npt
import pandas as pd
from stumpy import stump, snippets, config
import naive


def test_stump_identical_subsequence_self_join_rare_cases_1():
    # This test function is designed to capture the errors that migtht be raised
    # due the imprecision in the calculation of pearson values in the edge case
    # where two subsequences are identical.
    m = 3
    zone = int(np.ceil(m / 4))

    seed_values = [27343, 84451]
    for seed in seed_values:
        np.random.seed(seed)

        identical = np.random.rand(8)
        T_A = np.random.rand(20)
        T_A[1 : 1 + identical.shape[0]] = identical
        T_A[11 : 11 + identical.shape[0]] = identical

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


def test_stump_identical_subsequence_self_join_rare_cases_2():
    m = 3
    zone = int(np.ceil(m / 4))

    seed_values = [27343, 84451]
    for seed in seed_values:
        np.random.seed(seed)

        identical = np.random.rand(8)
        T_A = np.random.rand(20)
        T_A[1 : 1 + identical.shape[0]] = identical * 0.001
        T_A[11 : 11 + identical.shape[0]] = identical * 1000

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


def test_stump_identical_subsequence_self_join_rare_cases_3():
    m = 3
    zone = int(np.ceil(m / 4))

    seed_values = [27343, 84451]
    for seed in seed_values:
        np.random.seed(seed)

        identical = np.random.rand(8)
        T_A = np.random.rand(20)
        T_A[1 : 1 + identical.shape[0]] = identical * 0.00001
        T_A[11 : 11 + identical.shape[0]] = identical * 100000

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


def test_stump_volatile():
    # return True  # bypassing test for now
    m = 3
    zone = int(np.ceil(m / 4))

    seed_values = [0, 1, 2]
    for seed in seed_values:
        np.random.seed(seed)
        T = np.random.rand(64)
        scale = np.random.choice(np.array([0.001, 0, 1000]), len(T), replace=True)
        T[:] = T * scale

        ref_mp = naive.stump(T, m, exclusion_zone=zone, row_wise=True)
        comp_mp = stump(T, m, ignore_trivial=True)
        naive.replace_inf(ref_mp)
        naive.replace_inf(comp_mp)

        npt.assert_almost_equal(
            ref_mp[:, 0], comp_mp[:, 0], decimal=config.STUMPY_TEST_PRECISION
        )  # ignore indices


def test_snippet_fixed_seeds():
    seed = 15
    np.random.seed(seed)
    T = np.random.uniform(-1000, 1000, [64]).astype(np.float64)
    m = 8
    k = 3
    s = 3

    (
        ref_snippets,
        ref_indices,
        ref_profiles,
        ref_fractions,
        ref_areas,
        ref_regimes,
    ) = naive.mpdist_snippets(T, m, k, s=s)
    (
        cmp_snippets,
        cmp_indices,
        cmp_profiles,
        cmp_fractions,
        cmp_areas,
        cmp_regimes,
    ) = snippets(T, m, k, s=s)

    npt.assert_almost_equal(
        ref_snippets, cmp_snippets, decimal=config.STUMPY_TEST_PRECISION
    )
    npt.assert_almost_equal(
        ref_indices, cmp_indices, decimal=config.STUMPY_TEST_PRECISION
    )
    npt.assert_almost_equal(
        ref_profiles, cmp_profiles, decimal=config.STUMPY_TEST_PRECISION
    )
    npt.assert_almost_equal(
        ref_fractions, cmp_fractions, decimal=config.STUMPY_TEST_PRECISION
    )
    npt.assert_almost_equal(ref_areas, cmp_areas, decimal=config.STUMPY_TEST_PRECISION)
    npt.assert_almost_equal(ref_regimes, cmp_regimes)
