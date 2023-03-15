import naive
import numpy as np
import numpy.testing as npt
import pytest

from stumpy import aampdist_snippets, config

test_data = [np.random.uniform(-1000, 1000, [64]).astype(np.float64)]
s = [6, 7, 8]
percentage = [0.7, 0.8, 0.9]
m = [8, 9, 10]
k = [1, 2, 3]


@pytest.mark.parametrize("T", test_data)
@pytest.mark.parametrize("m", m)
@pytest.mark.parametrize("k", k)
def test_aampdist_snippets(T, m, k):
    for p in [1.0, 2.0, 3.0]:
        (
            ref_snippets,
            ref_indices,
            ref_profiles,
            ref_fractions,
            ref_areas,
            ref_regimes,
        ) = naive.aampdist_snippets(T, m, k, p=p)
        (
            cmp_snippets,
            cmp_indices,
            cmp_profiles,
            cmp_fractions,
            cmp_areas,
            cmp_regimes,
        ) = aampdist_snippets(T, m, k, p=p)

        npt.assert_almost_equal(
            ref_snippets, cmp_snippets, decimal=config.STUMPY_TEST_PRECISION
        )
        npt.assert_almost_equal(
            ref_indices, cmp_indices, decimal=config.STUMPY_TEST_PRECISION
        )
        # npt.assert_almost_equal(
        #     ref_profiles, cmp_profiles, decimal=config.STUMPY_TEST_PRECISION
        # )
        npt.assert_almost_equal(
            ref_fractions, cmp_fractions, decimal=config.STUMPY_TEST_PRECISION
        )
        # npt.assert_almost_equal(
        #     ref_areas, cmp_areas, decimal=config.STUMPY_TEST_PRECISION
        # )
        npt.assert_almost_equal(ref_regimes, cmp_regimes)


@pytest.mark.parametrize("T", test_data)
@pytest.mark.parametrize("m", m)
@pytest.mark.parametrize("k", k)
@pytest.mark.parametrize("percentage", percentage)
def test_mpdist_snippets_percentage(T, m, k, percentage):
    (
        ref_snippets,
        ref_indices,
        ref_profiles,
        ref_fractions,
        ref_areas,
        ref_regimes,
    ) = naive.aampdist_snippets(T, m, k, percentage=percentage)
    (
        cmp_snippets,
        cmp_indices,
        cmp_profiles,
        cmp_fractions,
        cmp_areas,
        cmp_regimes,
    ) = aampdist_snippets(T, m, k, percentage=percentage)

    npt.assert_almost_equal(
        ref_snippets, cmp_snippets, decimal=config.STUMPY_TEST_PRECISION
    )
    npt.assert_almost_equal(
        ref_indices, cmp_indices, decimal=config.STUMPY_TEST_PRECISION
    )
    # npt.assert_almost_equal(
    #     ref_profiles, cmp_profiles, decimal=config.STUMPY_TEST_PRECISION
    # )
    npt.assert_almost_equal(
        ref_fractions, cmp_fractions, decimal=config.STUMPY_TEST_PRECISION
    )
    # npt.assert_almost_equal(
    #     ref_areas, cmp_areas, decimal=config.STUMPY_TEST_PRECISION
    # )
    npt.assert_almost_equal(ref_regimes, cmp_regimes)


@pytest.mark.parametrize("T", test_data)
@pytest.mark.parametrize("m", m)
@pytest.mark.parametrize("k", k)
@pytest.mark.parametrize("s", s)
def test_aampdist_snippets_s(T, m, k, s):
    (
        ref_snippets,
        ref_indices,
        ref_profiles,
        ref_fractions,
        ref_areas,
        ref_regimes,
    ) = naive.aampdist_snippets(T, m, k, s=s)
    (
        cmp_snippets,
        cmp_indices,
        cmp_profiles,
        cmp_fractions,
        cmp_areas,
        cmp_regimes,
    ) = aampdist_snippets(T, m, k, s=s)

    npt.assert_almost_equal(
        ref_snippets, cmp_snippets, decimal=config.STUMPY_TEST_PRECISION
    )
    npt.assert_almost_equal(
        ref_indices, cmp_indices, decimal=config.STUMPY_TEST_PRECISION
    )
    # npt.assert_almost_equal(
    #     ref_profiles, cmp_profiles, decimal=config.STUMPY_TEST_PRECISION
    # )
    npt.assert_almost_equal(
        ref_fractions, cmp_fractions, decimal=config.STUMPY_TEST_PRECISION
    )
    # npt.assert_almost_equal(
    #     ref_areas, cmp_areas, decimal=config.STUMPY_TEST_PRECISION
    # )
    npt.assert_almost_equal(ref_regimes, cmp_regimes)
