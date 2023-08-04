import functools

import naive
import numpy as np
import numpy.testing as npt

import stumpy
from stumpy import config, core


def test_mpdist_snippets_s():
    # This test function raises an error if the distance between
    # a subsequence (of length `s`) and itelf becomes non-zero
    # in the performant version. Fixing this loss-of-precision can
    # result in this test being passed.
    seed = 0
    np.random.seed(seed)
    T = np.random.uniform(-1000, 1000, [64]).astype(np.float64)
    m = 10
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
    ) = stumpy.snippets(T, m, k, s=s)

    npt.assert_almost_equal(
        ref_fractions, cmp_fractions, decimal=config.STUMPY_TEST_PRECISION
    )


def test_distace_profile():
    # This test function raises an error when the distance profile between
    # the query `Q = T[i: i+m]`  and `T` becomes non-zero at index `i`.
    T = np.random.rand(64)
    m = 3
    T, M_T, Σ_T, T_subseq_isconstant = core.preprocess(T, m)

    for i in range(len(T) - m + 1):
        Q = T[i : i + m]
        D_ref = naive.distance_profile(Q, T, m)
        D_comp = core.mass(
            Q, T, M_T=M_T, Σ_T=Σ_T, T_subseq_isconstant=T_subseq_isconstant, query_idx=i
        )

        npt.assert_almost_equal(D_ref, D_comp)


def test_snippets():
    m = 10
    k = 3
    s = 3
    seed = 332
    np.random.seed(seed)
    T = np.random.uniform(-1000.0, 1000.0, [64])

    isconstant_custom_func = functools.partial(
        naive.isconstant_func_stddev_threshold, quantile_threshold=0.05
    )
    (
        ref_snippets,
        ref_indices,
        ref_profiles,
        ref_fractions,
        ref_areas,
        ref_regimes,
    ) = naive.mpdist_snippets(
        T, m, k, s=s, mpdist_T_subseq_isconstant=isconstant_custom_func
    )
    (
        cmp_snippets,
        cmp_indices,
        cmp_profiles,
        cmp_fractions,
        cmp_areas,
        cmp_regimes,
    ) = stumpy.snippets(T, m, k, s=s, mpdist_T_subseq_isconstant=isconstant_custom_func)

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


def test_calculate_squared_distance():
    # This test function raises an error if the distance between a subsequence
    # and another does not satisfy the symmetry property.
    seed = 332
    np.random.seed(seed)
    T = np.random.uniform(-1000.0, 1000.0, [64])
    m = 3

    T_subseq_isconstant = core.rolling_isconstant(T, m)
    M_T, Σ_T = core.compute_mean_std(T, m)

    i, j = 2, 10
    # testing the distance between query `Q_i = T[i : i + m]`, and the
    # subsequence T[j : j + m] should be the same as the distance between
    # the query `Q_j = T[j : j + m]` and the subsequence T[i : i + m]

    QT_i = core.sliding_dot_product(T[i : i + m], T)
    dist_ij = core._calculate_squared_distance(
        m,
        QT_i[j],
        M_T[i],
        Σ_T[i],
        M_T[j],
        Σ_T[j],
        T_subseq_isconstant[i],
        T_subseq_isconstant[j],
    )

    QT_j = core.sliding_dot_product(T[j : j + m], T)
    dist_ji = core._calculate_squared_distance(
        m,
        QT_j[i],
        M_T[j],
        Σ_T[j],
        M_T[i],
        Σ_T[i],
        T_subseq_isconstant[j],
        T_subseq_isconstant[i],
    )

    comp = dist_ij - dist_ji
    ref = 0.0

    npt.assert_almost_equal(ref, comp)
