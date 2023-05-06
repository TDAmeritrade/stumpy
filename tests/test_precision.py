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
            Q, T, M_T=M_T, Σ_T=Σ_T, T_subseq_isconstant=T_subseq_isconstant
        )

        npt.assert_almost_equal(D_ref, D_comp)
