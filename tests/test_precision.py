import functools
from unittest.mock import patch

import naive
import numba
import numpy as np
import numpy.testing as npt
import pytest
from numba import cuda

import stumpy
from stumpy import cache, config, core, fastmath

try:
    from numba.errors import NumbaPerformanceWarning
except ModuleNotFoundError:
    from numba.core.errors import NumbaPerformanceWarning

TEST_THREADS_PER_BLOCK = 10


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


def test_calculate_squared_distance():
    # This test function raises an error if the distance between a subsequence
    # and another does not satisfy the symmetry property.
    seed = 332
    np.random.seed(seed)
    T = np.random.uniform(-1000.0, 1000.0, [64])
    m = 3

    T_subseq_isconstant = core.rolling_isconstant(T, m)
    M_T, Σ_T = core.compute_mean_std(T, m)

    n = len(T)
    k = n - m + 1
    for i in range(k):
        for j in range(k):
            QT_i = core._sliding_dot_product(T[i : i + m], T)
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

            QT_j = core._sliding_dot_product(T[j : j + m], T)
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

            npt.assert_almost_equal(ref, comp, decimal=14)


def test_snippets():
    # This test function raises an error if there is a considerable loss of precision
    # that violates the symmetry property of a distance measure.
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

    if (
        not np.allclose(ref_snippets, cmp_snippets) and not numba.config.DISABLE_JIT
    ):  # pragma: no cover
        # Revise fastmath flags by removing reassoc (to improve precision),
        # recompile njit functions, and re-compute snippets.
        fastmath._set(
            "core", "_calculate_squared_distance", {"nsz", "arcp", "contract", "afn"}
        )
        cache._recompile()

        (
            cmp_snippets,
            cmp_indices,
            cmp_profiles,
            cmp_fractions,
            cmp_areas,
            cmp_regimes,
        ) = stumpy.snippets(
            T, m, k, s=s, mpdist_T_subseq_isconstant=isconstant_custom_func
        )

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

    if not numba.config.DISABLE_JIT:  # pragma: no cover
        # Revert fastmath flag back to their default values
        fastmath._reset("core", "_calculate_squared_distance")
        cache._recompile()


@pytest.mark.filterwarnings("ignore", category=NumbaPerformanceWarning)
@patch("stumpy.config.STUMPY_THREADS_PER_BLOCK", TEST_THREADS_PER_BLOCK)
def test_distance_symmetry_property_in_gpu():
    if not cuda.is_available():  # pragma: no cover
        pytest.skip("Skipping Tests No GPUs Available")

    # This test function raises an error if the distance between a subsequence
    # and another one does not satisfy the symmetry property.
    seed = 332
    np.random.seed(seed)
    T = np.random.uniform(-1000.0, 1000.0, [64])
    m = 3

    i, j = 2, 10
    # M_T, Σ_T = core.compute_mean_std(T, m)
    # Σ_T[i] is `650.912209452633`
    # Σ_T[j] is `722.0717285148525`

    # This test raises an error if arithmetic operation in ...
    # ... `gpu_stump._compute_and_update_PI_kernel` does not
    # generates the same result if values of variable for mean and std
    # are swapped.

    T_A = T[i : i + m]
    T_B = T[j : j + m]

    mp_AB = stumpy.gpu_stump(T_A, m, T_B)
    mp_BA = stumpy.gpu_stump(T_B, m, T_A)

    d_ij = mp_AB[0, 0]
    d_ji = mp_BA[0, 0]

    comp = d_ij - d_ji
    ref = 0.0

    npt.assert_almost_equal(comp, ref, decimal=15)
