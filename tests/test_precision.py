import functools
from unittest.mock import patch

import mersenne
import naive
import numpy as np
import numpy.testing as npt
import pytest
from numba import cuda

from stumpy import config, core, rng, sdp

if cuda.is_available():
    from stumpy.gpu_stump import gpu_stump
else:  # pragma: no cover
    from stumpy.core import _gpu_stump_dnf as gpu_stump  # noqa: F401

from stumpy.snippets import _get_all_profiles, snippets

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
    state = mersenne.seed_to_state(0)
    with rng.fix_state(state):
        T = rng.RNG.uniform(-1000, 1000, [64]).astype(np.float64)
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
        ) = snippets(T, m, k, s=s)

        npt.assert_almost_equal(
            ref_fractions, cmp_fractions, decimal=config.STUMPY_TEST_PRECISION
        )


def test_distace_profile():
    # This test function raises an error when the distance profile between
    # the query `Q = T[i: i+m]`  and `T` becomes non-zero at index `i`.
    T = rng.RNG.rand(64)
    m = 3
    T, M_T, Σ_T, T_subseq_isconstant = core.preprocess(T, m)

    for i in range(len(T) - m + 1):
        Q = T[i : i + m]
        D_ref = naive.distance_profile(Q, T, m)
        D_comp = core.mass(
            Q, T, M_T=M_T, Σ_T=Σ_T, T_subseq_isconstant=T_subseq_isconstant, query_idx=i
        )

        npt.assert_almost_equal(D_ref, D_comp)


def test_mass_near_perfect_correlation():
    # Regression data from #1171 contains two nearly affine-equivalent subsequences.
    Q = np.array(
        [
            0.93406220661832629,
            0.94732544958491172,
            0.94288413821602846,
        ]
    )
    T = np.array(
        [
            0.25089093970715337,
            0.44046885357649379,
            0.37698714499372254,
        ]
    )
    m = Q.shape[0]

    ref = naive.distance_profile(Q, T, m)
    comp = core.mass(Q, T)

    npt.assert_allclose(ref, comp, rtol=1e-12, atol=1e-12)


def test_mass_near_perfect_correlation_rounded_to_zero():
    Q = np.array([1.0, 2.0, 4.0])
    T = np.array([-4.0, -0.99999999, 5.0])
    m = Q.shape[0]

    ref = naive.distance_profile(Q, T, m)
    comp = core.mass(Q, T)

    npt.assert_allclose(ref, comp, rtol=1e-12, atol=1e-12)


def test_calculate_squared_distance():
    # This test function raises an error if the distance between a subsequence
    # and another does not satisfy the symmetry property.
    state = mersenne.seed_to_state(332)
    with rng.fix_state(state):
        T = rng.RNG.uniform(-1000.0, 1000.0, [64])
        m = 3

        T_subseq_isconstant = core.rolling_isconstant(T, m)
        M_T, Σ_T = core.compute_mean_std(T, m)

        n = len(T)
        k = n - m + 1
        for i in range(k):
            for j in range(k):
                QT_i = sdp._njit_sliding_dot_product(T[i : i + m], T)
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

                QT_j = sdp._njit_sliding_dot_product(T[j : j + m], T)
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


@pytest.mark.parametrize(
    "seed, m, k, s",
    [(2135137202, 10, 3, 3), (2636, 9, 3, 3), (332, 10, 3, 3), (1615, 10, 3, 3)],
)
def test_snippets(seed, m, k, s):
    state = mersenne.seed_to_state(seed)
    with rng.fix_state(state):
        T = rng.RNG.uniform(-1000, 1000, [64]).astype(np.float64)
        isconstant_custom_func = functools.partial(
            naive.isconstant_func_stddev_threshold, quantile_threshold=0.05
        )
        D = _get_all_profiles(
            T,
            m,
            s=s,
            mpdist_T_subseq_isconstant=isconstant_custom_func,
        )
        (
            ref_snippets,
            ref_indices,
            ref_profiles,
            ref_fractions,
            ref_areas,
            ref_regimes,
        ) = naive.mpdist_snippets(
            T,
            m,
            k,
            s=s,
            mpdist_T_subseq_isconstant=isconstant_custom_func,
            D=D,
        )

        (
            cmp_snippets,
            cmp_indices,
            cmp_profiles,
            cmp_fractions,
            cmp_areas,
            cmp_regimes,
        ) = snippets(T, m, k, s=s, mpdist_T_subseq_isconstant=isconstant_custom_func)

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
        npt.assert_almost_equal(
            ref_areas, cmp_areas, decimal=config.STUMPY_TEST_PRECISION
        )
        npt.assert_almost_equal(ref_regimes, cmp_regimes)


@pytest.mark.filterwarnings("ignore", category=NumbaPerformanceWarning)
@patch("stumpy.config.STUMPY_THREADS_PER_BLOCK", TEST_THREADS_PER_BLOCK)
def test_distance_symmetry_property_in_gpu():
    if not cuda.is_available():  # pragma: no cover
        pytest.skip("Skipping Tests No GPUs Available")

    # This test function raises an error if the distance between a subsequence
    # and another one does not satisfy the symmetry property.
    state = mersenne.seed_to_state(332)
    with rng.fix_state(state):
        T = rng.RNG.uniform(-1000.0, 1000.0, [64])
        m = 3

        i, j = 2, 10
        # M_T, Σ_T = core.compute_mean_std(T, m)
        # Σ_T[i] is `650.912209452633`
        # Σ_T[j] is `722.0717285148525`

        # This test raises an error if arithmetic operation in ...
        # ... `gpu_stump._compute_and_update_PI_kernel` does not
        # generate the same result if values of variable for mean and std
        # are swapped.

        T_A = T[i : i + m]
        T_B = T[j : j + m]

        mp_AB = gpu_stump(T_A, m, T_B)
        mp_BA = gpu_stump(T_B, m, T_A)

        d_ij = mp_AB[0, 0]
        d_ji = mp_BA[0, 0]

        comp = d_ij - d_ji
        ref = 0.0

        npt.assert_almost_equal(comp, ref, decimal=15)
