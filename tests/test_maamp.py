import naive
import numpy as np
import numpy.testing as npt
import pandas as pd
import pytest

from stumpy import config, core, maamp, maamp_mdl, maamp_subspace
from stumpy.maamp import (
    _get_first_maamp_profile,
    _multi_mass_absolute,
    maamp_multi_distance_profile,
)

test_data = [
    (np.array([[584, -11, 23, 79, 1001, 0, -19]], dtype=np.float64), 3),
    (np.random.uniform(-1000, 1000, [5, 20]).astype(np.float64), 5),
]

substitution_locations = [(slice(0, 0), 0, -1, slice(1, 3), [0, 3])]
substitution_values = [np.nan, np.inf]


def test_multi_mass_absolute_seeded():
    np.random.seed(5)
    T = np.random.uniform(-1000, 1000, [3, 10]).astype(np.float64)
    m = 5

    trivial_idx = 2

    Q = T[:, trivial_idx : trivial_idx + m]

    ref = naive.multi_mass_absolute(Q, T, m)

    T, T_subseq_isfinite = core.preprocess_non_normalized(T, m)
    comp = _multi_mass_absolute(
        Q, T, m, T_subseq_isfinite[:, trivial_idx], T_subseq_isfinite
    )

    npt.assert_almost_equal(ref, comp, decimal=config.STUMPY_TEST_PRECISION)


@pytest.mark.parametrize("T, m", test_data)
def test_multi_mass_absolute(T, m):
    trivial_idx = 2

    Q = T[:, trivial_idx : trivial_idx + m]

    for p in [1.0, 2.0, 3.0]:
        ref = naive.multi_mass_absolute(Q, T, m, p=p)

        _T, T_subseq_isfinite = core.preprocess_non_normalized(T, m)
        comp = _multi_mass_absolute(
            Q, _T, m, T_subseq_isfinite[:, trivial_idx], T_subseq_isfinite, p=p
        )

        npt.assert_almost_equal(ref, comp, decimal=config.STUMPY_TEST_PRECISION)


@pytest.mark.parametrize("T, m", test_data)
def test_maamp_multi_distance_profile(T, m):
    _T, T_subseq_isfinite = core.preprocess_non_normalized(T, m)
    for p in [1.0, 2.0, 3.0]:
        for query_idx in range(_T.shape[0] - m + 1):
            ref_D = naive.maamp_multi_distance_profile(query_idx, _T, m, p=p)

            comp_D = maamp_multi_distance_profile(query_idx, _T, m, p=p)

            npt.assert_almost_equal(ref_D, comp_D)


@pytest.mark.parametrize("T, m", test_data)
def test_get_first_maamp_profile(T, m):
    excl_zone = int(np.ceil(m / 4))
    start = 0

    for p in [1.0, 2.0, 3.0]:
        ref_P, ref_I = naive.maamp(T, m, excl_zone, p=p)
        ref_P = ref_P[:, start]
        ref_I = ref_I[:, start]

        _T, T_subseq_isfinite = core.preprocess_non_normalized(T, m)
        comp_P, comp_I = _get_first_maamp_profile(
            start, _T, _T, m, excl_zone, T_subseq_isfinite, p=p
        )

        npt.assert_almost_equal(ref_P, comp_P)
        npt.assert_equal(ref_I, comp_I)


@pytest.mark.parametrize("T, m", test_data)
def test_maamp_subspace(T, m):
    motif_idx = 1
    nn_idx = 4

    for p in [1.0, 2.0, 3.0]:
        for k in range(T.shape[0]):
            ref_S = naive.maamp_subspace(T, m, motif_idx, nn_idx, k, p=p)
            comp_S = maamp_subspace(T, m, motif_idx, nn_idx, k, p=p)
            npt.assert_almost_equal(ref_S, comp_S)


@pytest.mark.parametrize("T, m", test_data)
def test_maamp_subspace_include(T, m):
    motif_idx = 1
    nn_idx = 4
    for width in range(T.shape[0]):
        for i in range(T.shape[0] - width):
            include = np.asarray(range(i, i + width + 1))

            for k in range(T.shape[0]):
                ref_S = naive.maamp_subspace(T, m, motif_idx, nn_idx, k, include)
                comp_S = maamp_subspace(T, m, motif_idx, nn_idx, k, include)
                npt.assert_almost_equal(ref_S, comp_S)


@pytest.mark.parametrize("T, m", test_data)
def test_maamp_subspace_discords(T, m):
    discord_idx = 1
    nn_idx = 4

    for k in range(T.shape[0]):
        ref_S = naive.maamp_subspace(T, m, discord_idx, nn_idx, k, discords=True)
        comp_S = maamp_subspace(T, m, discord_idx, nn_idx, k, discords=True)
        npt.assert_almost_equal(ref_S, comp_S)


@pytest.mark.parametrize("T, m", test_data)
def test_maamp_subspace_include_discords(T, m):
    discord_idx = 1
    nn_idx = 4
    for width in range(T.shape[0]):
        for i in range(T.shape[0] - width):
            include = np.asarray(range(i, i + width + 1))

            for k in range(T.shape[0]):
                ref_S = naive.maamp_subspace(
                    T, m, discord_idx, nn_idx, k, include, discords=True
                )
                comp_S = maamp_subspace(
                    T, m, discord_idx, nn_idx, k, include, discords=True
                )
                npt.assert_almost_equal(ref_S, comp_S)


@pytest.mark.parametrize("T, m", test_data)
def test_maamp_mdl(T, m):
    subseq_idx = np.full(T.shape[0], 1)
    nn_idx = np.full(T.shape[0], 4)

    for p in [1.0, 2.0, 3.0]:
        ref_MDL, ref_S = naive.maamp_mdl(T, m, subseq_idx, nn_idx, p=p)
        comp_MDL, comp_S = maamp_mdl(T, m, subseq_idx, nn_idx, p=p)
        npt.assert_almost_equal(ref_MDL, comp_MDL)

        for ref, cmp in zip(ref_S, comp_S):
            npt.assert_almost_equal(ref, cmp)


def test_naive_maamp():
    T = np.random.uniform(-1000, 1000, [1, 1000]).astype(np.float64)
    m = 20

    zone = int(np.ceil(m / 4))

    ref_mp = naive.aamp(T[0], m, exclusion_zone=zone)
    ref_P = ref_mp[np.newaxis, :, 0]
    ref_I = ref_mp[np.newaxis, :, 1]

    comp_P, comp_I = naive.maamp(T, m, zone)

    npt.assert_almost_equal(ref_P, comp_P)
    npt.assert_almost_equal(ref_I, comp_I)


def test_maamp_int_input():
    with pytest.raises(TypeError):
        maamp(np.arange(20).reshape(2, 10), 5)


@pytest.mark.parametrize("T, m", test_data)
def test_maamp(T, m):
    excl_zone = int(np.ceil(m / 4))

    for p in [1.0, 2.0, 3.0]:
        ref_P, ref_I = naive.maamp(T, m, excl_zone, p=p)
        comp_P, comp_I = maamp(T, m, p=p)

        npt.assert_almost_equal(ref_P, comp_P)
        npt.assert_almost_equal(ref_I, comp_I)


@pytest.mark.parametrize("T, m", test_data)
def test_maamp_include(T, m):
    for width in range(T.shape[0]):
        for i in range(T.shape[0] - width):
            include = np.asarray(range(i, i + width + 1))
            excl_zone = int(np.ceil(m / 4))

            ref_P, ref_I = naive.maamp(T, m, excl_zone, include)
            comp_P, comp_I = maamp(T, m, include)

            npt.assert_almost_equal(ref_P, comp_P)
            npt.assert_almost_equal(ref_I, comp_I)


@pytest.mark.parametrize("T, m", test_data)
def test_maamp_discords(T, m):
    excl_zone = int(np.ceil(m / 4))

    ref_P, ref_I = naive.maamp(T, m, excl_zone, discords=True)
    comp_P, comp_I = maamp(T, m, discords=True)

    npt.assert_almost_equal(ref_P, comp_P)
    npt.assert_almost_equal(ref_I, comp_I)


@pytest.mark.parametrize("T, m", test_data)
def test_maamp_include_discords(T, m):
    for width in range(T.shape[0]):
        for i in range(T.shape[0] - width):
            include = np.asarray(range(i, i + width + 1))

            excl_zone = int(np.ceil(m / 4))

            ref_P, ref_I = naive.maamp(T, m, excl_zone, include, discords=True)
            comp_P, comp_I = maamp(T, m, include, discords=True)

            npt.assert_almost_equal(ref_P, comp_P)
            npt.assert_almost_equal(ref_I, comp_I)


@pytest.mark.parametrize("T, m", test_data)
def test_maamp_wrapper(T, m):
    excl_zone = int(np.ceil(m / 4))

    ref_P, ref_I = naive.maamp(T, m, excl_zone)
    comp_P, comp_I = maamp(T, m)

    npt.assert_almost_equal(ref_P, comp_P)
    npt.assert_almost_equal(ref_I, comp_I)

    df = pd.DataFrame(T.T)
    comp_P, comp_I = maamp(df, m)

    npt.assert_almost_equal(ref_P, comp_P)
    npt.assert_almost_equal(ref_I, comp_I)


@pytest.mark.parametrize("T, m", test_data)
def test_maamp_wrapper_include(T, m):
    for width in range(T.shape[0]):
        for i in range(T.shape[0] - width):
            include = np.asarray(range(i, i + width + 1))

            excl_zone = int(np.ceil(m / 4))

            ref_P, ref_I = naive.maamp(T, m, excl_zone, include)
            comp_P, comp_I = maamp(T, m, include)

            npt.assert_almost_equal(ref_P, comp_P)
            npt.assert_almost_equal(ref_I, comp_I)

            df = pd.DataFrame(T.T)
            comp_P, comp_I = maamp(df, m, include)

            npt.assert_almost_equal(ref_P, comp_P)
            npt.assert_almost_equal(ref_I, comp_I)


def test_constant_subsequence_self_join():
    T_A = np.concatenate((np.zeros(20, dtype=np.float64), np.ones(5, dtype=np.float64)))
    T = np.array([T_A, T_A, np.random.rand(T_A.shape[0])])
    m = 3

    excl_zone = int(np.ceil(m / 4))

    ref_P, ref_I = naive.maamp(T, m, excl_zone)
    comp_P, comp_I = maamp(T, m)

    npt.assert_almost_equal(ref_P, comp_P)  # ignore indices


def test_identical_subsequence_self_join():
    identical = np.random.rand(8)
    T_A = np.random.rand(20)
    T_A[1 : 1 + identical.shape[0]] = identical
    T_A[11 : 11 + identical.shape[0]] = identical
    T = np.array([T_A, T_A, np.random.rand(T_A.shape[0])])
    m = 3

    excl_zone = int(np.ceil(m / 4))

    ref_P, ref_I = naive.maamp(T, m, excl_zone)
    comp_P, comp_I = maamp(T, m)

    npt.assert_almost_equal(
        ref_P, comp_P, decimal=config.STUMPY_TEST_PRECISION
    )  # ignore indices


@pytest.mark.parametrize("T, m", test_data)
@pytest.mark.parametrize("substitute", substitution_values)
@pytest.mark.parametrize("substitution_locations", substitution_locations)
def test_maamp_nan_inf_self_join_first_dimension(
    T, m, substitute, substitution_locations
):
    excl_zone = int(np.ceil(m / 4))

    T_sub = T.copy()

    for substitution_location in substitution_locations:
        T_sub[:] = T[:]
        T_sub[0, substitution_location] = substitute

        ref_P, ref_I = naive.maamp(T_sub, m, excl_zone)
        comp_P, comp_I = maamp(T_sub, m)

        npt.assert_almost_equal(ref_P, comp_P)
        npt.assert_almost_equal(ref_I, comp_I)


@pytest.mark.parametrize("T, m", test_data)
@pytest.mark.parametrize("substitute", substitution_values)
@pytest.mark.parametrize("substitution_locations", substitution_locations)
def test_maamp_nan_self_join_all_dimensions(T, m, substitute, substitution_locations):
    excl_zone = int(np.ceil(m / 4))

    T_sub = T.copy()

    for substitution_location in substitution_locations:
        T_sub[:] = T[:]
        T_sub[:, substitution_location] = substitute

        ref_P, ref_I = naive.maamp(T_sub, m, excl_zone)
        comp_P, comp_I = maamp(T_sub, m)

        npt.assert_almost_equal(ref_P, comp_P)
        npt.assert_almost_equal(ref_I, comp_I)
