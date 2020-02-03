import numpy as np
import numpy.testing as npt
import pandas as pd
from stumpy import stump, _calculate_squared_distance_profile, core
import pytest
import utils


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

substitution_locations = [slice(0, 0), 0, -1, slice(1, 3), [0, 3]]
substitution_values = [np.nan, np.inf]


@pytest.mark.parametrize("Q, T", test_data)
def test_calculate_squared_distance_profile(Q, T):
    m = Q.shape[0]
    left = np.linalg.norm(
        utils.z_norm(core.rolling_window(T, m), 1) - utils.z_norm(Q), axis=1
    )
    left = np.square(left)
    M_T, Σ_T = core.compute_mean_std(T, m)
    QT = core.sliding_dot_product(Q, T)
    μ_Q, σ_Q = core.compute_mean_std(Q, m)
    right = _calculate_squared_distance_profile(m, QT, μ_Q[0], σ_Q[0], M_T, Σ_T)
    npt.assert_almost_equal(left, right)


@pytest.mark.parametrize("T_A, T_B", test_data)
def test_stump_self_join(T_A, T_B):
    m = 3
    zone = int(np.ceil(m / 4))
    left = np.array(
        [
            utils.naive_mass(Q, T_B, m, i, zone, True)
            for i, Q in enumerate(core.rolling_window(T_B, m))
        ],
        dtype=object,
    )
    right = stump(T_B, m, ignore_trivial=True)
    utils.replace_inf(left)
    utils.replace_inf(right)
    npt.assert_almost_equal(left, right)

    right = stump(pd.Series(T_B), m, ignore_trivial=True)
    utils.replace_inf(right)
    npt.assert_almost_equal(left, right)


@pytest.mark.parametrize("T_A, T_B", test_data)
def test_stump_A_B_join(T_A, T_B):
    m = 3
    left = np.array(
        [utils.naive_mass(Q, T_A, m) for Q in core.rolling_window(T_B, m)], dtype=object
    )
    right = stump(T_A, m, T_B, ignore_trivial=False)
    utils.replace_inf(left)
    utils.replace_inf(right)
    npt.assert_almost_equal(left, right)

    right = stump(pd.Series(T_A), m, pd.Series(T_B), ignore_trivial=False)
    utils.replace_inf(right)
    npt.assert_almost_equal(left, right)


def test_constant_subsequence_self_join():
    T_A = np.concatenate((np.zeros(20, dtype=np.float64), np.ones(5, dtype=np.float64)))
    m = 3
    zone = int(np.ceil(m / 4))
    left = np.array(
        [
            utils.naive_mass(Q, T_A, m, i, zone, True)
            for i, Q in enumerate(core.rolling_window(T_A, m))
        ],
        dtype=object,
    )
    right = stump(T_A, m, ignore_trivial=True)
    utils.replace_inf(left)
    utils.replace_inf(right)
    npt.assert_almost_equal(left[:, 0], right[:, 0])  # ignore indices

    right = stump(pd.Series(T_A), m, ignore_trivial=True)
    utils.replace_inf(right)
    npt.assert_almost_equal(left[:, 0], right[:, 0])  # ignore indices


def test_one_constant_subsequence_A_B_join():
    T_A = np.random.rand(20)
    T_B = np.concatenate((np.zeros(20, dtype=np.float64), np.ones(5, dtype=np.float64)))
    m = 3
    left = np.array(
        [utils.naive_mass(Q, T_A, m) for Q in core.rolling_window(T_B, m)], dtype=object
    )
    right = stump(T_A, m, T_B, ignore_trivial=False)
    utils.replace_inf(left)
    utils.replace_inf(right)
    npt.assert_almost_equal(left[:, 0], right[:, 0])  # ignore indices

    right = stump(pd.Series(T_A), m, pd.Series(T_B), ignore_trivial=False)
    utils.replace_inf(right)
    npt.assert_almost_equal(left[:, 0], right[:, 0])  # ignore indices

    # Swap inputs
    left = np.array(
        [utils.naive_mass(Q, T_B, m) for Q in core.rolling_window(T_A, m)], dtype=object
    )
    right = stump(T_B, m, T_A, ignore_trivial=False)
    utils.replace_inf(left)
    utils.replace_inf(right)
    npt.assert_almost_equal(left[:, 0], right[:, 0])  # ignore indices


def test_two_constant_subsequences_A_B_join():
    T_A = np.concatenate(
        (np.zeros(10, dtype=np.float64), np.ones(10, dtype=np.float64))
    )
    T_B = np.concatenate((np.zeros(20, dtype=np.float64), np.ones(5, dtype=np.float64)))
    m = 3
    left = np.array(
        [utils.naive_mass(Q, T_A, m) for Q in core.rolling_window(T_B, m)], dtype=object
    )
    right = stump(T_A, m, T_B, ignore_trivial=False)
    utils.replace_inf(left)
    utils.replace_inf(right)
    npt.assert_almost_equal(left[:, 0], right[:, 0])  # ignore indices

    right = stump(pd.Series(T_A), m, pd.Series(T_B), ignore_trivial=False)
    utils.replace_inf(right)
    npt.assert_almost_equal(left[:, 0], right[:, 0])  # ignore indices

    # Swap inputs
    left = np.array(
        [utils.naive_mass(Q, T_B, m) for Q in core.rolling_window(T_A, m)], dtype=object
    )
    right = stump(T_B, m, T_A, ignore_trivial=False)
    utils.replace_inf(left)
    utils.replace_inf(right)
    npt.assert_almost_equal(left[:, 0], right[:, 0])  # ignore indices

    right = stump(pd.Series(T_B), m, pd.Series(T_A), ignore_trivial=False)
    utils.replace_inf(right)
    npt.assert_almost_equal(left[:, 0], right[:, 0])  # ignore indices


@pytest.mark.parametrize("T_A, T_B", test_data)
@pytest.mark.parametrize("substitute_B", substitution_values)
def test_stump_nan_inf_self_join(T_A, T_B, substitute_B):
    m = 3

    T_B_sub = T_B.copy()

    substitution_locations = [slice(0, 0), 0, -1, slice(1, 3), [0, 3]]
    for substitution_location_B in substitution_locations:
        T_B_sub[:] = T_B[:]
        T_B_sub[substitution_location_B] = substitute_B

        zone = int(np.ceil(m / 4))
        left = np.array(
            [
                utils.naive_mass(Q, T_B_sub, m, i, zone, True)
                for i, Q in enumerate(core.rolling_window(T_B_sub, m))
            ],
            dtype=object,
        )
        right = stump(T_B_sub, m, ignore_trivial=True)
        utils.replace_inf(left)
        utils.replace_inf(right)
        npt.assert_almost_equal(left, right)

        right = stump(pd.Series(T_B_sub), m, ignore_trivial=True)
        utils.replace_inf(right)
        npt.assert_almost_equal(left, right)


@pytest.mark.parametrize("T_A, T_B", test_data)
@pytest.mark.parametrize("substitute_A", substitution_values)
@pytest.mark.parametrize("substitute_B", substitution_values)
def test_stump_nan_inf_A_B_join(T_A, T_B, substitute_A, substitute_B):
    m = 3

    T_A_sub = T_A.copy()
    T_B_sub = T_B.copy()

    substitution_locations = [slice(0, 0), 0, -1, slice(1, 3), [0, 3]]
    for substitution_location_B in substitution_locations:
        for substitution_location_A in substitution_locations:
            T_A_sub[:] = T_A[:]
            T_B_sub[:] = T_B[:]
            T_A_sub[substitution_location_A] = substitute_A
            T_B_sub[substitution_location_B] = substitute_B

            left = np.array(
                [
                    utils.naive_mass(Q, T_A_sub, m)
                    for Q in core.rolling_window(T_B_sub, m)
                ],
                dtype=object,
            )
            right = stump(T_A_sub, m, T_B_sub, ignore_trivial=False)
            utils.replace_inf(left)
            utils.replace_inf(right)
            npt.assert_almost_equal(left, right)

            right = stump(
                pd.Series(T_A_sub), m, pd.Series(T_B_sub), ignore_trivial=False
            )
            utils.replace_inf(right)
            npt.assert_almost_equal(left, right)
