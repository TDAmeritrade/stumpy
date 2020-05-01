import numpy as np
import numpy.testing as npt
import pandas as pd
from stumpy import scrimp, core, stump
from stumpy.scrimp import _get_max_order_idx, _get_orders_ranges, _prescrimp
import pytest
import utils


test_data = [
    np.array([584, -11, 23, 79, 1001, 0, -19], dtype=np.float64),
    np.random.uniform(-1000, 1000, [64]).astype(np.float64),
]

substitution_locations = [(slice(0, 0), 0, -1, slice(1, 3), [0, 3])]
substitution_values = [np.nan, np.inf]


@pytest.mark.parametrize("T", test_data)
def test_get_max_order_idx(T):
    n = T.shape[0]
    m = 3
    l = n - m + 1
    zone = int(np.ceil(m / 4))
    orders = np.arange(zone + 1, n - m + 2)
    start = 0
    percentage = 1.0
    # Note that the below code only works for `percentage=1.0`
    left_max_order_idx = n - m
    left_n_dist_computed = left_max_order_idx * (left_max_order_idx + 1) // 2

    right_max_order_idx, right_n_dist_computed = _get_max_order_idx(
        m, n, orders, start, percentage
    )

    npt.assert_almost_equal(left_max_order_idx, right_max_order_idx)
    npt.assert_almost_equal(left_n_dist_computed, right_n_dist_computed)


@pytest.mark.parametrize("T", test_data)
def test_get_orders_ranges(T):
    n = T.shape[0]
    m = 3
    l = n - m + 1
    zone = int(np.ceil(m / 4))
    orders = np.arange(zone + 1, n - m + 2)
    n_split = 2
    start = 0
    percentage = 1.0

    # Note that the below code only works for `percentage=1.0`
    left = np.zeros((n_split, 2), np.int64)
    n_distances = n - m + 2 - orders
    cum_arr = n_distances.cumsum() / n_distances.sum()
    idx = 1 + np.searchsorted(cum_arr, np.linspace(0, 1, n_split, endpoint=False)[1:])
    left[1:, 0] = idx  # Fill the first column with start indices
    left[:-1, 1] = idx  # Fill the second column with exclusive stop indices
    left[-1, 1] = n_distances.shape[0]  # Handle the stop index for the final chunk

    right = _get_orders_ranges(n_split, m, n, orders, start, percentage)

    npt.assert_almost_equal(left, right)


@pytest.mark.parametrize("T", test_data)
def test_prescrimp(T):
    m = 3
    zone = int(np.ceil(m / 4))
    left = np.array(
        [
            utils.naive_mass(Q, T, m, i, zone, True)
            for i, Q in enumerate(core.rolling_window(T, m))
        ],
        dtype=object,
    )
    μ, σ = core.compute_mean_std(T, m)
    # Note that the below code only works for `s=1`
    right = _prescrimp(T, m, μ, σ, s=1)


@pytest.mark.parametrize("T", test_data)
def test_scrimp_self_join(T):
    m = 3
    left = np.zeros(T.shape[0])
    right = scrimp(T, m, percentage=0.0)
    utils.replace_inf(right)
    npt.assert_almost_equal(left, right[:, 0])


@pytest.mark.parametrize("T", test_data)
def test_scrimp_self_join(T):
    m = 3
    zone = int(np.ceil(m / 4))
    left = np.array(
        [
            utils.naive_mass(Q, T, m, i, zone, True)
            for i, Q in enumerate(core.rolling_window(T, m))
        ],
        dtype=object,
    )
    for right in scrimp(T, m):
        continue
    utils.replace_inf(left)
    utils.replace_inf(right)
    npt.assert_almost_equal(left[:, 0], right[:, 0])

    for right in scrimp(pd.Series(T), m):
        continue
    utils.replace_inf(right)
    npt.assert_almost_equal(left[:, 0], right[:, 0])


@pytest.mark.parametrize("T", test_data)
def test_scrimp_plus_plus_self_join(T):
    m = 3
    zone = int(np.ceil(m / 4))
    left = np.array(
        [
            utils.naive_mass(Q, T, m, i, zone, True)
            for i, Q in enumerate(core.rolling_window(T, m))
        ],
        dtype=object,
    )
    for right in scrimp(T, m, prescrimp=True):
        continue
    utils.replace_inf(left)
    utils.replace_inf(right)
    npt.assert_almost_equal(left[:, 0], right[:, 0])

    for right in scrimp(pd.Series(T), m, prescrimp=True):
        continue
    utils.replace_inf(right)
    npt.assert_almost_equal(left[:, 0], right[:, 0])


@pytest.mark.parametrize("T", test_data)
def test_scrimp_self_join_larger_window(T):
    for m in [8, 16, 32]:
        if len(T) > m:
            zone = int(np.ceil(m / 4))
            left = np.array(
                [
                    utils.naive_mass(Q, T, m, i, zone, True)
                    for i, Q in enumerate(core.rolling_window(T, m))
                ],
                dtype=object,
            )
            for right in scrimp(T, m):
                continue
            utils.replace_inf(left)
            utils.replace_inf(right)
            npt.assert_almost_equal(left[:, 0], right[:, 0])

            for right in scrimp(pd.Series(T), m):
                continue
            utils.replace_inf(right)
            npt.assert_almost_equal(left[:, 0], right[:, 0])


@pytest.mark.parametrize("T", test_data)
def test_scrimp_plus_plus_self_join_larger_window(T):
    for m in [8, 16, 32]:
        if len(T) > m:
            zone = int(np.ceil(m / 4))
            left = np.array(
                [
                    utils.naive_mass(Q, T, m, i, zone, True)
                    for i, Q in enumerate(core.rolling_window(T, m))
                ],
                dtype=object,
            )
            for right in scrimp(T, m, prescrimp=True):
                continue
            utils.replace_inf(left)
            utils.replace_inf(right)
            npt.assert_almost_equal(left[:, 0], right[:, 0])

            for right in scrimp(pd.Series(T), m, prescrimp=True):
                continue
            utils.replace_inf(right)
            npt.assert_almost_equal(left[:, 0], right[:, 0])


def test_scrimp_constant_subsequence_self_join():
    T = np.concatenate((np.zeros(20, dtype=np.float64), np.ones(5, dtype=np.float64)))
    m = 3
    zone = int(np.ceil(m / 4))
    left = np.array(
        [
            utils.naive_mass(Q, T, m, i, zone, True)
            for i, Q in enumerate(core.rolling_window(T, m))
        ],
        dtype=object,
    )
    for right in scrimp(T, m):
        continue
    utils.replace_inf(left)
    utils.replace_inf(right)
    npt.assert_almost_equal(left[:, 0], right[:, 0])

    for right in scrimp(pd.Series(T), m):
        continue
    utils.replace_inf(right)
    npt.assert_almost_equal(left[:, 0], right[:, 0])


@pytest.mark.parametrize("T", test_data)
@pytest.mark.parametrize("substitute", substitution_values)
@pytest.mark.parametrize("substitution_locations", substitution_locations)
def test_scrimp_nan_inf_self_join(T, substitute, substitution_locations):
    m = 3

    T_sub = T.copy()

    for substitution_location in substitution_locations:
        T_sub[:] = T[:]
        T_sub[substitution_location] = substitute

        zone = int(np.ceil(m / 4))
        left = np.array(
            [
                utils.naive_mass(Q, T_sub, m, i, zone, True)
                for i, Q in enumerate(core.rolling_window(T_sub, m))
            ],
            dtype=object,
        )
        for right in scrimp(T_sub, m):
            continue
        utils.replace_inf(left)
        utils.replace_inf(right)
        npt.assert_almost_equal(left[:, 0], right[:, 0])

        for right in scrimp(pd.Series(T_sub), m):
            continue
        utils.replace_inf(right)
        npt.assert_almost_equal(left[:, 0], right[:, 0])


def test_scrimp_nan_zero_mean_self_join():
    T = np.array([-1, 0, 1, np.inf, 1, 0, -1])
    m = 3

    zone = int(np.ceil(m / 4))
    left = np.array(
        [
            utils.naive_mass(Q, T, m, i, zone, True)
            for i, Q in enumerate(core.rolling_window(T, m))
        ],
        dtype=object,
    )
    for right in scrimp(T, m):
        continue

    utils.replace_inf(left)
    utils.replace_inf(right)
    npt.assert_almost_equal(left[:, 0], right[:, 0])
