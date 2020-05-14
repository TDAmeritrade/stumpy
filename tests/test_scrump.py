import numpy as np
import numpy.testing as npt
import pandas as pd
from stumpy import scrump, core, stump
from stumpy.scrump import _get_max_order_idx, _get_orders_ranges, prescrump
import pytest
import utils


test_data = [
    np.array([584, -11, 23, 79, 1001, 0, -19], dtype=np.float64),
    np.random.uniform(-1000, 1000, [64]).astype(np.float64),
]

substitution_locations = [(slice(0, 0), 0, -1, slice(1, 3), [0, 3])]
substitution_values = [np.nan, np.inf]
percentages = [(0.01, 0.1, 1.0)]


def naive_get_max_order_idx(m, n, orders, start, percentage):
    max_number_of_distances = 0
    for k in orders:
        max_number_of_distances += (n - m + 1) - k

    distances_to_compute = max_number_of_distances * percentage
    number_of_distances = 0
    for k in orders[start:]:
        number_of_distances += (n - m + 1) - k
        if number_of_distances > distances_to_compute:
            break

    max_order_index = list(orders).index(k) + 1
    return max_order_index, number_of_distances


def naive_get_orders_ranges(n_split, m, n, orders, start, percentage):
    orders_ranges = np.zeros((n_split, 2), np.int64)

    max_order_index, number_of_distances = naive_get_max_order_idx(
        m, n, orders, start, percentage
    )
    number_of_distances_per_thread = number_of_distances / n_split + 1

    current_thread = 0
    current_start = start
    current_number_of_distances = 0
    for index in range(start, max_order_index):
        k = orders[index]

        current_number_of_distances += (n - m + 1) - k
        if current_number_of_distances >= number_of_distances_per_thread:
            orders_ranges[current_thread, 0] = current_start
            orders_ranges[current_thread, 1] = index + 1

            current_thread += 1
            current_start = index + 1
            current_number_of_distances = 0

    # Handle final range outside of for loop
    orders_ranges[current_thread, 0] = current_start
    orders_ranges[current_thread, 1] = index + 1

    return orders_ranges


def naive_prescrump():
    pass


def naive_scrump(T, m, percentage, exclusion_zone, pre_scrump, s):
    distance_matrix = np.array(
        [utils.naive_distance_profile(Q, T, m) for Q in core.rolling_window(T, m)]
    )

    n = T.shape[0]
    l = n - m + 1

    orders = np.random.permutation(range(exclusion_zone + 1, n - m + 1))
    orders_ranges = naive_get_orders_ranges(1, m, n, orders, 0, percentage)
    orders_ranges_start = orders_ranges[0][0]
    orders_ranges_stop = orders_ranges[0][1]

    out = np.full((l, 2), np.inf, dtype=object)
    out[:, 1] = -1

    for order_idx in range(orders_ranges_start, orders_ranges_stop):
        k = orders[order_idx]

        current_diagonal = np.array([[i - j == k for i in range(l)] for j in range(l)])
        current_diagonal_values = distance_matrix[current_diagonal]

        for i in range(0, n - m + 1 - k):
            if current_diagonal_values[i] < out[i, 0]:
                out[i, 0] = current_diagonal_values[i]
                out[i, 1] = i + k
            if current_diagonal_values[i] < out[i + k, 0]:
                out[i + k, 0] = current_diagonal_values[i]
                out[i + k, 1] = i

    return out


@pytest.mark.parametrize("T", test_data)
@pytest.mark.parametrize("percentages", percentages)
def test_get_max_order_idx(T, percentages):
    n = T.shape[0]
    m = 3

    for percentage in percentages:
        zone = int(np.ceil(m / 4))
        orders = np.arange(zone + 1, n - m + 2)
        start = 0

        left_max_order_idx, left_n_dist_computed = naive_get_max_order_idx(
            m, n, orders, start, percentage
        )

        right_max_order_idx, right_n_dist_computed = _get_max_order_idx(
            m, n, orders, start, percentage
        )

        npt.assert_almost_equal(left_max_order_idx, right_max_order_idx)
        npt.assert_almost_equal(left_n_dist_computed, right_n_dist_computed)


@pytest.mark.parametrize("T", test_data)
@pytest.mark.parametrize("percentages", percentages)
def test_get_orders_ranges(T, percentages):
    n = T.shape[0]
    m = 3

    for percentage in percentages:
        zone = int(np.ceil(m / 4))
        orders = np.arange(zone + 1, n - m + 2)
        n_split = 2
        start = 0

        left = naive_get_orders_ranges(n_split, m, n, orders, start, percentage)
        right = _get_orders_ranges(n_split, m, n, orders, start, percentage)

        npt.assert_almost_equal(left, right)


@pytest.mark.parametrize("T", test_data)
def test_prescrump(T):
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
    right = prescrump(T, m, μ, σ, s=1)


@pytest.mark.parametrize("T", test_data)
@pytest.mark.parametrize("percentages", percentages)
def test_scrump_self_join(T, percentages):
    m = 3
    zone = int(np.ceil(m / 4))

    for percentage in percentages:
        seed = np.random.randint(100000)

        np.random.seed(seed)
        left = naive_scrump(T, m, percentage, zone, False, None)

        np.random.seed(seed)
        right_gen = scrump(T, m, percentage=percentage, pre_scrump=False)
        right = next(right_gen)

        utils.replace_inf(left)
        utils.replace_inf(right)
        npt.assert_almost_equal(left, right)


@pytest.mark.parametrize("T", test_data)
def test_scrump_self_join_full(T):
    m = 3
    zone = int(np.ceil(m / 4))

    left = utils.naive_stamp(T, m, exclusion_zone=zone)

    right_gen = scrump(T, m, percentage=1.0, pre_scrump=False)
    right = next(right_gen)

    utils.replace_inf(left)
    utils.replace_inf(right)
    npt.assert_almost_equal(left[:, :2], right)


@pytest.mark.skip(reason="naive PRESCRUMP is not yet implemented")
@pytest.mark.parametrize("T", test_data)
@pytest.mark.parametrize("percentages", percentages)
def test_scrump_plus_plus_self_join(T, percentages):
    m = 3
    zone = int(np.ceil(m / 4))
    s = 1

    for percentage in percentages:
        seed = np.random.randint(100000)

        np.random.seed(seed)
        left = naive_scrump(T, m, percentage, zone, True, s)

        np.random.seed(seed)
        right_gen = scrump(T, m, percentage=percentage, pre_scrump=True, s=s)
        right = next(right_gen)

        utils.replace_inf(left)
        utils.replace_inf(right)
        npt.assert_almost_equal(left, right)


@pytest.mark.parametrize("T", test_data)
def test_scrump_plus_plus_self_join_full(T):
    m = 3
    zone = int(np.ceil(m / 4))
    s = 1

    left = utils.naive_stamp(T, m, exclusion_zone=zone)

    right_gen = scrump(T, m, percentage=1.0, pre_scrump=True, s=s)
    right = next(right_gen)

    utils.replace_inf(left)
    utils.replace_inf(right)
    npt.assert_almost_equal(left[:, :2], right)


@pytest.mark.parametrize("T", test_data)
@pytest.mark.parametrize("percentages", percentages)
def test_scrump_self_join_larger_window(T, percentages):
    for m in [8, 16, 32]:
        if len(T) > m:
            zone = int(np.ceil(m / 4))

            for percentage in percentages:
                seed = np.random.randint(100000)

                np.random.seed(seed)
                left = naive_scrump(T, m, percentage, zone, False, None)

                np.random.seed(seed)
                right_gen = scrump(T, m, percentage=percentage, pre_scrump=False)
                right = next(right_gen)

                utils.replace_inf(left)
                utils.replace_inf(right)
                npt.assert_almost_equal(left, right)


@pytest.mark.parametrize("percentages", percentages)
def test_scrump_constant_subsequence_self_join(percentages):
    T = np.concatenate((np.zeros(20, dtype=np.float64), np.ones(5, dtype=np.float64)))

    m = 3
    zone = int(np.ceil(m / 4))

    for percentage in percentages:
        seed = np.random.randint(100000)

        np.random.seed(seed)
        left = naive_scrump(T, m, percentage, zone, False, None)

        np.random.seed(seed)
        right_gen = scrump(T, m, percentage=percentage, pre_scrump=False)
        right = next(right_gen)

        utils.replace_inf(left)
        utils.replace_inf(right)
        npt.assert_almost_equal(left, right)


@pytest.mark.parametrize("T", test_data)
@pytest.mark.parametrize("substitute", substitution_values)
@pytest.mark.parametrize("substitution_locations", substitution_locations)
@pytest.mark.parametrize("percentages", percentages)
def test_scrump_nan_inf_self_join(T, substitute, substitution_locations, percentages):
    m = 3

    T_sub = T.copy()

    for substitution_location in substitution_locations:
        T_sub[:] = T[:]
        T_sub[substitution_location] = substitute

        zone = int(np.ceil(m / 4))

        for percentage in percentages:
            seed = np.random.randint(100000)

            np.random.seed(seed)
            left = naive_scrump(T_sub, m, percentage, zone, False, None)

            np.random.seed(seed)
            right_gen = scrump(T_sub, m, percentage=percentage, pre_scrump=False)
            right = next(right_gen)

            utils.replace_inf(left)
            utils.replace_inf(right)
            npt.assert_almost_equal(left, right)


@pytest.mark.parametrize("percentages", percentages)
def test_scrump_nan_zero_mean_self_join(percentages):
    T = np.array([-1, 0, 1, np.inf, 1, 0, -1])

    m = 3
    zone = int(np.ceil(m / 4))

    for percentage in percentages:
        seed = np.random.randint(100000)

        np.random.seed(seed)
        left = naive_scrump(T, m, percentage, zone, False, None)

        np.random.seed(seed)
        right_gen = scrump(T, m, percentage=percentage, pre_scrump=False)
        right = next(right_gen)

        utils.replace_inf(left)
        utils.replace_inf(right)
        npt.assert_almost_equal(left, right)
