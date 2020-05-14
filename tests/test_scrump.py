import numpy as np
import numpy.testing as npt
import pandas as pd
from stumpy import scrump, core, stump
from stumpy.scrump import _get_max_order_idx, _get_orders_ranges, prescrump
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

substitution_locations = [(slice(0, 0), 0, -1, slice(1, 3), [0, 3])]
substitution_values = [np.nan, np.inf]
percentages = [(0.01, 0.1, 1.0)]


def naive_get_max_order_idx(m, n_A, n_B, orders, start, percentage):
    matrix = np.empty((n_A - m + 1, n_B - m + 1))
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            matrix[i, j] = j - i

    max_number_of_distances = 0
    for k in orders:
        max_number_of_distances += matrix[matrix == k].size

    distances_to_compute = max_number_of_distances * percentage
    number_of_distances = 0
    for k in orders[start:]:
        number_of_distances += matrix[matrix == k].size
        if number_of_distances > distances_to_compute:
            break

    max_order_index = list(orders).index(k) + 1
    return max_order_index, number_of_distances


def naive_get_orders_ranges(n_split, m, n_A, n_B, orders, start, percentage):
    orders_ranges = np.zeros((n_split, 2), np.int64)

    max_order_index, number_of_distances = naive_get_max_order_idx(
        m, n_A, n_B, orders, start, percentage
    )
    number_of_distances_per_thread = number_of_distances / n_split

    matrix = np.empty((n_A - m + 1, n_B - m + 1))
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            matrix[i, j] = j - i

    current_thread = 0
    current_start = start
    current_number_of_distances = 0
    for index in range(start, max_order_index):
        k = orders[index]
        current_number_of_distances += matrix[matrix == k].size

        if current_number_of_distances > number_of_distances_per_thread:
            orders_ranges[current_thread, 0] = current_start
            orders_ranges[current_thread, 1] = index + 1

            current_thread += 1
            current_start = index + 1
            current_number_of_distances = 0

    # Handle final range outside of for loop if the last thread was not saturated
    if current_thread < orders_ranges.shape[0]:
        orders_ranges[current_thread, 0] = current_start
        orders_ranges[current_thread, 1] = index + 1

    return orders_ranges


def naive_prescrump():
    pass


def naive_scrump(T_A, m, T_B, percentage, exclusion_zone, pre_scrump, s):
    distance_matrix = np.array(
        [utils.naive_distance_profile(Q, T_B, m) for Q in core.rolling_window(T_A, m)]
    )

    n_A = T_A.shape[0]
    n_B = T_B.shape[0]
    l = n_B - m + 1

    if exclusion_zone is not None:
        orders = np.random.permutation(range(exclusion_zone + 1, n_B - m + 1))
    else:
        orders = np.random.permutation(range(-(n_A - m + 1) + 1, n_B - m + 1))

    orders_ranges = naive_get_orders_ranges(1, m, n_A, n_B, orders, 0, percentage)
    orders_ranges_start = orders_ranges[0][0]
    orders_ranges_stop = orders_ranges[0][1]

    out = np.full((l, 2), np.inf, dtype=object)
    out[:, 1] = -1

    for order_idx in range(orders_ranges_start, orders_ranges_stop):
        k = orders[order_idx]

        current_diagonal = np.array(
            [[i - j == k for i in range(n_B - m + 1)] for j in range(n_A - m + 1)]
        )
        current_diagonal_values = distance_matrix[current_diagonal]

        for i in range(current_diagonal_values.size):
            if exclusion_zone is not None and current_diagonal_values[i] < out[i, 0]:
                out[i, 0] = current_diagonal_values[i]
                out[i, 1] = i + k
            if current_diagonal_values[i] < out[i + max(0, k), 0]:
                out[i + max(0, k), 0] = current_diagonal_values[i]
                out[i + max(0, k), 1] = i - min(0, k)

    return out


@pytest.mark.parametrize("T_A, T_B", test_data)
@pytest.mark.parametrize("percentages", percentages)
def test_get_max_order_idx(T_A, T_B, percentages):
    n_A = T_A.shape[0]
    n_B = T_B.shape[0]
    m = 3

    for percentage in percentages:
        # self-joins
        zone = int(np.ceil(m / 4))
        orders = np.arange(zone + 1, n_B - m + 1)
        start = 0

        left_max_order_idx, left_n_dist_computed = naive_get_max_order_idx(
            m, n_B, n_B, orders, start, percentage
        )

        right_max_order_idx, right_n_dist_computed = _get_max_order_idx(
            m, n_B, n_B, orders, start, percentage
        )

        npt.assert_almost_equal(left_max_order_idx, right_max_order_idx)
        npt.assert_almost_equal(left_n_dist_computed, right_n_dist_computed)

        # A-B-joins
        orders = np.arange(-(n_A - m + 1) + 1, n_B - m + 1)
        start = 0
        left_max_order_idx, left_n_dist_computed = naive_get_max_order_idx(
            m, n_A, n_B, orders, start, percentage
        )

        right_max_order_idx, right_n_dist_computed = _get_max_order_idx(
            m, n_A, n_B, orders, start, percentage
        )

        npt.assert_almost_equal(left_max_order_idx, right_max_order_idx)
        npt.assert_almost_equal(left_n_dist_computed, right_n_dist_computed)


@pytest.mark.parametrize("T_A, T_B", test_data)
@pytest.mark.parametrize("percentages", percentages)
def test_get_orders_ranges(T_A, T_B, percentages):
    n_A = T_A.shape[0]
    n_B = T_B.shape[0]
    m = 3

    for percentage in percentages:
        # self-joins
        zone = int(np.ceil(m / 4))
        orders = np.arange(zone + 1, n_B - m + 1)
        n_split = 2
        start = 0

        left = naive_get_orders_ranges(n_split, m, n_B, n_B, orders, start, percentage)
        right = _get_orders_ranges(n_split, m, n_B, n_B, orders, start, percentage)

        npt.assert_almost_equal(left, right)

        # A-B-joins
        orders = np.arange(-(n_A - m + 1) + 1, n_B - m + 1)
        n_split = 2
        start = 0

        left = naive_get_orders_ranges(n_split, m, n_A, n_B, orders, start, percentage)
        right = _get_orders_ranges(n_split, m, n_A, n_B, orders, start, percentage)

        npt.assert_almost_equal(left, right)


@pytest.mark.parametrize("T_A, T_B", test_data)
def test_prescrump_self_join(T_A, T_B):
    m = 3
    zone = int(np.ceil(m / 4))
    left = np.array(
        [
            utils.naive_mass(Q, T_B, m, i, zone, True)
            for i, Q in enumerate(core.rolling_window(T_B, m))
        ],
        dtype=object,
    )
    μ, σ = core.compute_mean_std(T_B, m)
    # Note that the below code only works for `s=1`
    right = prescrump(T_B, m, μ, σ, s=1)


@pytest.mark.parametrize("T_A, T_B", test_data)
@pytest.mark.parametrize("percentages", percentages)
def test_scrump_self_join(T_A, T_B, percentages):
    m = 3
    zone = int(np.ceil(m / 4))

    for percentage in percentages:
        seed = np.random.randint(100000)

        np.random.seed(seed)
        left = naive_scrump(T_B, m, T_B, percentage, zone, False, None)

        np.random.seed(seed)
        right_gen = scrump(
            T_B, m, ignore_trivial=True, percentage=percentage, pre_scrump=False
        )
        right = next(right_gen)

        utils.replace_inf(left)
        utils.replace_inf(right)
        npt.assert_almost_equal(left, right)


@pytest.mark.parametrize("T_A, T_B", test_data)
@pytest.mark.parametrize("percentages", percentages)
def test_scrump_A_B_join(T_A, T_B, percentages):
    m = 3

    for percentage in percentages:
        seed = np.random.randint(100000)

        np.random.seed(seed)
        left = naive_scrump(T_A, m, T_B, percentage, None, False, None)

        np.random.seed(seed)
        right_gen = scrump(
            T_A, m, T_B, ignore_trivial=False, percentage=percentage, pre_scrump=False
        )
        right = next(right_gen)

        utils.replace_inf(left)
        utils.replace_inf(right)
        npt.assert_almost_equal(left, right)


@pytest.mark.parametrize("T_A, T_B", test_data)
def test_scrump_self_join_full(T_A, T_B):
    m = 3
    zone = int(np.ceil(m / 4))

    left = utils.naive_stamp(T_B, m, exclusion_zone=zone)

    right_gen = scrump(T_B, m, ignore_trivial=True, percentage=1.0, pre_scrump=False)
    right = next(right_gen)

    utils.replace_inf(left)
    utils.replace_inf(right)
    npt.assert_almost_equal(left[:, :2], right)


@pytest.mark.parametrize("T_A, T_B", test_data)
def test_scrump_A_B_join_full(T_A, T_B):
    m = 3
    zone = int(np.ceil(m / 4))

    left = utils.naive_stamp(T_A, m, T_B=T_B)

    right_gen = scrump(
        T_A, m, T_B, ignore_trivial=False, percentage=1.0, pre_scrump=False
    )
    right = next(right_gen)

    utils.replace_inf(left)
    utils.replace_inf(right)
    npt.assert_almost_equal(left[:, :2], right)

    # change roles
    left = utils.naive_stamp(T_B, m, T_B=T_A)

    right_gen = scrump(
        T_B, m, T_A, ignore_trivial=False, percentage=1.0, pre_scrump=False
    )
    right = next(right_gen)

    utils.replace_inf(left)
    utils.replace_inf(right)
    npt.assert_almost_equal(left[:, :2], right)


@pytest.mark.skip(reason="naive PRESCRUMP is not yet implemented")
@pytest.mark.parametrize("T_A, T_B", test_data)
@pytest.mark.parametrize("percentages", percentages)
def test_scrump_plus_plus_self_join(T_A, T_B, percentages):
    m = 3
    zone = int(np.ceil(m / 4))
    s = 1

    for percentage in percentages:
        seed = np.random.randint(100000)

        np.random.seed(seed)
        left = naive_scrump(T_B, m, T_B, percentage, zone, True, s)

        np.random.seed(seed)
        right_gen = scrump(
            T_B, m, ignore_trivial=True, percentage=percentage, pre_scrump=True, s=s
        )
        right = next(right_gen)

        utils.replace_inf(left)
        utils.replace_inf(right)
        npt.assert_almost_equal(left, right)


@pytest.mark.parametrize("T_A, T_B", test_data)
def test_scrump_plus_plus_self_join_full(T_A, T_B):
    m = 3
    zone = int(np.ceil(m / 4))
    s = 1

    left = utils.naive_stamp(T_B, m, exclusion_zone=zone)

    right_gen = scrump(
        T_B, m, ignore_trivial=True, percentage=1.0, pre_scrump=True, s=s
    )
    right = next(right_gen)

    utils.replace_inf(left)
    utils.replace_inf(right)
    npt.assert_almost_equal(left[:, :2], right)


@pytest.mark.parametrize("T_A, T_B", test_data)
@pytest.mark.parametrize("percentages", percentages)
def test_scrump_self_join_larger_window(T_A, T_B, percentages):
    for m in [8, 16, 32]:
        if len(T_B) > m:
            zone = int(np.ceil(m / 4))

            for percentage in percentages:
                seed = np.random.randint(100000)

                np.random.seed(seed)
                left = naive_scrump(T_B, m, T_B, percentage, zone, False, None)

                np.random.seed(seed)
                right_gen = scrump(
                    T_B, m, ignore_trivial=True, percentage=percentage, pre_scrump=False
                )
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
        left = naive_scrump(T, m, T, percentage, zone, False, None)

        np.random.seed(seed)
        right_gen = scrump(
            T, m, ignore_trivial=True, percentage=percentage, pre_scrump=False
        )
        right = next(right_gen)

        utils.replace_inf(left)
        utils.replace_inf(right)
        npt.assert_almost_equal(left, right)


@pytest.mark.parametrize("T_A, T_B", test_data)
@pytest.mark.parametrize("substitute", substitution_values)
@pytest.mark.parametrize("substitution_locations", substitution_locations)
@pytest.mark.parametrize("percentages", percentages)
def test_scrump_nan_inf_self_join(
    T_A, T_B, substitute, substitution_locations, percentages
):
    m = 3

    T_B_sub = T_B.copy()

    for substitution_location in substitution_locations:
        T_B_sub[:] = T_B[:]
        T_B_sub[substitution_location] = substitute

        zone = int(np.ceil(m / 4))

        for percentage in percentages:
            seed = np.random.randint(100000)

            np.random.seed(seed)
            left = naive_scrump(T_B_sub, m, T_B_sub, percentage, zone, False, None)

            np.random.seed(seed)
            right_gen = scrump(T_B_sub, m, percentage=percentage, pre_scrump=False)
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
        left = naive_scrump(T, m, T, percentage, zone, False, None)

        np.random.seed(seed)
        right_gen = scrump(T, m, percentage=percentage, pre_scrump=False)
        right = next(right_gen)

        utils.replace_inf(left)
        utils.replace_inf(right)
        npt.assert_almost_equal(left, right)
