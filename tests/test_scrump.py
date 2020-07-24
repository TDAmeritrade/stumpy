import numpy as np
import numpy.testing as npt
import pandas as pd
from stumpy import scrump, core, stump
from stumpy.scrump import _get_max_order_idx, _get_orders_ranges, prescrump
import pytest
import naive


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
    matrix = np.empty((n_B - m + 1, n_A - m + 1))
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

    matrix = np.empty((n_B - m + 1, n_A - m + 1))
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


def naive_prescrump(T_A, m, T_B, s, exclusion_zone=None):
    distance_matrix = naive.distance_matrix(T_A, T_B, m)

    n_A = T_A.shape[0]
    n_B = T_B.shape[0]
    l = n_B - m + 1

    P = np.empty(l)
    I = np.empty(l, dtype=np.int64)
    P[:] = np.inf
    I[:] = -1

    for i in np.random.permutation(range(0, l, s)):
        distance_profile = distance_matrix[i]
        if exclusion_zone is not None:
            naive.apply_exclusion_zone(distance_profile, i, exclusion_zone)
        I[i] = np.argmin(distance_profile)
        P[i] = distance_profile[I[i]]
        if P[i] == np.inf:
            I[i] = -1

        j = I[i]
        for k in range(1, min(s, l - max(i, j))):
            d = distance_matrix[i + k, j + k]
            if d < P[i + k]:
                P[i + k] = d
                I[i + k] = j + k
            if d < P[j + k]:
                P[j + k] = d
                I[j + k] = i + k

        for k in range(1, min(s, i + 1, j + 1)):
            d = distance_matrix[i - k, j - k]
            if d < P[i - k]:
                P[i - k] = d
                I[i - k] = j - k
            if d < P[j - k]:
                P[j - k] = d
                I[j - k] = i - k

    return P, I


def naive_scrump(T_A, m, T_B, percentage, exclusion_zone, pre_scrump, s):
    distance_matrix = naive.distance_matrix(T_A, T_B, m)

    n_A = T_A.shape[0]
    n_B = T_B.shape[0]
    l = n_B - m + 1

    if exclusion_zone is not None:
        orders = np.random.permutation(range(exclusion_zone + 1, n_B - m + 1))
    else:
        orders = np.random.permutation(range(-(n_B - m + 1) + 1, n_A - m + 1))

    orders_ranges = naive_get_orders_ranges(1, m, n_A, n_B, orders, 0, percentage)
    orders_ranges_start = orders_ranges[0][0]
    orders_ranges_stop = orders_ranges[0][1]

    out = np.full((l, 4), np.inf, dtype=object)
    out[:, 1:] = -1
    left_P = np.full(l, np.inf, dtype=np.float64)
    right_P = np.full(l, np.inf, dtype=np.float64)

    for order_idx in range(orders_ranges_start, orders_ranges_stop):
        k = orders[order_idx]

        for i in range(n_B - m + 1):
            for j in range(n_A - m + 1):
                if j - i == k:
                    if distance_matrix[i, j] < out[i, 0]:
                        out[i, 0] = distance_matrix[i, j]
                        out[i, 1] = i + k

                    if (
                        exclusion_zone is not None
                        and distance_matrix[i, j] < out[i + k, 0]
                    ):
                        out[i + k, 0] = distance_matrix[i, j]
                        out[i + k, 1] = i

                    # left matrix profile and left matrix profile indices
                    if (
                        exclusion_zone is not None
                        and i < i + k
                        and distance_matrix[i, j] < left_P[i + k]
                    ):
                        left_P[i + k] = distance_matrix[i, j]
                        out[i + k, 2] = i

                    # right matrix profile and right matrix profile indices
                    if (
                        exclusion_zone is not None
                        and i + k > i
                        and distance_matrix[i, j] < right_P[i]
                    ):
                        right_P[i] = distance_matrix[i, j]
                        out[i, 3] = i + k

    return out


@pytest.mark.parametrize("T_A, T_B", test_data)
@pytest.mark.parametrize("percentages", percentages)
def test_get_max_order_idx(T_A, T_B, percentages):
    n_A = T_A.shape[0]
    n_B = T_B.shape[0]
    m = 3

    for percentage in percentages:
        # self-join
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

        # AB-join
        orders = np.arange(-(n_B - m + 1) + 1, n_A - m + 1)
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
        # self-join
        zone = int(np.ceil(m / 4))
        orders = np.arange(zone + 1, n_B - m + 1)
        n_split = 2
        start = 0

        left = naive_get_orders_ranges(n_split, m, n_B, n_B, orders, start, percentage)
        right = _get_orders_ranges(n_split, m, n_B, n_B, orders, start, percentage)

        npt.assert_almost_equal(left, right)

        # AB-join
        orders = np.arange(-(n_B - m + 1) + 1, n_A - m + 1)

        n_split = 2
        start = 0

        left = naive_get_orders_ranges(n_split, m, n_A, n_B, orders, start, percentage)
        right = _get_orders_ranges(n_split, m, n_A, n_B, orders, start, percentage)

        npt.assert_almost_equal(left, right)


@pytest.mark.parametrize("T_A, T_B", test_data)
def test_prescrump_self_join(T_A, T_B):
    m = 3
    zone = int(np.ceil(m / 4))
    for s in range(1, zone + 1):
        seed = np.random.randint(100000)

        np.random.seed(seed)
        left_P, left_I = naive_prescrump(T_B, m, T_B, s=s, exclusion_zone=zone)

        np.random.seed(seed)
        right_P, right_I = prescrump(T_B, m, s=s)

        npt.assert_almost_equal(left_P, right_P)
        npt.assert_almost_equal(left_I, right_I)


@pytest.mark.parametrize("T_A, T_B", test_data)
def test_prescrump_A_B_join(T_A, T_B):
    m = 3
    zone = int(np.ceil(m / 4))
    for s in range(1, zone + 1):
        seed = np.random.randint(100000)

        np.random.seed(seed)
        left_P, left_I = naive_prescrump(T_A, m, T_B, s=s)

        np.random.seed(seed)
        right_P, right_I = prescrump(T_A, m, T_B=T_B, s=s)

        npt.assert_almost_equal(left_P, right_P)
        npt.assert_almost_equal(left_I, right_I)


@pytest.mark.parametrize("T_A, T_B", test_data)
def test_prescrump_A_B_join_swap(T_A, T_B):
    m = 3
    zone = int(np.ceil(m / 4))
    for s in range(1, zone + 1):
        seed = np.random.randint(100000)

        np.random.seed(seed)
        left_P, left_I = naive_prescrump(T_B, m, T_A, s=s)

        np.random.seed(seed)
        right_P, right_I = prescrump(T_B, m, T_B=T_A, s=s)

        npt.assert_almost_equal(left_P, right_P)
        npt.assert_almost_equal(left_I, right_I)


@pytest.mark.parametrize("T_A, T_B", test_data)
def test_prescrump_self_join_larger_window(T_A, T_B):
    for m in [8, 16, 32]:
        if len(T_B) > m:
            zone = int(np.ceil(m / 4))
            for s in range(1, zone + 1):
                seed = np.random.randint(100000)

                np.random.seed(seed)
                left_P, left_I = naive_prescrump(T_B, m, T_B, s=s, exclusion_zone=zone)

                np.random.seed(seed)
                right_P, right_I = prescrump(T_B, m, s=s)

                npt.assert_almost_equal(left_P, right_P)
                npt.assert_almost_equal(left_I, right_I)


@pytest.mark.parametrize("T_A, T_B", test_data)
@pytest.mark.parametrize("percentages", percentages)
def test_scrump_self_join(T_A, T_B, percentages):
    m = 3
    zone = int(np.ceil(m / 4))

    for percentage in percentages:
        seed = np.random.randint(100000)

        np.random.seed(seed)
        left = naive_scrump(T_B, m, T_B, percentage, zone, False, None)
        left_P = left[:, 0]
        left_I = left[:, 1]
        left_left_I = left[:, 2]
        left_right_I = left[:, 3]

        np.random.seed(seed)
        approx = scrump(
            T_B, m, ignore_trivial=True, percentage=percentage, pre_scrump=False
        )
        approx.update()
        right_P = approx.P_
        right_I = approx.I_
        right_left_I = approx.left_I_
        right_right_I = approx.right_I_

        naive.replace_inf(left_P)
        naive.replace_inf(right_P)
        npt.assert_almost_equal(left_P, right_P)
        npt.assert_almost_equal(left_I, right_I)
        npt.assert_almost_equal(left_left_I, right_left_I)
        npt.assert_almost_equal(left_right_I, right_right_I)


@pytest.mark.parametrize("T_A, T_B", test_data)
@pytest.mark.parametrize("percentages", percentages)
def test_scrump_A_B_join(T_A, T_B, percentages):
    m = 3

    for percentage in percentages:
        seed = np.random.randint(100000)

        np.random.seed(seed)
        left = naive_scrump(T_A, m, T_B, percentage, None, False, None)
        left_P = left[:, 0]
        left_I = left[:, 1]
        left_left_I = left[:, 2]
        left_right_I = left[:, 3]

        np.random.seed(seed)
        approx = scrump(
            T_A, m, T_B, ignore_trivial=False, percentage=percentage, pre_scrump=False
        )
        approx.update()
        right_P = approx.P_
        right_I = approx.I_
        right_left_I = approx.left_I_
        right_right_I = approx.right_I_

        naive.replace_inf(left_P)
        naive.replace_inf(right_P)

        npt.assert_almost_equal(left_P, right_P)
        npt.assert_almost_equal(left_I, right_I)
        npt.assert_almost_equal(left_left_I, right_left_I)
        npt.assert_almost_equal(left_right_I, right_right_I)


@pytest.mark.parametrize("T_A, T_B", test_data)
@pytest.mark.parametrize("percentages", percentages)
def test_scrump_A_B_join_swap(T_A, T_B, percentages):
    m = 3

    for percentage in percentages:
        seed = np.random.randint(100000)

        np.random.seed(seed)
        left = naive_scrump(T_B, m, T_A, percentage, None, False, None)
        left_P = left[:, 0]
        left_I = left[:, 1]
        left_left_I = left[:, 2]
        left_right_I = left[:, 3]

        np.random.seed(seed)
        approx = scrump(
            T_B, m, T_A, ignore_trivial=False, percentage=percentage, pre_scrump=False
        )
        approx.update()
        right_P = approx.P_
        right_I = approx.I_
        right_left_I = approx.left_I_
        right_right_I = approx.right_I_

        naive.replace_inf(left_P)
        naive.replace_inf(right_P)

        npt.assert_almost_equal(left_P, right_P)
        npt.assert_almost_equal(left_P, right_P)
        npt.assert_almost_equal(left_left_I, right_left_I)
        npt.assert_almost_equal(left_right_I, right_right_I)


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
                left_P = left[:, 0]
                left_I = left[:, 1]
                left_left_I = left[:, 2]
                left_right_I = left[:, 3]

                np.random.seed(seed)
                approx = scrump(
                    T_B, m, ignore_trivial=True, percentage=percentage, pre_scrump=False
                )
                approx.update()
                right_P = approx.P_
                right_I = approx.I_
                right_left_I = approx.left_I_
                right_right_I = approx.right_I_

                naive.replace_inf(left_P)
                naive.replace_inf(right_P)

                npt.assert_almost_equal(left_P, right_P)
                npt.assert_almost_equal(left_I, right_I)
                npt.assert_almost_equal(left_left_I, right_left_I)
                npt.assert_almost_equal(left_right_I, right_right_I)


@pytest.mark.parametrize("T_A, T_B", test_data)
def test_scrump_self_join_full(T_A, T_B):
    m = 3
    zone = int(np.ceil(m / 4))

    left = naive.stamp(T_B, m, exclusion_zone=zone)
    left_P = left[:, 0]
    left_I = left[:, 1]
    left_left_I = left[:, 2]
    left_right_I = left[:, 3]

    approx = scrump(T_B, m, ignore_trivial=True, percentage=1.0, pre_scrump=False)
    approx.update()
    right_P = approx.P_
    right_I = approx.I_
    right_left_I = approx.left_I_
    right_right_I = approx.right_I_

    naive.replace_inf(left_P)
    naive.replace_inf(right_P)

    npt.assert_almost_equal(left_P, right_P)
    npt.assert_almost_equal(left_I, right_I)
    npt.assert_almost_equal(left_left_I, right_left_I)
    npt.assert_almost_equal(left_right_I, right_right_I)

    left = stump(T_B, m, ignore_trivial=True)
    left_P = left[:, 0]
    left_I = left[:, 1]
    left_left_I = left[:, 2]
    left_right_I = left[:, 3]

    npt.assert_almost_equal(left_P, right_P)
    npt.assert_almost_equal(left_I, right_I)
    npt.assert_almost_equal(left_left_I, right_left_I)
    npt.assert_almost_equal(left_right_I, right_right_I)


@pytest.mark.parametrize("T_A, T_B", test_data)
def test_scrump_A_B_join_full(T_A, T_B):

    m = 3
    zone = int(np.ceil(m / 4))

    left = naive.stamp(T_A, m, T_B=T_B)
    left_P = left[:, 0]
    left_I = left[:, 1]
    left_left_I = left[:, 2]
    left_right_I = left[:, 3]

    approx = scrump(T_A, m, T_B, ignore_trivial=False, percentage=1.0, pre_scrump=False)
    approx.update()
    right_P = approx.P_
    right_I = approx.I_
    right_left_I = approx.left_I_
    right_right_I = approx.right_I_

    naive.replace_inf(left_P)
    naive.replace_inf(right_P)

    npt.assert_almost_equal(left_P, right_P)
    npt.assert_almost_equal(left_I, right_I)
    npt.assert_almost_equal(left_left_I, right_left_I)
    npt.assert_almost_equal(left_right_I, right_right_I)

    left = stump(T_A, m, T_B=T_B, ignore_trivial=False)
    left_P = left[:, 0]
    left_I = left[:, 1]
    left_left_I = left[:, 2]
    left_right_I = left[:, 3]

    npt.assert_almost_equal(left_P, right_P)
    npt.assert_almost_equal(left_I, right_I)
    npt.assert_almost_equal(left_left_I, right_left_I)
    npt.assert_almost_equal(left_right_I, right_right_I)


@pytest.mark.parametrize("T_A, T_B", test_data)
def test_scrump_A_B_join_full_swap(T_A, T_B):

    m = 3
    zone = int(np.ceil(m / 4))

    left = naive.stamp(T_B, m, T_B=T_A)
    left_P = left[:, 0]
    left_I = left[:, 1]
    left_left_I = left[:, 2]
    left_right_I = left[:, 3]

    approx = scrump(T_B, m, T_A, ignore_trivial=False, percentage=1.0, pre_scrump=False)
    approx.update()
    right_P = approx.P_
    right_I = approx.I_
    right_left_I = approx.left_I_
    right_right_I = approx.right_I_

    naive.replace_inf(left_P)
    naive.replace_inf(right_P)

    npt.assert_almost_equal(left_P, right_P)
    npt.assert_almost_equal(left_I, right_I)
    npt.assert_almost_equal(left_left_I, right_left_I)
    npt.assert_almost_equal(left_right_I, right_right_I)


@pytest.mark.parametrize("T_A, T_B", test_data)
def test_scrump_self_join_full_larger_window(T_A, T_B):
    for m in [8, 16, 32]:
        if len(T_B) > m:
            zone = int(np.ceil(m / 4))

            left = naive.stamp(T_B, m, exclusion_zone=zone)
            left_P = left[:, 0]
            left_I = left[:, 1]
            left_left_I = left[:, 2]
            left_right_I = left[:, 3]

            approx = scrump(
                T_B, m, ignore_trivial=True, percentage=1.0, pre_scrump=False
            )
            approx.update()
            right_P = approx.P_
            right_I = approx.I_
            right_left_I = approx.left_I_
            right_right_I = approx.right_I_

            naive.replace_inf(left_P)
            naive.replace_inf(right_P)

            npt.assert_almost_equal(left_P, right_P)
            npt.assert_almost_equal(left_I, right_I)
            npt.assert_almost_equal(left_left_I, right_left_I)
            npt.assert_almost_equal(left_right_I, right_right_I)


@pytest.mark.parametrize("T_A, T_B", test_data)
@pytest.mark.parametrize("percentages", percentages)
def test_scrump_plus_plus_self_join(T_A, T_B, percentages):
    m = 3
    zone = int(np.ceil(m / 4))

    for s in range(1, zone + 1):
        for percentage in percentages:
            seed = np.random.randint(100000)

            np.random.seed(seed)
            left_P, left_I = naive_prescrump(T_B, m, T_B, s=s, exclusion_zone=zone)
            left = naive_scrump(T_B, m, T_B, percentage, zone, True, s)
            for i in range(left.shape[0]):
                if left_P[i] < left[i, 0]:
                    left[i, 0] = left_P[i]
                    left[i, 1] = left_I[i]
            left_P = left[:, 0]
            left_I = left[:, 1]
            left_left_I = left[:, 2]
            left_right_I = left[:, 3]

            np.random.seed(seed)
            approx = scrump(
                T_B, m, ignore_trivial=True, percentage=percentage, pre_scrump=True, s=s
            )
            approx.update()
            right_P = approx.P_
            right_I = approx.I_
            right_left_I = approx.left_I_
            right_right_I = approx.right_I_

            naive.replace_inf(left_P)
            naive.replace_inf(right_I)

            npt.assert_almost_equal(left_P, right_P)
            npt.assert_almost_equal(left_I, right_I)
            # npt.assert_almost_equal(left_left_I, right_left_I)
            # npt.assert_almost_equal(left_right_I, right_right_I)


@pytest.mark.parametrize("T_A, T_B", test_data)
@pytest.mark.parametrize("percentages", percentages)
def test_scrump_plus_plus_A_B_join(T_A, T_B, percentages):
    m = 3
    zone = int(np.ceil(m / 4))

    for s in range(1, zone + 1):
        for percentage in percentages:
            seed = np.random.randint(100000)

            np.random.seed(seed)
            left_P, left_I = naive_prescrump(T_A, m, T_B, s=s)
            left = naive_scrump(T_A, m, T_B, percentage, None, False, None)
            for i in range(left.shape[0]):
                if left_P[i] < left[i, 0]:
                    left[i, 0] = left_P[i]
                    left[i, 1] = left_I[i]
            left_P = left[:, 0]
            left_I = left[:, 1]
            left_left_I = left[:, 2]
            left_right_I = left[:, 3]

            approx = scrump(
                T_A,
                m,
                T_B,
                ignore_trivial=False,
                percentage=percentage,
                pre_scrump=True,
                s=s,
            )
            approx.update()
            right_P = approx.P_
            right_I = approx.I_
            right_left_I = approx.left_I_
            right_right_I = approx.right_I_

            naive.replace_inf(left_P)
            naive.replace_inf(right_P)

            npt.assert_almost_equal(left_P, right_P)
            npt.assert_almost_equal(left_I, right_I)
            npt.assert_almost_equal(left_left_I, right_left_I)
            npt.assert_almost_equal(left_right_I, right_right_I)


@pytest.mark.parametrize("T_A, T_B", test_data)
def test_scrump_plus_plus_self_join_full(T_A, T_B):
    m = 3
    zone = int(np.ceil(m / 4))

    left = naive.stamp(T_B, m, exclusion_zone=zone)
    left_P = left[:, 0]
    left_I = left[:, 1]
    left_left_I = left[:, 2]
    left_right_I = left[:, 3]

    approx = scrump(
        T_B, m, ignore_trivial=True, percentage=1.0, pre_scrump=True, s=zone
    )
    approx.update()
    right_P = approx.P_
    right_I = approx.I_
    right_left_I = approx.left_I_
    right_right_I = approx.right_I_

    naive.replace_inf(left_P)
    naive.replace_inf(right_P)

    npt.assert_almost_equal(left_P, right_P)
    npt.assert_almost_equal(left_I, right_I)
    npt.assert_almost_equal(left_left_I, right_left_I)
    npt.assert_almost_equal(left_right_I, right_right_I)


@pytest.mark.parametrize("T_A, T_B", test_data)
def test_scrump_plus_plus_A_B_join_full(T_A, T_B):
    m = 3
    zone = int(np.ceil(m / 4))

    left = naive.stamp(T_A, m, T_B=T_B)
    left_P = left[:, 0]
    left_I = left[:, 1]
    left_left_I = left[:, 2]
    left_right_I = left[:, 3]

    approx = scrump(
        T_A, m, T_B=T_B, ignore_trivial=False, percentage=1.0, pre_scrump=True, s=zone
    )
    approx.update()
    right_P = approx.P_
    right_I = approx.I_
    right_left_I = approx.left_I_
    right_right_I = approx.right_I_

    naive.replace_inf(left_P)
    naive.replace_inf(right_P)

    npt.assert_almost_equal(left_P, right_P)
    npt.assert_almost_equal(left_I, right_I)
    npt.assert_almost_equal(left_left_I, right_left_I)
    npt.assert_almost_equal(left_right_I, right_right_I)


@pytest.mark.parametrize("T_A, T_B", test_data)
def test_scrump_plus_plus_A_B_join_full_swap(T_A, T_B):
    m = 3
    zone = int(np.ceil(m / 4))

    left = naive.stamp(T_B, m, T_B=T_A)
    left_P = left[:, 0]
    left_I = left[:, 1]
    left_left_I = left[:, 2]
    left_right_I = left[:, 3]

    approx = scrump(
        T_B, m, T_B=T_A, ignore_trivial=False, percentage=1.0, pre_scrump=True, s=zone
    )
    approx.update()
    right_P = approx.P_
    right_I = approx.I_
    right_left_I = approx.left_I_
    right_right_I = approx.right_I_

    naive.replace_inf(left_P)
    naive.replace_inf(right_P)

    npt.assert_almost_equal(left_P, right_P)
    npt.assert_almost_equal(left_I, right_I)
    npt.assert_almost_equal(left_left_I, right_left_I)
    npt.assert_almost_equal(left_right_I, right_right_I)


@pytest.mark.parametrize("percentages", percentages)
def test_scrump_constant_subsequence_self_join(percentages):
    T = np.concatenate((np.zeros(20, dtype=np.float64), np.ones(5, dtype=np.float64)))

    m = 3
    zone = int(np.ceil(m / 4))

    for percentage in percentages:
        seed = np.random.randint(100000)

        np.random.seed(seed)
        left = naive_scrump(T, m, T, percentage, zone, False, None)
        left_P = left[:, 0]
        left_I = left[:, 1]
        left_left_I = left[:, 2]
        left_right_I = left[:, 3]

        np.random.seed(seed)
        approx = scrump(
            T, m, ignore_trivial=True, percentage=percentage, pre_scrump=False
        )
        approx.update()
        right_P = approx.P_
        right_I = approx.I_
        right_left_I = approx.left_I_
        right_right_I = approx.right_I_

        naive.replace_inf(left_P)
        naive.replace_inf(right_P)

        npt.assert_almost_equal(left_P, right_P)
        npt.assert_almost_equal(left_I, right_I)
        npt.assert_almost_equal(left_left_I, right_left_I)
        npt.assert_almost_equal(left_right_I, right_right_I)


@pytest.mark.parametrize("percentages", percentages)
def test_scrump_identical_subsequence_self_join(percentages):
    identical = np.random.rand(8)
    T = np.random.rand(20)
    T[1 : 1 + identical.shape[0]] = identical
    T[11 : 11 + identical.shape[0]] = identical
    m = 3
    zone = int(np.ceil(m / 4))

    for percentage in percentages:
        seed = np.random.randint(100000)

        np.random.seed(seed)
        left = naive_scrump(T, m, T, percentage, zone, False, None)
        left_P = left[:, 0]
        left_I = left[:, 1]
        left_left_I = left[:, 2]
        left_right_I = left[:, 3]

        np.random.seed(seed)
        approx = scrump(
            T, m, ignore_trivial=True, percentage=percentage, pre_scrump=False
        )
        approx.update()
        right_P = approx.P_
        right_I = approx.I_
        right_left_I = approx.left_I_
        right_right_I = approx.right_I_

        naive.replace_inf(left_P)
        naive.replace_inf(right_P)

        npt.assert_almost_equal(left_P, right_P, decimal=naive.PRECISION)
        # npt.assert_almost_equal(left_I, right_I)
        # npt.assert_almost_equal(left_left_I, right_left_I)
        # npt.assert_almost_equal(left_right_I, right_right_I)


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
            left_P = left[:, 0]
            left_I = left[:, 1]
            left_left_I = left[:, 2]
            left_right_I = left[:, 3]

            np.random.seed(seed)
            approx = scrump(T_B_sub, m, percentage=percentage, pre_scrump=False)
            approx.update()
            right_P = approx.P_
            right_I = approx.I_
            right_left_I = approx.left_I_
            right_right_I = approx.right_I_

            naive.replace_inf(left_P)
            naive.replace_inf(right_P)

            npt.assert_almost_equal(left_P, right_P)
            npt.assert_almost_equal(left_I, right_I)
            npt.assert_almost_equal(left_left_I, right_left_I)
            npt.assert_almost_equal(left_right_I, right_right_I)


@pytest.mark.parametrize("percentages", percentages)
def test_scrump_nan_zero_mean_self_join(percentages):
    T = np.array([-1, 0, 1, np.inf, 1, 0, -1])

    m = 3
    zone = int(np.ceil(m / 4))

    for percentage in percentages:
        seed = np.random.randint(100000)

        np.random.seed(seed)
        left = naive_scrump(T, m, T, percentage, zone, False, None)
        left_P = left[:, 0]
        left_I = left[:, 1]
        left_left_I = left[:, 2]
        left_right_I = left[:, 3]

        np.random.seed(seed)
        approx = scrump(T, m, percentage=percentage, pre_scrump=False)
        approx.update()
        right_P = approx.P_
        right_I = approx.I_
        right_left_I = approx.left_I_
        right_right_I = approx.right_I_

        naive.replace_inf(left_P)
        naive.replace_inf(right_P)

        npt.assert_almost_equal(left_P, right_P)
        npt.assert_almost_equal(left_I, right_I)
        npt.assert_almost_equal(left_left_I, right_left_I)
        npt.assert_almost_equal(left_right_I, right_right_I)
