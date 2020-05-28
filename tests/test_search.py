import numpy as np
import numpy.testing as npt
import pytest

from stumpy import search, stump
import utils

test_data = [
    (
        np.array([-1, 1, 2], dtype=np.float64),
        np.array([0, 1, 1, 2, 3, 4, 4], dtype=np.float64),
    ),
    (
        np.array([9, 8100, -60], dtype=np.float64),
        np.array([584, -11, 23, 79, 1001], dtype=np.float64),
    ),
    (np.random.uniform(-1000, 1000, [8]), np.random.uniform(-1000, 1000, [64])),
]


def naive_search_occurrences(Q, T, excl_zone, profile_value, atol, rtol):
    m = Q.shape[0]
    distance_profile = utils.naive_distance_profile(Q, T, m)

    occurrences = []
    for i in range(distance_profile.size):
        dist = distance_profile[i]
        if dist < atol + rtol * profile_value:
            occurrences.append(i)

    occurrences.sort(key=lambda x: distance_profile[x])
    result = []
    while len(occurrences) > 0:
        o = occurrences[0]
        result.append(o)
        occurrences = [x for x in occurrences if x < o - excl_zone or x > o + excl_zone]

    return result


def test_search_k_motifs_static():
    T = np.array([0.0, 1.0, 0.0, -1.0, -1.0, 0.0, 1.0, 0.0, -0.5])
    m = 3
    k = 1

    left_indices = [[0, 5]]
    left_profile_values = [0]

    P = stump(T, m, ignore_trivial=True)
    right_indices, right_profile_values = search.search_k_motifs(
        T, P[:, 0], k=k, atol=0.001
    )

    npt.assert_array_equal(left_indices, right_indices)
    npt.assert_almost_equal(left_profile_values, right_profile_values, decimal=4)


def test_search_k_motifs_synthetic():
    T = np.random.normal(size=200)
    m = 20

    T[10:30] = 1
    T[12:28] = 2
    T[110:130] = 1
    T[112:128] = 2
    T[129] = 1.1

    T[70:90] = np.arange(m) * 0.1
    T[170:190] = np.arange(m) * 0.1

    k = 2

    P = stump(T, m, ignore_trivial=True)

    left_indices = [[70, 170], [10, 110]]
    left_profile_values = [P[70, 0], P[10, 0]]

    right_indices, right_profile_values = search.search_k_motifs(T, P[:, 0], k=k)
    right_indices = np.sort(right_indices, axis=1)

    npt.assert_array_equal(left_indices, right_indices)
    npt.assert_almost_equal(left_profile_values, right_profile_values, decimal=4)


def test_search_k_discords_synthetic():
    T = np.sin(2 * np.pi * np.linspace(0, 10, 200))
    m = 20

    T[30] = 20
    T[100] = -10

    k = 2

    left_indices = [[30], [100]]
    left_profile_values = [0, 0]

    P = stump(T, m, ignore_trivial=True)
    right_indices, right_profile_values = search.search_k_discords(T, P[:, 0], k=k)

    npt.assert_array_equal(left_indices, right_indices)
    npt.assert_almost_equal(left_profile_values, right_profile_values, decimal=4)


def test_search_occurrences_static():
    T = np.array([0.0, 1.0, 0.0, -1.0, -1.0, 0.0, 1.0, 0.0, -0.5])
    Q = np.array([0.0, 1.0, 0.0])

    left = [0, 5]
    right = search.search_occurrences(Q, T, atol=0.001)

    npt.assert_array_equal(left, right)


@pytest.mark.parametrize("Q, T", test_data)
def test_search_occurrences(Q, T):
    m = Q.shape[0]
    excl_zone = int(np.ceil(m / 4))
    rtol = 1
    atol = 1
    profile_value = 1

    left = naive_search_occurrences(Q, T, excl_zone, profile_value, atol, rtol)
    right = search.search_occurrences(
        Q,
        T,
        excl_zone=excl_zone,
        max_occurrences=None,
        profile_value=profile_value,
        atol=atol,
        rtol=rtol,
    )

    npt.assert_array_equal(left, right)
