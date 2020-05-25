import numpy as np
import numpy.testing as npt
import pytest

from stumpy import motifs, stump
import utils

test_data = [
    (np.array([-1, 1, 2], dtype=np.float64), np.array([0, 1, 1, 2, 3, 4, 4], dtype=np.float64)),
    (
        np.array([9, 8100, -60], dtype=np.float64),
        np.array([584, -11, 23, 79, 1001], dtype=np.float64),
    ),
    (np.random.uniform(-1000, 1000, [8]), np.random.uniform(-1000, 1000, [64])),
]


def naive_find_occurrences(Q, T, excl_zone, profile_value, atol, rtol):
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


def test_k_motifs_static():
    T = np.array([0.0, 1.0, 0.0, -1.0, -1.0, 0.0, 1.0, 0.0, -0.5])
    m = 3
    k = 1

    left_indices = [[0, 5]]
    left_profile_values = [0]

    P = stump(T, m, ignore_trivial=True)
    right_indices, right_profile_values = motifs.k_motifs(T, P, k=k, atol=0.001)

    npt.assert_array_equal(left_indices, right_indices)
    npt.assert_almost_equal(left_profile_values, right_profile_values)


def test_find_occurrences_static():
    T = np.array([0.0, 1.0, 0.0, -1.0, -1.0, 0.0, 1.0, 0.0, -0.5])
    Q = np.array([0.0, 1.0, 0.0])

    left = [0, 5]
    right = motifs.find_occurrences(Q, T, atol=0.001)

    npt.assert_array_equal(left, right)


@pytest.mark.parametrize("Q, T", test_data)
def test_find_occurrences(Q, T):
    m = Q.shape[0]
    excl_zone = int(np.ceil(m / 4))
    rtol = 1
    atol = 1
    profile_value = 1
    
    left = naive_find_occurrences(Q, T, excl_zone, profile_value, atol, rtol)
    right = motifs.find_occurrences(Q, T, excl_zone=excl_zone, profile_value=profile_value, atol=atol, rtol=rtol)

    npt.assert_array_equal(left, right)