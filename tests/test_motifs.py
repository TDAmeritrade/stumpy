import numpy as np
import numpy.testing as npt
import pytest

from stumpy import core, motifs, occurrences

from stumpy.motifs import _fill_result_with_nans

import naive

test_data = [
    (
        np.array([0.0, 1.0, 0.0]),
        np.array([0.0, 1.0, 0.0, -1.0, -1.0, 0.0, 1.0, 0.0, -0.5]),
    ),
    (
        np.array([0.0, 1.0, 2.0]),
        np.array([0.1, 1.0, 2.0, 3.0, -1.0, 0.1, 1.0, 2.0, -0.5]),
    ),
    (np.random.uniform(-1000, 1000, [8]), np.random.uniform(-1000, 1000, [64])),
]
excl_zones = [-1, None]


def naive_occurrences(Q, T, excl_zone, profile_value, atol, rtol):
    m = Q.shape[0]
    D = naive.distance_profile(Q, T, m)

    # Finds all indices that have a lower distance profile value `D`
    # than `atol + rtol * D`
    occurrences = []
    for i in range(D.size):
        dist = D[i]
        if dist < atol + rtol * profile_value:
            occurrences.append(i)

    # Removes indices that are inside the exclusion zone of some occurrence with
    # a smaller distance to the query
    occurrences.sort(key=lambda x: D[x])
    result = []
    while len(occurrences) > 0:
        o = occurrences[0]
        result.append([o, D[o]])
        occurrences = [x for x in occurrences if x < o - excl_zone or x > o + excl_zone]

    return np.array(result, dtype=object)


def test_fill_result_with_nans():
    arr = [np.array([0, 1]), np.array([0]), np.array([0, 1, 2, 3])]

    left1 = np.array([[0, 1, -1, -1], [0, -1, -1, -1], [0, 1, 2, 3]], dtype=int)
    left2 = np.array(
        [[0, 1, np.nan, np.nan], [0, np.nan, np.nan, np.nan], [0, 1, 2, 3]], dtype=float
    )

    right1, right2 = _fill_result_with_nans(arr, arr)

    npt.assert_array_equal(left1, right1)
    npt.assert_almost_equal(left2, right2)


def test_motifs_one_motif():
    # The top motif for m=3 is a [0 1 0] at indices 0 and 5
    T = np.array([0.0, 1.0, 0.0, -1.0, -1.0, 0.0, 1.0, 0.0, -0.5])
    m = 3
    k = 1

    left_indices = [[0, 5]]
    left_profile_values = [[0.0, 0.0]]

    P = naive.stump(T, m)
    right_indices, right_distance_values = motifs(T, P[:, 0], k=k, atol=0.001)

    npt.assert_array_equal(left_indices, right_indices)
    npt.assert_almost_equal(left_profile_values, right_distance_values, decimal=4)


def test_motifs_two_motifs():
    # The time series is random noise with two motifs for m=10:
    # * (almost) identical step functions at indices 10, 110 and 210
    # * identical linear slopes at indices 70 and 170
    T = np.random.normal(size=300)
    m = 20

    T[10:30] = 1
    T[12:28] = 2

    T[110:130] = 1
    T[112:128] = 2
    T[120] = 2.2

    T[210:230] = 1
    T[212:228] = 2
    T[220] = 1.9
    # naive.distance(naive.z_norm(T[10:30]), naive.z_norm(T[110:130])) = 0.47
    # naive.distance(naive.z_norm(T[10:30]), naive.z_norm(T[210:230])) = 0.24
    # naive.distance(naive.z_norm(T[110:130]), naive.z_norm(T[210:230])) = 0.72
    # Hence T[10:30] is the motif representative for this motif

    T[70:90] = np.arange(m) * 0.1
    T[170:190] = np.arange(m) * 0.1
    # naive.distance(naive.z_norm(T[70:90]), naive.z_norm(T[170:190])) = 0.0

    k = 2

    P = naive.stump(T, m)

    # left_indices = [[70, 170, -1], [10, 210, 110]]
    left_profile_values = [
        [0.0, 0.0, np.nan],
        [
            0.0,
            naive.distance(core.z_norm(T[10:30]), core.z_norm(T[210:230])),
            naive.distance(core.z_norm(T[10:30]), core.z_norm(T[110:130])),
        ],
    ]

    right_indices, right_distance_values = motifs(T, P[:, 0], k=k)

    # We ignore indices because of sorting ambiguities for equal distances.
    # As long as the distances are correct, the indices will be too.
    npt.assert_almost_equal(left_profile_values, right_distance_values, decimal=6)


def test_naive_occurrences_exact():
    # The query can be found as a perfect match two times in the time series
    T = np.array([0.0, 1.0, 0.0, -1.0, -1.0, 0.0, 1.0, 0.0, -0.5])
    Q = np.array([0.0, 1.0, 0.0])
    m = Q.shape[0]
    excl_zone = int(np.ceil(m / 4))

    left = [[0, 0], [5, 0]]
    right = naive_occurrences(
        Q, T, excl_zone=excl_zone, profile_value=0.0, atol=0.001, rtol=0.001
    )

    npt.assert_almost_equal(left, right)


def test_naive_occurrences_exclusion_zone():
    # The query appears as a perfect match at location 1 and as very close matches
    # (z-normalized distance of 0.05) at location 0 and 5.
    # However, since we apply an exclusion zone, the match at index 0 is ignored
    T = np.array([0.1, 1.0, 2.0, 3.0, -1.0, 0.1, 1.0, 2.0, -0.5])
    Q = np.array([0.0, 1.0, 2.0])
    m = Q.shape[0]
    excl_zone = int(np.ceil(m / 4))

    left = [[1, 0], [5, naive.distance(core.z_norm(Q), core.z_norm(T[5 : 5 + m]))]]
    right = naive_occurrences(
        Q, T, excl_zone=excl_zone, profile_value=0.0, atol=0.1, rtol=0.001
    )

    npt.assert_almost_equal(left, right)


@pytest.mark.parametrize("Q, T", test_data)
def test_occurrences(Q, T):
    m = Q.shape[0]
    excl_zone = int(np.ceil(m / 4))
    rtol = 1
    atol = 1
    profile_value = 1

    left = naive_occurrences(Q, T, excl_zone, profile_value, atol, rtol)
    right = occurrences(
        Q,
        T,
        excl_zone=excl_zone,
        max_occurrences=None,
        profile_value=profile_value,
        atol=atol,
        rtol=rtol,
    )

    npt.assert_almost_equal(left, right)


"""
These are tests for multidimensional motif discovery, to be ignored for the moment

def test_motifs_multidimensional_one_motif_all_dimensions():
    T = np.array(
        [
            [0.0, 1.0, 0.0, -1.0, -1.0, 0.0, 1.0, 0.0, -0.5],
            [0.0, 1.0, 0.0, -1.0, -1.0, 0.0, 1.0, 0.0, -0.5],
        ]
    )
    m = 3
    k = 1

    left_indices = [[0, 5]]
    left_profile_values = [0]

    P, I = mstump(T, m)
    S = np.array([[0, 0, 0, 0, 0, 0, 0], [1, 1, 1, 1, 1, 1, 1]], dtype=float)
    right_indices, right_profile_values = search.motifs_multidimensional(
        T, P, S, num_dimensions=2, k=k, atol=0.001
    )

    npt.assert_array_equal(left_indices, right_indices)
    npt.assert_almost_equal(left_profile_values, right_profile_values, decimal=4)


def test_motifs_multidimensional_two_motifs_all_dimensions():
    n = 200
    T = np.random.normal(size=(2, n))
    m = 20

    T[:, 10:30] = 1
    T[:, 12:28] = 2
    T[:, 110:130] = 1
    T[:, 112:128] = 2
    T[:, 129] = 1.1

    T[:, 70:90] = np.arange(m) * 0.1
    T[:, 170:190] = np.arange(m) * 0.1

    k = 2

    P, I = mstump(T, m)
    S = np.zeros((2, n - m + 1), dtype=float)
    S[1, :] = 1

    left_indices = [[70, 170], [10, 110]]
    left_profile_values = [P[70, 0], P[10, 0]]

    right_indices, right_profile_values = search.motifs_multidimensional(
        T, P, S, num_dimensions=2, k=k
    )
    right_indices = np.sort(right_indices, axis=1)

    npt.assert_array_equal(left_indices, right_indices)
    npt.assert_almost_equal(left_profile_values, right_profile_values, decimal=4)
"""
