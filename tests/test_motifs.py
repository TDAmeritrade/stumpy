import numpy as np
import numpy.testing as npt
import pytest

from stumpy import core, motifs, match

from stumpy.motifs import _jagged_list_to_array

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
normalized = [True, False]


def naive_match(Q, T, excl_zone, max_distance, normalize):
    m = Q.shape[0]
    if normalize:
        D = naive.distance_profile(Q, T, m)
    else:
        D = naive.aamp_distance_profile(Q, T, m)

    # Finds all indices that have a lower distance profile value `D`
    # than `atol + rtol * D`
    matches = []
    for i in range(D.size):
        dist = D[i]
        if dist <= max_distance:
            matches.append(i)

    # Removes indices that are inside the exclusion zone of some occurrence with
    # a smaller distance to the query
    matches.sort(key=lambda x: D[x])
    result = []
    while len(matches) > 0:
        o = matches[0]
        result.append([D[o], o])
        matches = [x for x in matches if x < o - excl_zone or x > o + excl_zone]

    return np.array(result, dtype=object)


def test_jagged_list_to_array():
    arr = [np.array([0, 1]), np.array([0]), np.array([0, 1, 2, 3])]

    left = np.array([[0, 1, -1, -1], [0, -1, -1, -1], [0, 1, 2, 3]], dtype="int64")
    right = _jagged_list_to_array(arr, fill_value=-1, dtype="int64")
    npt.assert_array_equal(left, right)

    left = np.array(
        [[0, 1, np.nan, np.nan], [0, np.nan, np.nan, np.nan], [0, 1, 2, 3]],
        dtype="float64",
    )
    right = _jagged_list_to_array(arr, fill_value=np.nan, dtype="float64")
    npt.assert_array_equal(left, right)


def test_jagged_list_to_array_empty():
    arr = []

    left = np.array([[]], dtype="int64")
    right = _jagged_list_to_array(arr, fill_value=-1, dtype="int64")
    npt.assert_array_equal(left, right)

    left = np.array([[]], dtype="float64")
    right = _jagged_list_to_array(arr, fill_value=np.nan, dtype="float64")
    npt.assert_array_equal(left, right)


def test_motifs_one_motif():
    # The top motif for m=3 is a [0 1 0] at indices 0, 5 and 9
    T = np.array([0.0, 1.0, 0.0, -1.0, -1.0, 0.0, 1.0, 0.0, -0.5, 2.0, 3.0, 2.0])
    m = 3
    max_motifs = 1

    left_indices = [[0, 5, 9]]
    left_profile_values = [[0.0, 0.0, 0.0]]

    P = naive.stump(T, m)
    right_distance_values, right_indices = motifs(
        T,
        P[:, 0],
        max_distance=lambda D: 0.001,  # Also test lambda functionality
        max_motifs=max_motifs,
        cutoff=np.inf,
        normalize=True,
    )

    npt.assert_array_equal(left_indices, right_indices)
    npt.assert_almost_equal(left_profile_values, right_distance_values, decimal=4)


def test_motifs_two_motifs():
    # Fix seed, because in some case motifs can be of by on index resulting in test
    # fails, which is caused since one of the motifs is not repeaded perfectly in T.
    np.random.seed(1234)

    # The time series is random noise with two motifs for m=10:
    # * (almost) identical step functions at indices 10, 110 and 210
    # * identical linear slopes at indices 70 and 170
    T = np.random.normal(size=300)
    m = 20

    T[10:30] = 1
    T[12:28] = 2

    T[110:130] = 3
    T[112:128] = 6
    T[120] = 6.6

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

    max_motifs = 2

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

    right_distance_values, right_indices = motifs(
        T,
        P[:, 0],
        max_motifs=max_motifs,
        max_distance=0.5,
        cutoff=np.inf,
        normalize=True,
    )

    # We ignore indices because of sorting ambiguities for equal distances.
    # As long as the distances are correct, the indices will be too.
    npt.assert_almost_equal(left_profile_values, right_distance_values, decimal=6)

    # Reset seed
    np.random.seed(None)


def test_motifs_max_matches():
    # This test covers the following:

    # A time series contains motif A at four locations and motif B at two.
    # If `max_motifs=2` the result should contain only the top two matches of motif A
    # and the top two matches of motif B as two separate motifs.
    T = np.array(
        [
            0.0,  # motif A
            1.0,
            0.0,
            2.3,
            -1.0,  # motif B
            -1.0,
            -2.0,
            0.0,  # motif A
            1.0,
            0.0,
            -2.0,
            -1.0,  # motif B
            -1.03,
            -2.0,
            -0.5,
            2.0,  # motif A
            3.0,
            2.04,
            2.3,
            2.0,  # motif A
            3.0,
            2.02,
        ]
    )
    m = 3
    max_motifs = 3

    left_indices = [[0, 7], [4, 11]]
    left_profile_values = [
        [0.0, 0.0],
        [
            0.0,
            naive.distance(
                core.z_norm(T[left_indices[1][0] : left_indices[1][0] + m]),
                core.z_norm(T[left_indices[1][1] : left_indices[1][1] + m]),
            ),
        ],
    ]

    P = naive.stump(T, m)
    right_distance_values, right_indices = motifs(
        T,
        P[:, 0],
        max_motifs=max_motifs,
        max_distance=0.1,
        cutoff=np.inf,
        normalize=True,
        max_matches=2,
    )

    # We ignore indices because of sorting ambiguities for equal distances.
    # As long as the distances are correct, the indices will be too.
    npt.assert_almost_equal(left_profile_values, right_distance_values, decimal=4)


def test_motifs_one_motif_aamp():
    # The top motif for m=3 is a [0 1 0] at indices 0 and 5, while the occurrence
    # at index 9 is not a motif in the aamp case.
    T = np.array([0.0, 1.0, 0.0, -1.0, -1.0, 0.0, 1.0, 0.0, -0.5, 2.0, 3.0, 2.0])
    m = 3
    max_motifs = 1

    left_indices = [[0, 5]]
    left_profile_values = [[0.0, 0.0]]

    P = naive.stump(T, m)
    right_distance_values, right_indices = motifs(
        T,
        P[:, 0],
        max_motifs=max_motifs,
        max_distance=0.001,
        cutoff=np.inf,
        normalize=False,
    )

    npt.assert_array_equal(left_indices, right_indices)
    npt.assert_almost_equal(left_profile_values, right_distance_values, decimal=4)


def test_motifs_two_motifs_aamp():
    # Fix seed, because in some case motifs can be of by on index resulting in test
    # fails, which is caused since one of the motifs is not repeaded perfectly in T.
    np.random.seed(1234)

    # The time series is random noise with two motifs for m=10:
    # * (almost) identical step functions at indices 10, 110 and 210
    # * identical linear slopes at indices 70 and 170
    T = np.random.normal(size=300)
    m = 20

    T[10:30] = 1
    T[12:28] = 2

    # This is not part of the motif in the aamp case
    T[110:130] = 3
    T[112:128] = 6
    T[120] = 6.6

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

    max_motifs = 2

    P = naive.stump(T, m)

    # left_indices = [[70, 170], [10, 210]]
    left_profile_values = [
        [0.0, 0.0],
        [
            0.0,
            naive.distance(T[10:30], T[210:230]),
        ],
    ]

    right_distance_values, right_indices = motifs(
        T,
        P[:, 0],
        max_motifs=max_motifs,
        max_distance=0.5,
        cutoff=np.inf,
        normalize=False,
    )

    # We ignore indices because of sorting ambiguities for equal distances.
    # As long as the distances are correct, the indices will be too.
    npt.assert_almost_equal(left_profile_values, right_distance_values, decimal=6)

    # Reset seed
    np.random.seed(None)


def test_naive_matches_exact():
    # The query can be found as a perfect match two times in the time series
    T = np.array([0.0, 1.0, 0.0, -1.0, -1.0, 0.0, 1.0, 0.0, -0.5, 0.0, 2.0, 0, 0])
    Q = np.array([0.0, 1.0, 0.0])
    m = Q.shape[0]
    excl_zone = int(np.ceil(m / 4))

    left = [[0, 0], [0, 5], [0, 9]]
    right = list(
        naive_match(
            Q,
            T,
            excl_zone=excl_zone,
            max_distance=0.001,  # Small max_distance as matches are identical
            normalize=True,
        )
    )
    # To avoid sorting errors we first sort based on disance and then based on indices
    right.sort(key=lambda x: (x[1], x[0]))

    npt.assert_almost_equal(left, right)


def test_naive_match_exact_aamp():
    # The query can be found as a perfect match two times in the time series
    T = np.array([0.0, 1.0, 0.0, -1.0, -1.0, 0.0, 1.0, 0.0, -0.5])
    Q = np.array([0.0, 1.0, 0.0])
    m = Q.shape[0]
    excl_zone = int(np.ceil(m / 4))

    left = [[0, 0], [0, 5]]
    right = list(
        naive_match(
            Q,
            T,
            excl_zone=excl_zone,
            max_distance=0.001,  # Small max_distance as matches are identical
            normalize=False,
        )
    )
    # To avoid sorting errors we first sort based on disance and then based on indices
    right.sort(key=lambda x: (x[1], x[0]))

    npt.assert_almost_equal(left, right)


def test_naive_match_exclusion_zone():
    # The query appears as a perfect match at location 1 and as very close matches
    # (z-normalized distance of 0.05) at location 0, 5 and 9.
    # However, since we apply an exclusion zone, the match at index 0 is ignored
    T = np.array([0.1, 1.0, 2.0, 3.0, -1.0, 0.1, 1.0, 2.0, -0.5, 0.2, 2.0, 4.0])
    Q = np.array([0.0, 1.0, 2.0])
    m = Q.shape[0]
    excl_zone = int(np.ceil(m / 4))

    left = [
        [0, 1],
        [naive.distance(core.z_norm(Q), core.z_norm(T[5 : 5 + m])), 5],
        [naive.distance(core.z_norm(Q), core.z_norm(T[9 : 9 + m])), 9],
    ]
    right = list(
        naive_match(
            Q,
            T,
            excl_zone=excl_zone,
            max_distance=0.1,
            normalize=True,
        )
    )
    # To avoid sorting errors we first sort based on disance and then based on indices
    right.sort(key=lambda x: (x[1], x[0]))

    npt.assert_almost_equal(left, right)


def test_naive_match_exclusion_zone_aamp():
    # The query appears as a perfect match at location 1 and as very close matches
    # (z-normalized distance of 0.05) at location 0 and 7 (at index 11, the query is
    # not matched in the aamp case).
    # However, since we apply an exclusion zone, the match at index 0 is ignored
    T = np.array(
        [0.1, 1.0, 2.0, 0.0, 1.0, 2.0, -1.0, 0.1, 1.0, 2.0, -0.5, 0.2, 2.0, 4.0]
    )
    Q = np.array([0.0, 1.0, 2.0])
    m = Q.shape[0]
    # Extra large exclusion zone to exclude the first almost perfect match
    excl_zone = m

    left = [
        [0, 3],
        [naive.distance(Q, T[7 : 7 + m]), 7],
    ]
    right = list(
        naive_match(
            Q,
            T,
            excl_zone=excl_zone,
            max_distance=0.2,
            normalize=False,
        )
    )
    # To avoid sorting errors we first sort based on disance and then based on indices
    right.sort(key=lambda x: (x[0], x[1]))

    npt.assert_almost_equal(left, right)


@pytest.mark.parametrize("Q, T", test_data)
def test_match(Q, T):
    m = Q.shape[0]
    excl_zone = int(np.ceil(m / 4))
    max_distance = 0.3

    left = naive_match(
        Q,
        T,
        excl_zone,
        max_distance=max_distance,
        normalize=True,
    )

    right = match(
        Q,
        T,
        excl_zone=excl_zone,
        max_matches=None,
        max_distance=lambda D: max_distance,  # also test lambda functionality
        normalize=True,
    )

    npt.assert_almost_equal(left, right)


@pytest.mark.parametrize("Q, T", test_data)
def test_match_aamp(Q, T):
    m = Q.shape[0]
    excl_zone = int(np.ceil(m / 4))
    max_distance = 0.3

    left = naive_match(
        Q,
        T,
        excl_zone,
        max_distance=max_distance,
        normalize=False,
    )

    right = match(
        Q,
        T,
        excl_zone=excl_zone,
        max_matches=None,
        max_distance=max_distance,
        normalize=False,
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
