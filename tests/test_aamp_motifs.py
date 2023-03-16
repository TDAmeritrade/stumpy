import naive
import numpy as np
import numpy.testing as npt
import pytest

from stumpy import aamp_match, aamp_motifs, core

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


def naive_aamp_match(Q, T, p, excl_zone, max_distance):
    m = Q.shape[0]
    D = naive.aamp_distance_profile(Q, T, m, p)

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


def test_aamp_motifs_one_motif():
    # The top motif for m=3 is a [0 1 0] at indices 0 and 5, while the occurrence
    # at index 9 is not a motif in the aamp case.
    T = np.array([0.0, 1.0, 0.0, -1.0, -1.0, 0.0, 1.0, 0.0, -0.5, 2.0, 3.0, 2.0])
    m = 3
    max_motifs = 1

    left_indices = [[0, 5]]
    left_profile_values = [[0.0, 0.0]]

    for p in [1.0, 2.0, 3.0]:
        mp = naive.aamp(T, m, p=p)
        right_distance_values, right_indices = aamp_motifs(
            T,
            mp[:, 0],
            max_motifs=max_motifs,
            max_distance=0.001,
            cutoff=np.inf,
            p=p,
        )

        npt.assert_array_equal(left_indices, right_indices)
        npt.assert_almost_equal(left_profile_values, right_distance_values, decimal=4)


def test_aamp_motifs_two_motifs():
    # Fix seed, because in some case motifs can be off by an index resulting in test
    # fails, which is caused since one of the motifs is not repeated perfectly in T.
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

    mp = naive.aamp(T, m)

    # left_indices = [[70, 170], [10, 210]]
    left_profile_values = [
        [0.0, 0.0],
        [
            0.0,
            naive.distance(T[10:30], T[210:230]),
        ],
    ]

    right_distance_values, right_indices = aamp_motifs(
        T,
        mp[:, 0],
        max_motifs=max_motifs,
        max_distance=0.5,
        cutoff=np.inf,
    )

    # We ignore indices because of sorting ambiguities for equal distances.
    # As long as the distances are correct, the indices will be too.
    npt.assert_almost_equal(left_profile_values, right_distance_values, decimal=6)

    # Reset seed
    np.random.seed(None)


def test_aamp_naive_match_exact():
    # The query can be found as a perfect match two times in the time series
    T = np.array([0.0, 1.0, 0.0, -1.0, -1.0, 0.0, 1.0, 0.0, -0.5])
    Q = np.array([0.0, 1.0, 0.0])
    m = Q.shape[0]
    excl_zone = int(np.ceil(m / 4))

    for p in [1.0, 2.0, 3.0]:
        left = [[0, 0], [0, 5]]
        right = list(
            naive_aamp_match(
                Q,
                T,
                p=p,
                excl_zone=excl_zone,
                max_distance=0.001,  # Small max_distance as matches are identical
            )
        )
        # To avoid sorting errors we first sort based on distance and then based on
        # indices
        right.sort(key=lambda x: (x[1], x[0]))

        npt.assert_almost_equal(left, right)


def test_aamp_naive_match_exclusion_zone():
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

    for p in [1.0, 2.0, 3.0]:
        left = [
            [0, 3],
            [naive.distance(Q, T[7 : 7 + m], p=p), 7],
        ]
        right = list(
            naive_aamp_match(
                Q,
                T,
                p=p,
                excl_zone=excl_zone,
                max_distance=0.2,
            )
        )
        # To avoid sorting errors we first sort based on distance and then based on
        # indices
        right.sort(key=lambda x: (x[0], x[1]))

        npt.assert_almost_equal(left, right)


@pytest.mark.parametrize("Q, T", test_data)
def test_aamp_match(Q, T):
    m = Q.shape[0]
    excl_zone = int(np.ceil(m / 4))
    max_distance = 0.3

    for p in [1.0, 2.0, 3.0]:
        left = naive_aamp_match(
            Q,
            T,
            p=p,
            excl_zone=excl_zone,
            max_distance=max_distance,
        )

        right = aamp_match(
            Q,
            T,
            p=p,
            max_matches=None,
            max_distance=max_distance,
        )

        npt.assert_almost_equal(left, right)


@pytest.mark.parametrize("Q, T", test_data)
def test_aamp_match_T_subseq_isfinite(Q, T):
    m = Q.shape[0]
    excl_zone = int(np.ceil(m / 4))
    max_distance = 0.3
    T, T_subseq_isfinite = core.preprocess_non_normalized(T, len(Q))

    for p in [1.0, 2.0, 3.0]:
        left = naive_aamp_match(
            Q,
            T,
            p=p,
            excl_zone=excl_zone,
            max_distance=max_distance,
        )

        right = aamp_match(
            Q,
            T,
            T_subseq_isfinite,
            p=p,
            max_matches=None,
            max_distance=max_distance,
        )

        npt.assert_almost_equal(left, right)
