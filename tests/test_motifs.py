import naive
import numpy as np
import numpy.testing as npt
import pytest

from stumpy import core, match, motifs


def naive_match(Q, T, excl_zone, max_distance, max_matches=None):
    m = Q.shape[0]
    D = naive.distance_profile(Q, T, m)

    return naive.find_matches(D, excl_zone, max_distance, max_matches)


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


def test_motifs_one_motif():
    # The top motif for m=3 is a [0 1 0] at indices 0, 5 and 9
    T = np.array([0.0, 1.0, 0.0, -1.0, -1.0, 0.0, 1.0, 0.0, -0.5, 2.0, 3.0, 2.0])
    m = 3
    max_motifs = 1

    left_indices = [[0, 5, 9]]
    left_profile_values = [[0.0, 0.0, 0.0]]

    mp = naive.stump(T, m)
    right_distance_values, right_indices = motifs(
        T,
        mp[:, 0],
        max_distance=lambda D: 0.001,  # Also test lambda functionality
        max_motifs=max_motifs,
        cutoff=np.inf,
    )

    npt.assert_array_equal(left_indices, right_indices)
    npt.assert_almost_equal(left_profile_values, right_distance_values, decimal=4)


def test_motifs_two_motifs():
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

    mp = naive.stump(T, m)

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


def test_motifs_max_matches():
    # This test covers the following:

    # A time series contains motif A at four locations and motif B at two.
    # If `max_moitf=2` and `max_matches=3`, the result should contain
    # (at most) two sets of motifs and each motif set should contain
    # (at most) the top three matches
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
            3.0,
        ]
    )
    m = 3
    max_motifs = 2
    max_matches = 3

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

    mp = naive.stump(T, m)
    right_distance_values, right_indices = motifs(
        T,
        mp[:, 0],
        max_motifs=max_motifs,
        max_matches=max_matches,
        max_distance=0.05,
        cutoff=np.inf,
    )

    # We ignore indices because of sorting ambiguities for equal distances.
    # As long as the distances are correct, the indices will be too.
    npt.assert_almost_equal(left_profile_values, right_distance_values, decimal=4)


def test_motifs_max_matches_max_distances_inf():
    # This test covers the following:

    # A time series contains motif A at two locations and motif B at two.
    # If `max_moitf=2` and `max_matches=2`, the result should contain
    # (at most) two sets of motifs and each motif set should contain
    # (at most) two matches.
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
            2.0,
            3.0,
            2.04,
            2.3,
            2.0,
            3.0,
            3.0,
        ]
    )
    m = 3
    max_motifs = 2
    max_matches = 2
    max_distance = np.inf

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

    # set `row_wise` to True so that we can compare the indices of motifs as well
    mp = naive.stump(T, m, row_wise=True)
    right_distance_values, right_indices = motifs(
        T,
        mp[:, 0],
        max_motifs=max_motifs,
        max_distance=max_distance,
        cutoff=np.inf,
        max_matches=max_matches,
    )

    npt.assert_almost_equal(left_indices, right_indices)
    npt.assert_almost_equal(left_profile_values, right_distance_values, decimal=4)


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
        )
    )
    # To avoid sorting errors we first sort based on distance and then based on indices
    right.sort(key=lambda x: (x[1], x[0]))

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
    )

    right = match(
        Q,
        T,
        max_matches=None,
        max_distance=lambda D: max_distance,  # also test lambda functionality
    )

    npt.assert_almost_equal(left, right)


@pytest.mark.parametrize("Q, T", test_data)
def test_match_mean_stddev(Q, T):
    m = Q.shape[0]
    excl_zone = int(np.ceil(m / 4))
    max_distance = 0.3

    left = naive_match(
        Q,
        T,
        excl_zone,
        max_distance=max_distance,
    )

    M_T, Σ_T = naive.compute_mean_std(T, len(Q))

    right = match(
        Q,
        T,
        M_T,
        Σ_T,
        max_matches=None,
        max_distance=lambda D: max_distance,  # also test lambda functionality
    )

    npt.assert_almost_equal(left, right)


@pytest.mark.parametrize("Q, T", test_data)
def test_match_isconstant(Q, T):
    m = Q.shape[0]
    excl_zone = int(np.ceil(m / 4))
    max_distance = 0.3

    left = naive_match(
        Q,
        T,
        excl_zone,
        max_distance=max_distance,
    )

    T_subseq_isconstant = naive.rolling_isconstant(T, m)

    right = match(
        Q,
        T,
        max_matches=None,
        max_distance=lambda D: max_distance,  # also test lambda functionality
        T_subseq_isconstant=T_subseq_isconstant,
    )

    npt.assert_almost_equal(left, right)


@pytest.mark.parametrize("Q, T", test_data)
def test_match_mean_stddev_isconstant(Q, T):
    m = Q.shape[0]
    excl_zone = int(np.ceil(m / 4))
    max_distance = 0.3

    left = naive_match(
        Q,
        T,
        excl_zone,
        max_distance=max_distance,
    )

    T_subseq_isconstant = naive.rolling_isconstant(T, m)
    M_T, Σ_T = naive.compute_mean_std(T, len(Q))

    right = match(
        Q,
        T,
        M_T,
        Σ_T,
        max_matches=None,
        max_distance=lambda D: max_distance,  # also test lambda functionality
        T_subseq_isconstant=T_subseq_isconstant,
    )

    npt.assert_almost_equal(left, right)


def test_motif_random_input():
    T = np.random.rand(64)
    m = 3
    excl_zone = int(np.ceil(m / 4))
    l = len(T) - m + 1

    max_motifs = 3
    max_matches = 4
    max_distance = np.inf
    cutoff = np.inf

    # since len(T) is greater than m * max_motifs * max_matches,
    # the shape of output is defiitely `(max_motifs, max_matches)`
    output_shape = (max_motifs, max_matches)

    # naive approach
    ref_distances = np.full(output_shape, np.NINF, dtype=np.float64)
    ref_indices = np.full(output_shape, -1, dtype=np.int64)

    D = naive.distance_matrix(T, T, m)
    np.fill_diagonal(D, val=np.inf)
    P = np.min(D, axis=1)

    for i in range(max_motifs):
        distances = []
        indices = []

        idx = np.argmin(P)
        distances.append(0)  # self match
        indices.append(idx)  # self match
        naive.apply_exclusion_zone(P, idx, excl_zone, np.inf)

        # Explore distance profile D[idx] till `max_matches` are found.
        for _ in range(l):
            if len(distances) >= max_matches:
                break

            nn = np.argmin(D[idx])
            distances.append(D[idx, nn])
            indices.append(nn)

            # Update D[idx] to avoid finding matches that are trivial to
            # each other.
            naive.apply_exclusion_zone(D[idx], nn, excl_zone, np.inf)

            # Update P after the discovery of each match so that the
            # match cannot be selected as the motif next time.
            naive.apply_exclusion_zone(P, nn, excl_zone, np.inf)

            # Note: that a discovered match cannot be selected as motif
            # but it can still be selected again as a match for another
            # motif.

        ref_distances[i] = distances
        ref_indices[i] = indices

    # performant
    mp = naive.stump(T, m, row_wise=True)
    comp_distance, comp_indices = motifs(
        T,
        mp[:, 0].astype(np.float64),
        min_neighbors=1,
        max_distance=max_distance,
        cutoff=cutoff,
        max_matches=max_matches,
        max_motifs=max_motifs,
    )

    npt.assert_almost_equal(ref_indices, comp_indices)
    npt.assert_almost_equal(ref_distances, comp_distance)
