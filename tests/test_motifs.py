import functools

import naive
import numpy as np
import numpy.testing as npt
import pytest

from stumpy import core, match, motifs


def naive_motifs(T, m, max_motifs, max_matches, T_subseq_isconstant=None):
    # To avoid complexity, this naive function is written
    # such that each array in the ouput has shape
    # (max_motif, max_matches).

    # To this end, the following items are considered:
    # 1. `max_distance` and `cutoff` are both hardcoded and
    # set to np.inf
    # 2. If the number of subsequence, i.e. `len(T)-m+1`, is
    # not less than `m * max_motifs * max_matches`, then the
    # output definitely has the shape (max_motif, max_matches).

    l = len(T) - m + 1
    excl_zone = int(np.ceil(m / 4))
    T_subseq_isconstant = naive.rolling_isconstant(T, m, T_subseq_isconstant)

    output_shape = (max_motifs, max_matches)
    motif_distances = np.full(output_shape, -np.inf, dtype=np.float64)
    motif_indices = np.full(output_shape, -1, dtype=np.int64)

    D = naive.distance_matrix(T, T, m)
    D[np.isnan(D)] = np.inf
    for i in range(D.shape[0]):
        for j in range(D.shape[1]):
            if np.isfinite(D[i, j]):
                if T_subseq_isconstant[i] and T_subseq_isconstant[j]:
                    D[i, j] = 0.0
                elif T_subseq_isconstant[i] or T_subseq_isconstant[j]:
                    D[i, j] = np.sqrt(m)
                else:  # pragma: no cover
                    pass

    for i in range(D.shape[0]):
        naive.apply_exclusion_zone(D[i], i, excl_zone, np.inf)

    P = np.min(D, axis=1)
    for i in range(max_motifs):
        distances = []
        indices = []

        idx = np.argmin(P)

        # self match
        distances.append(0)
        indices.append(idx)
        naive.apply_exclusion_zone(P, idx, excl_zone, np.inf)

        # Explore distance profile D[idx] till `max_matches` are found.
        naive.apply_exclusion_zone(D[idx], idx, excl_zone, np.inf)
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

            # Note that a discovered match cannot be selected as motif but
            # it can still be selected again as a match for another motif.

        motif_distances[i] = distances
        motif_indices[i] = indices

    return motif_distances, motif_indices


def naive_multi_match(
    Q,
    T,
    excl_zone,
    max_distance,
    max_matches=None,
    T_subseq_isconstant=None,
    Q_subseq_isconstant=None,
):
    m = Q.shape[-1]
    T_subseq_isconstant = naive.rolling_isconstant(T, m, T_subseq_isconstant)
    Q_subseq_isconstant = naive.rolling_isconstant(Q, m, Q_subseq_isconstant)

    d, n = T.shape
    D_total = np.zeros(n - m + 1, np.float64)
    for i in range(d):
        D = naive.distance_profile(Q[i], T[i], m)
        D[np.isnan(D)] = np.inf
        for j in range(len(D)):
            if np.isfinite(D[j]):
                if T_subseq_isconstant[i, j] and Q_subseq_isconstant[i]:
                    D[j] = 0
                elif T_subseq_isconstant[i, j] or Q_subseq_isconstant[i]:
                    D[j] = np.sqrt(m)
                else:  # pragma: no cover
                    pass
        D_total[:] = D_total + D

    D_mean = D_total / d

    return naive.find_matches(D_mean, excl_zone, max_distance, max_matches)


def naive_match(
    Q,
    T,
    excl_zone,
    max_distance,
    max_matches=None,
    T_subseq_isconstant=None,
    Q_subseq_isconstant=None,
):
    # Q_subseq_isconstant is a boolean array of size 1
    # this does not support multi-dim T, and Q

    m = Q.shape[0]
    T_subseq_isconstant = naive.rolling_isconstant(T, m, T_subseq_isconstant)
    Q_subseq_isconstant = naive.rolling_isconstant(Q, m, Q_subseq_isconstant)[0]

    D = naive.distance_profile(Q, T, m)
    D[np.isnan(D)] = np.inf
    for i in range(len(D)):
        if np.isfinite(D[i]):
            if T_subseq_isconstant[i] and Q_subseq_isconstant:
                D[i] = 0
            elif T_subseq_isconstant[i] or Q_subseq_isconstant:
                D[i] = np.sqrt(m)
            else:  # pragma: no cover
                pass

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
    npt.assert_almost_equal(left_profile_values, right_distance_values)


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
    npt.assert_almost_equal(left_profile_values, right_distance_values)

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
    npt.assert_almost_equal(left_profile_values, right_distance_values)


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
    npt.assert_almost_equal(left_profile_values, right_distance_values)


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

    T_subseq_isconstant = functools.partial(
        naive.isconstant_func_stddev_threshold, quantile_threshold=0.05
    )

    left = naive_match(
        Q,
        T,
        excl_zone,
        max_distance=max_distance,
        T_subseq_isconstant=T_subseq_isconstant,
    )

    right = match(
        Q,
        T,
        max_matches=None,
        max_distance=lambda D: max_distance,  # also test lambda functionality
        T_subseq_isconstant=T_subseq_isconstant,
    )

    npt.assert_almost_equal(left, right)

    # Test for when Q is constant
    Q_subseq_isconstant = np.array([True])

    left = naive_match(
        Q,
        T,
        excl_zone,
        max_distance=max_distance,
        T_subseq_isconstant=T_subseq_isconstant,
        Q_subseq_isconstant=Q_subseq_isconstant,
    )

    right = match(
        Q,
        T,
        max_matches=None,
        max_distance=lambda D: max_distance,  # also test lambda functionality
        T_subseq_isconstant=T_subseq_isconstant,
        Q_subseq_isconstant=Q_subseq_isconstant,
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


def test_multi_match():
    T = np.random.uniform(-1000, 1000, size=(2, 64))
    Q = np.random.uniform(-1000, 1000, size=(2, 64))

    m = Q.shape[-1]
    excl_zone = int(np.ceil(m / 4))
    max_distance = 0.3

    left = naive_multi_match(
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


def test_multi_match_isconstant():
    T = np.random.rand(2, 64)
    Q = np.random.rand(2, 8)

    m = Q.shape[-1]
    excl_zone = int(np.ceil(m / 4))
    max_distance = 0.3

    T_subseq_isconstant = functools.partial(
        naive.isconstant_func_stddev_threshold, quantile_threshold=0.05
    )

    Q_subseq_isconstant = np.array(
        [
            [True],
            [False],
        ]
    )

    left = naive_multi_match(
        Q,
        T,
        excl_zone,
        max_distance=max_distance,
        T_subseq_isconstant=T_subseq_isconstant,
        Q_subseq_isconstant=Q_subseq_isconstant,
    )

    right = match(
        Q,
        T,
        max_matches=None,
        max_distance=lambda D: max_distance,  # also test lambda functionality
        T_subseq_isconstant=T_subseq_isconstant,
        Q_subseq_isconstant=Q_subseq_isconstant,
    )

    npt.assert_almost_equal(left, right)


def test_motifs():
    T = np.random.rand(64)
    m = 3

    max_motifs = 3
    max_matches = 4
    max_distance = np.inf
    cutoff = np.inf

    # naive
    # `max_distance` and `cutoff` are hard-coded, and set to np.inf.
    ref_distances, ref_indices = naive_motifs(T, m, max_motifs, max_matches)

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


def test_motifs_with_isconstant():
    isconstant_custom_func = functools.partial(
        naive.isconstant_func_stddev_threshold, quantile_threshold=0.05
    )

    T = np.random.rand(64)
    m = 3

    max_motifs = 3
    max_matches = 4
    max_distance = np.inf
    cutoff = np.inf

    # naive
    # `max_distance` and `cutoff` are hard-coded, and set to np.inf.
    ref_distances, ref_indices = naive_motifs(
        T, m, max_motifs, max_matches, T_subseq_isconstant=isconstant_custom_func
    )

    # performant
    mp = naive.stump(T, m, row_wise=True, T_A_subseq_isconstant=isconstant_custom_func)
    comp_distance, comp_indices = motifs(
        T,
        mp[:, 0].astype(np.float64),
        min_neighbors=1,
        max_distance=max_distance,
        cutoff=cutoff,
        max_matches=max_matches,
        max_motifs=max_motifs,
        T_subseq_isconstant=isconstant_custom_func,
    )

    npt.assert_almost_equal(ref_distances, comp_distance)
    npt.assert_almost_equal(ref_indices, comp_indices)


def test_motifs_with_max_matches_none():
    T = np.random.rand(16)
    m = 3

    max_motifs = 1
    max_matches = None
    max_distance = np.inf
    cutoff = np.inf

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

    ref_len = len(T) - m + 1

    npt.assert_(ref_len >= comp_distance.shape[1])
    npt.assert_(ref_len >= comp_indices.shape[1])
