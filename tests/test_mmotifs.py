import numpy as np
import numpy.testing as npt
import pytest

from stumpy.mmotifs import mmotifs
from stumpy.mstump import mstump
from stumpy import match

import naive


# These are tests for multidimensional motif discovery


def naive_match(Q, T, excl_zone, max_distance):
    m = Q.shape[0]
    D = naive.distance_profile(Q, T, m)

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


def test_motifs_multidimensional_one_motif_all_dimensions():
    # Find the two dimensional motif pair

    # Arrange
    motif_distances_expected = np.array([[0.0, 0.04540775]])
    motif_indices_expected = np.array([[3, 10]])
    motif_subspaces_expected = [np.array([1, 3])]
    motif_mdls_expected = [np.array([176.0, 177.509775, 191.26466251, 235.01955001])]

    # Act
    T = np.array(
        [
            [5.0, 0.0, 3.0, 3.0, 7.0, 9.0, 3.0, 5.0, 2.0, 4.0, 7.0, 6.0, 8.0, 8.0, 1.0],
            [
                7.0,
                3.0,
                5.0,
                9.0,
                8.0,
                7.0,
                4.0,
                5.0,
                10.0,
                8.0,
                4.0,
                3.0,
                2.0,
                0.0,
                1.0,
            ],
            [6.0, 7.0, 7.0, 8.0, 1.0, 5.0, 9.0, 8.0, 9.0, 4.0, 3.0, 0.0, 3.0, 5.0, 0.0],
            [
                0.0,
                1.0,
                3.0,
                2.0,
                6.0,
                1.0,
                9.0,
                10.0,
                1.0,
                2.0,
                2.0,
                5.0,
                1.0,
                0.0,
                4.0,
            ],
        ]
    )
    m = 3

    P, I = mstump(T, m)
    motif_distances, motif_indices, motif_subspaces, motif_mdls = mmotifs(
        T, P, I, max_distance=np.inf, max_matches=2, k=1
    )

    # Assert
    npt.assert_array_almost_equal(motif_distances_expected, motif_distances)
    npt.assert_array_almost_equal(motif_indices_expected, motif_indices)
    npt.assert_array_almost_equal(motif_subspaces_expected, motif_subspaces)
    npt.assert_array_almost_equal(motif_mdls_expected, motif_mdls)


# def test_motifs_multidimensional_two_motifs_all_dimensions():


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
