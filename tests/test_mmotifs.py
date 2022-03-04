import numpy as np
import numpy.testing as npt

from stumpy.mmotifs import mmotifs
from stumpy.mstump import mstump


# These are tests for multidimensional motif discovery


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


def test_motifs_multidimensional_with_deafault_parameters():
    # Find the motif pair while only setting the default parameters

    # Arrange
    motif_distances_expected = np.array([[0.0, 0.0]])
    motif_indices_expected = np.array([[2, 12]])
    motif_subspaces_expected = [np.array([3])]
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
    motif_distances, motif_indices, motif_subspaces, motif_mdls = mmotifs(T, P, I)

    # Assert
    npt.assert_array_almost_equal(motif_distances_expected, motif_distances)
    npt.assert_array_almost_equal(motif_indices_expected, motif_indices)
    npt.assert_array_almost_equal(motif_subspaces_expected, motif_subspaces)
    npt.assert_array_almost_equal(motif_mdls_expected, motif_mdls)


def test_motifs_multidimensional_two_motifs_all_dimensions():
    # Find the best two motif pairs

    # Arrange
    motif_distances_expected = np.array([[0.0, 0.0], [0.0, 0.0]])
    motif_indices_expected = np.array([[2, 12], [8, 11]])
    motif_subspaces_expected = [np.array([3]), np.array([1])]
    motif_mdls_expected = [
        np.array([176.0, 177.509775, 191.26466251, 235.01955001]),
        np.array([176.0, 177.509775, 191.26466251, 235.01955001]),
    ]

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
        T, P, I, cutoffs=np.inf, max_motifs=2
    )

    # Assert
    npt.assert_array_almost_equal(motif_distances_expected, motif_distances)
    npt.assert_array_almost_equal(motif_indices_expected, motif_indices)
    npt.assert_array_almost_equal(motif_subspaces_expected, motif_subspaces)
    npt.assert_array_almost_equal(motif_mdls_expected, motif_mdls)
