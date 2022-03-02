import numpy as np
import numpy.testing as npt

from stumpy.mmotifs import mmotifs
from stumpy.mstump import mstump


# These are tests for non-normalized multidimensional motif discovery


def test_motifs_multidimensional_one_motif_all_dimensions():
    # Find the two dimensional motif pair

    # Arrange
    motif_distances_expected = np.array([[0.0, 1.72474487]])
    motif_indices_expected = np.array([[0, 5]])
    motif_subspaces_expected = [np.array([1, 2])]
    motif_mdls_expected = [np.array([187.0, 188.0, 191.26466251, 196.0])]

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

    P, I = mstump(T, m, normalize=False)
    motif_distances, motif_indices, motif_subspaces, motif_mdls = mmotifs(
        T, P, I, max_distance=np.inf, max_matches=2, k=1, normalize=False
    )

    # Assert
    npt.assert_array_almost_equal(motif_distances_expected, motif_distances)
    npt.assert_array_almost_equal(motif_indices_expected, motif_indices)
    npt.assert_array_almost_equal(motif_subspaces_expected, motif_subspaces)
    npt.assert_array_almost_equal(motif_mdls_expected, motif_mdls)


def test_motifs_multidimensional_with_default_parameters_and_max_distance():
    # Find the motif pair while only setting the default parameters and
    # max_distance

    # Arrange
    motif_distances_expected = np.array(
        [[0.0, 1.0, 2.44948974, 5.74456265, 6.4807407, 7.87400787]]
    )
    motif_indices_expected = np.array([[0, 5, 9, 3, 11, 7]])
    motif_subspaces_expected = [np.array([1])]
    motif_mdls_expected = [np.array([187.0, 188.0, 191.26466251, 196.0])]

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

    P, I = mstump(T, m, normalize=False)
    motif_distances, motif_indices, motif_subspaces, motif_mdls = mmotifs(
        T, P, I, max_distance=np.inf, normalize=False
    )

    # Assert
    npt.assert_array_almost_equal(motif_distances_expected, motif_distances)
    npt.assert_array_almost_equal(motif_indices_expected, motif_indices)
    npt.assert_array_almost_equal(motif_subspaces_expected, motif_subspaces)
    npt.assert_array_almost_equal(motif_mdls_expected, motif_mdls)
