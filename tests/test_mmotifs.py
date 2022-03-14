import numpy as np
import numpy.testing as npt
import naive

from stumpy.mmotifs import mmotifs
from stumpy import config


def test_motifs_multidimensional_with_deafault_parameters():
    # Find the motif pair while only setting the default parameters

    motif_distances_expected = np.array([[0.0000000e00, 1.1151008e-07]])
    motif_indices_expected = np.array([[2, 9]])
    motif_subspaces_expected = [np.array([1])]
    motif_mdls_expected = [np.array([232.0, 250.57542476, 260.0, 271.3509059])]

    T = np.array(
        [
            [5.2, 0.1, 3.5, 3.4, 7.1, 9.8, 3.7, 5.0, 2.1, 4.3, 7.5, 6.8, 8.0, 8.1, 1.2],
            [
                7.3,
                3.2,
                5.0,
                9.1,
                8.2,
                7.3,
                4.8,
                8.2,
                10.0,
                0.0,
                4.1,
                3.2,
                2.3,
                0.1,
                1.4,
            ],
            [6.2, 7.6, 7.6, 8.4, 1.1, 5.9, 9.2, 8.5, 9.3, 4.6, 3.5, 0.0, 3.1, 5.3, 0.9],
            [
                0.1,
                1.3,
                3.0,
                2.1,
                6.2,
                1.3,
                9.5,
                10.0,
                1.8,
                2.0,
                2.1,
                5.2,
                1.3,
                0.5,
                4.3,
            ],
        ]
    )
    m = 4

    excl_zone = int(np.ceil(m / config.STUMPY_EXCL_ZONE_DENOM))
    P, I = naive.mstump(T, m, excl_zone)
    motif_distances, motif_indices, motif_subspaces, motif_mdls = mmotifs(T, P, I)

    npt.assert_array_almost_equal(motif_distances_expected, motif_distances)
    npt.assert_array_almost_equal(motif_indices_expected, motif_indices)
    npt.assert_array_almost_equal(motif_subspaces_expected, motif_subspaces)
    npt.assert_array_almost_equal(motif_mdls_expected, motif_mdls)


def test_motifs_multidimensional_one_motif_max_matches_none():
    # Find the best multidimensional motif when max_distance is None

    motif_distances_expected = np.array([[0.0000000e00, 1.1151008e-07]])
    motif_indices_expected = np.array([[2, 9]])
    motif_subspaces_expected = [np.array([1])]
    motif_mdls_expected = [np.array([232.0, 250.57542476, 260.0, 271.3509059])]

    T = np.array(
        [
            [5.2, 0.1, 3.5, 3.4, 7.1, 9.8, 3.7, 5.0, 2.1, 4.3, 7.5, 6.8, 8.0, 8.1, 1.2],
            [
                7.3,
                3.2,
                5.0,
                9.1,
                8.2,
                7.3,
                4.8,
                8.2,
                10.0,
                0.0,
                4.1,
                3.2,
                2.3,
                0.1,
                1.4,
            ],
            [6.2, 7.6, 7.6, 8.4, 1.1, 5.9, 9.2, 8.5, 9.3, 4.6, 3.5, 0.0, 3.1, 5.3, 0.9],
            [
                0.1,
                1.3,
                3.0,
                2.1,
                6.2,
                1.3,
                9.5,
                10.0,
                1.8,
                2.0,
                2.1,
                5.2,
                1.3,
                0.5,
                4.3,
            ],
        ]
    )
    m = 4

    excl_zone = int(np.ceil(m / config.STUMPY_EXCL_ZONE_DENOM))
    P, I = naive.mstump(T, m, excl_zone)
    motif_distances, motif_indices, motif_subspaces, motif_mdls = mmotifs(
        T, P, I, max_matches=None
    )

    npt.assert_array_almost_equal(motif_distances_expected, motif_distances)
    npt.assert_array_almost_equal(motif_indices_expected, motif_indices)
    npt.assert_array_almost_equal(motif_subspaces_expected, motif_subspaces)
    npt.assert_array_almost_equal(motif_mdls_expected, motif_mdls)


def test_motifs_multidimensional_one_motif_all_dimensions():
    # Find the two-dimensional motif pair

    motif_distances_expected = np.array([[0.0, 0.20948156]])
    motif_indices_expected = np.array([[2, 9]])
    motif_subspaces_expected = [np.array([1, 3])]
    motif_mdls_expected = [np.array([232.0, 250.57542476, 260.0, 271.3509059])]

    T = np.array(
        [
            [5.2, 0.1, 3.5, 3.4, 7.1, 9.8, 3.7, 5.0, 2.1, 4.3, 7.5, 6.8, 8.0, 8.1, 1.2],
            [
                7.3,
                3.2,
                5.0,
                9.1,
                8.2,
                7.3,
                4.8,
                8.2,
                10.0,
                0.0,
                4.1,
                3.2,
                2.3,
                0.1,
                1.4,
            ],
            [6.2, 7.6, 7.6, 8.4, 1.1, 5.9, 9.2, 8.5, 9.3, 4.6, 3.5, 0.0, 3.1, 5.3, 0.9],
            [
                0.1,
                1.3,
                3.0,
                2.1,
                6.2,
                1.3,
                9.5,
                10.0,
                1.8,
                2.0,
                2.1,
                5.2,
                1.3,
                0.5,
                4.3,
            ],
        ]
    )
    m = 4

    excl_zone = int(np.ceil(m / config.STUMPY_EXCL_ZONE_DENOM))
    P, I = naive.mstump(T, m, excl_zone)
    motif_distances, motif_indices, motif_subspaces, motif_mdls = mmotifs(
        T, P, I, max_distance=np.inf, max_matches=2, k=1
    )

    npt.assert_array_almost_equal(motif_distances_expected, motif_distances)
    npt.assert_array_almost_equal(motif_indices_expected, motif_indices)
    npt.assert_array_almost_equal(motif_subspaces_expected, motif_subspaces)
    npt.assert_array_almost_equal(motif_mdls_expected, motif_mdls)


def test_motifs_multidimensional_more_motifs_when_cutoffs_is_set():
    # Find the best multidimensional motif pairs if cutoffs is set
    # Only one pair here since 'max_motifs' is set per default

    motif_distances_expected = np.array([[0.0000000e00, 1.1151008e-07]])
    motif_indices_expected = np.array([[2, 9]])
    motif_subspaces_expected = [np.array([1])]
    motif_mdls_expected = [np.array([232.0, 250.57542476, 260.0, 271.3509059])]

    T = np.array(
        [
            [5.2, 0.1, 3.5, 3.4, 7.1, 9.8, 3.7, 5.0, 2.1, 4.3, 7.5, 6.8, 8.0, 8.1, 1.2],
            [
                7.3,
                3.2,
                5.0,
                9.1,
                8.2,
                7.3,
                4.8,
                8.2,
                10.0,
                0.0,
                4.1,
                3.2,
                2.3,
                0.1,
                1.4,
            ],
            [6.2, 7.6, 7.6, 8.4, 1.1, 5.9, 9.2, 8.5, 9.3, 4.6, 3.5, 0.0, 3.1, 5.3, 0.9],
            [
                0.1,
                1.3,
                3.0,
                2.1,
                6.2,
                1.3,
                9.5,
                10.0,
                1.8,
                2.0,
                2.1,
                5.2,
                1.3,
                0.5,
                4.3,
            ],
        ]
    )
    m = 4

    excl_zone = int(np.ceil(m / config.STUMPY_EXCL_ZONE_DENOM))
    P, I = naive.mstump(T, m, excl_zone)
    motif_distances, motif_indices, motif_subspaces, motif_mdls = mmotifs(
        T, P, I, cutoffs=3, max_motifs=10
    )

    npt.assert_array_almost_equal(motif_distances_expected, motif_distances)
    npt.assert_array_almost_equal(motif_indices_expected, motif_indices)
    npt.assert_array_almost_equal(motif_subspaces_expected, motif_subspaces)
    npt.assert_array_almost_equal(motif_mdls_expected, motif_mdls)


def test_motifs_multidimensional_two_motifs_all_dimensions():
    # Find the best two motif pairs

    motif_distances_expected = np.array(
        [[0.00000000e00, 1.11510080e-07], [1.68587394e-07, 2.58694429e-01]]
    )
    motif_indices_expected = np.array([[2, 9], [6, 1]])
    motif_subspaces_expected = [np.array([1]), np.array([2])]
    motif_mdls_expected = [
        np.array([232.0, 250.57542476, 260.0, 271.3509059]),
        np.array([264.0, 280.0, 299.01955001, 310.51024953]),
    ]

    T = np.array(
        [
            [5.2, 0.1, 3.5, 3.4, 7.1, 9.8, 3.7, 5.0, 2.1, 4.3, 7.5, 6.8, 8.0, 8.1, 1.2],
            [
                7.3,
                3.2,
                5.0,
                9.1,
                8.2,
                7.3,
                4.8,
                8.2,
                10.0,
                0.0,
                4.1,
                3.2,
                2.3,
                0.1,
                1.4,
            ],
            [6.2, 7.6, 7.6, 8.4, 1.1, 5.9, 9.2, 8.5, 9.3, 4.6, 3.5, 0.0, 3.1, 5.3, 0.9],
            [
                0.1,
                1.3,
                3.0,
                2.1,
                6.2,
                1.3,
                9.5,
                10.0,
                1.8,
                2.0,
                2.1,
                5.2,
                1.3,
                0.5,
                4.3,
            ],
        ]
    )
    m = 4

    excl_zone = int(np.ceil(m / config.STUMPY_EXCL_ZONE_DENOM))
    P, I = naive.mstump(T, m, excl_zone)
    motif_distances, motif_indices, motif_subspaces, motif_mdls = mmotifs(
        T, P, I, cutoffs=np.inf, max_motifs=2, max_distance=np.inf, max_matches=2
    )

    npt.assert_array_almost_equal(motif_distances_expected, motif_distances)
    npt.assert_array_almost_equal(motif_indices_expected, motif_indices)
    npt.assert_array_almost_equal(motif_subspaces_expected, motif_subspaces)
    npt.assert_array_almost_equal(motif_mdls_expected, motif_mdls)
