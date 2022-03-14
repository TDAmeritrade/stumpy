import numpy as np
import numpy.testing as npt
import naive

from stumpy.aamp_mmotifs import aamp_mmotifs
from stumpy import config


def test_motifs_multidimensional_with_default_parameters_and_max_distance():
    # Find the motif pair while only setting the default parameters and
    # max_distance

    motif_distances_expected = np.array(
        [[0.0, 1.41421356, 4.46430286, 6.85346628, 8.207923, 8.50529247]]
    )
    motif_indices_expected = np.array([[2, 9, 0, 11, 7, 5]])
    motif_subspaces_expected = [np.array([3])]
    motif_mdls_expected = [np.array([244.0, 260.67970001, 279.86313714, 281.35940001])]

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
    P, I = naive.maamp(T, m, excl_zone)
    motif_distances, motif_indices, motif_subspaces, motif_mdls = aamp_mmotifs(
        T, P, I, max_distance=np.inf
    )

    npt.assert_array_almost_equal(motif_distances_expected, motif_distances)
    npt.assert_array_almost_equal(motif_indices_expected, motif_indices)
    npt.assert_array_almost_equal(motif_subspaces_expected, motif_subspaces)
    npt.assert_array_almost_equal(motif_mdls_expected, motif_mdls)


def test_motifs_multidimensional_with_one_motif_max_matches_none():
    # Find the motif pair while setting 'max_matches=None'

    motif_distances_expected = np.array(
        [[0.0, 1.41421356, 4.46430286, 6.85346628, 8.207923, 8.50529247]]
    )
    motif_indices_expected = np.array([[2, 9, 0, 11, 7, 5]])
    motif_subspaces_expected = [np.array([3])]
    motif_mdls_expected = [np.array([244.0, 260.67970001, 279.86313714, 281.35940001])]

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
    P, I = naive.maamp(T, m, excl_zone)
    motif_distances, motif_indices, motif_subspaces, motif_mdls = aamp_mmotifs(
        T, P, I, max_distance=np.inf, max_matches=None
    )

    npt.assert_array_almost_equal(motif_distances_expected, motif_distances)
    npt.assert_array_almost_equal(motif_indices_expected, motif_indices)
    npt.assert_array_almost_equal(motif_subspaces_expected, motif_subspaces)
    npt.assert_array_almost_equal(motif_mdls_expected, motif_mdls)


def test_motifs_multidimensional_one_motif_all_dimensions():
    # Find the two-dimensional motif pair

    motif_distances_expected = np.array([[0.0, 2.87778559]])
    motif_indices_expected = np.array([[0, 5]])
    motif_subspaces_expected = [np.array([2, 1])]
    motif_mdls_expected = [np.array([244.0, 260.67970001, 279.86313714, 281.35940001])]

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
    P, I = naive.maamp(T, m, excl_zone)
    motif_distances, motif_indices, motif_subspaces, motif_mdls = aamp_mmotifs(
        T, P, I, max_distance=np.inf, max_matches=2, k=1
    )

    npt.assert_array_almost_equal(motif_distances_expected, motif_distances)
    npt.assert_array_almost_equal(motif_indices_expected, motif_indices)
    npt.assert_array_almost_equal(motif_subspaces_expected, motif_subspaces)
    npt.assert_array_almost_equal(motif_mdls_expected, motif_mdls)


def test_motifs_multidimensional_more_motifs_when_cutoffs_is_set():
    # Find the best multidimensional motif pairs if cutoffs is set

    motif_distances_expected = np.array([[0.0, 1.41421356], [0.0, 2.06639783]])
    motif_indices_expected = np.array([[2, 9], [0, 5]])
    motif_subspaces_expected = [np.array([3]), np.array([2])]
    motif_mdls_expected = [
        np.array([244.0, 260.67970001, 279.86313714, 281.35940001]),
        np.array([254.33985, 260.67970001, 279.86313714, 291.20703549]),
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
    P, I = naive.maamp(T, m, excl_zone)
    motif_distances, motif_indices, motif_subspaces, motif_mdls = aamp_mmotifs(
        T, P, I, max_distance=np.inf, cutoffs=3, max_matches=2, max_motifs=10
    )

    npt.assert_array_almost_equal(motif_distances_expected, motif_distances)
    npt.assert_array_almost_equal(motif_indices_expected, motif_indices)
    npt.assert_array_almost_equal(motif_subspaces_expected, motif_subspaces)
    npt.assert_array_almost_equal(motif_mdls_expected, motif_mdls)


def test_motifs_multidimensional_two_motifs_all_dimensions():
    # Find the best two motif pairs

    motif_distances_expected = np.array([[0.0, 1.41421356], [0.0, 2.06639783]])
    motif_indices_expected = np.array([[2, 9], [0, 5]])
    motif_subspaces_expected = [np.array([3]), np.array([2])]
    motif_mdls_expected = [
        np.array([244.0, 260.67970001, 279.86313714, 281.35940001]),
        np.array([254.33985, 260.67970001, 279.86313714, 291.20703549]),
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
    P, I = naive.maamp(T, m, excl_zone)
    motif_distances, motif_indices, motif_subspaces, motif_mdls = aamp_mmotifs(
        T,
        P,
        I,
        max_distance=np.inf,
        cutoffs=np.inf,
        max_matches=2,
        max_motifs=2,
    )

    npt.assert_array_almost_equal(motif_distances_expected, motif_distances)
    npt.assert_array_almost_equal(motif_indices_expected, motif_indices)
    npt.assert_array_almost_equal(motif_subspaces_expected, motif_subspaces)
    npt.assert_array_almost_equal(motif_mdls_expected, motif_mdls)
