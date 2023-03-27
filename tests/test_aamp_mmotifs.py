import naive
import numpy as np
import numpy.testing as npt
import pytest

from stumpy import aamp_mmotifs, config

test_data = [
    np.array(
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
]


def test_aamp_mmotifs_default_parameters():
    motif_distances_ref = np.array(
        [[0.0, 0.06315749, 0.25275899, 0.34087884, 0.3452315]]
    )
    motif_indices_ref = np.array([[19, 77, 63, 52, 71]])
    motif_subspaces_ref = [np.array([2])]
    motif_mdls_ref = [
        np.array([411.60964047, 423.69925001, 449.11032383, 476.95855027, 506.62406252])
    ]

    np.random.seed(0)
    T = np.random.rand(500).reshape(5, 100)

    m = 5
    excl_zone = int(np.ceil(m / config.STUMPY_EXCL_ZONE_DENOM))
    P, I = naive.maamp(T, m, excl_zone)
    (
        motif_distances_cmp,
        motif_indices_cmp,
        motif_subspaces_cmp,
        motif_mdls_cmp,
    ) = aamp_mmotifs(T, P, I)

    npt.assert_array_almost_equal(motif_distances_ref, motif_distances_cmp)
    npt.assert_array_almost_equal(motif_indices_ref, motif_indices_cmp)
    npt.assert_array_almost_equal(motif_subspaces_ref, motif_subspaces_cmp)
    npt.assert_array_almost_equal(motif_mdls_ref, motif_mdls_cmp)


@pytest.mark.parametrize("T", test_data)
def test_aamp_mmotifs_max_distance(T):
    motif_distances_ref = np.array(
        [[0.0, 1.41421356, 4.46430286, 6.85346628, 8.207923, 8.50529247]]
    )
    motif_indices_ref = np.array([[2, 9, 0, 11, 7, 5]])
    motif_subspaces_ref = [np.array([3])]
    motif_mdls_ref = [np.array([244.0, 260.67970001, 279.86313714, 281.35940001])]

    m = 4
    excl_zone = int(np.ceil(m / config.STUMPY_EXCL_ZONE_DENOM))
    P, I = naive.maamp(T, m, excl_zone)
    (
        motif_distances_cmp,
        motif_indices_cmp,
        motif_subspaces_cmp,
        motif_mdls_cmp,
    ) = aamp_mmotifs(T, P, I, max_distance=np.inf)

    npt.assert_array_almost_equal(motif_distances_ref, motif_distances_cmp)
    npt.assert_array_almost_equal(motif_indices_ref, motif_indices_cmp)
    npt.assert_array_almost_equal(motif_subspaces_ref, motif_subspaces_cmp)
    npt.assert_array_almost_equal(motif_mdls_ref, motif_mdls_cmp)


@pytest.mark.parametrize("T", test_data)
def test_aamp_mmotifs_max_distance_max_matches_none(T):
    motif_distances_ref = np.array(
        [[0.0, 1.41421356, 4.46430286, 6.85346628, 8.207923, 8.50529247]]
    )
    motif_indices_ref = np.array([[2, 9, 0, 11, 7, 5]])
    motif_subspaces_ref = [np.array([3])]
    motif_mdls_ref = [np.array([244.0, 260.67970001, 279.86313714, 281.35940001])]

    m = 4
    excl_zone = int(np.ceil(m / config.STUMPY_EXCL_ZONE_DENOM))
    P, I = naive.maamp(T, m, excl_zone)
    (
        motif_distances_cmp,
        motif_indices_cmp,
        motif_subspaces_cmp,
        motif_mdls_cmp,
    ) = aamp_mmotifs(T, P, I, max_distance=np.inf, max_matches=None)

    npt.assert_array_almost_equal(motif_distances_ref, motif_distances_cmp)
    npt.assert_array_almost_equal(motif_indices_ref, motif_indices_cmp)
    npt.assert_array_almost_equal(motif_subspaces_ref, motif_subspaces_cmp)
    npt.assert_array_almost_equal(motif_mdls_ref, motif_mdls_cmp)


@pytest.mark.parametrize("T", test_data)
def test_aamp_mmotifs_max_motifs_1_max_matches_2_k_1(T):
    motif_distances_ref = np.array([[0.0, 2.87778559]])
    motif_indices_ref = np.array([[0, 5]])
    motif_subspaces_ref = [np.array([2, 1])]
    motif_mdls_ref = [np.array([244.0, 260.67970001, 279.86313714, 281.35940001])]

    m = 4
    excl_zone = int(np.ceil(m / config.STUMPY_EXCL_ZONE_DENOM))
    P, I = naive.maamp(T, m, excl_zone)
    (
        motif_distances_cmp,
        motif_indices_cmp,
        motif_subspaces_cmp,
        motif_mdls_cmp,
    ) = aamp_mmotifs(T, P, I, max_distance=np.inf, max_matches=2, k=1)

    npt.assert_array_almost_equal(motif_distances_ref, motif_distances_cmp)
    npt.assert_array_almost_equal(motif_indices_ref, motif_indices_cmp)
    npt.assert_array_almost_equal(motif_subspaces_ref, motif_subspaces_cmp)
    npt.assert_array_almost_equal(motif_mdls_ref, motif_mdls_cmp)


@pytest.mark.parametrize("T", test_data)
def test_aamp_mmotifs_more_motif_pairs_cutoffs_3(T):
    motif_distances_ref = np.array([[0.0, 1.41421356], [0.0, 2.06639783]])
    motif_indices_ref = np.array([[2, 9], [0, 5]])
    motif_subspaces_ref = [np.array([3]), np.array([2])]
    motif_mdls_ref = [
        np.array([244.0, 260.67970001, 279.86313714, 281.35940001]),
        np.array([254.33985, 260.67970001, 279.86313714, 291.20703549]),
    ]

    m = 4
    excl_zone = int(np.ceil(m / config.STUMPY_EXCL_ZONE_DENOM))
    P, I = naive.maamp(T, m, excl_zone)
    (
        motif_distances_cmp,
        motif_indices_cmp,
        motif_subspaces_cmp,
        motif_mdls_cmp,
    ) = aamp_mmotifs(
        T, P, I, max_distance=np.inf, cutoffs=3, max_matches=2, max_motifs=10
    )

    npt.assert_array_almost_equal(motif_distances_ref, motif_distances_cmp)
    npt.assert_array_almost_equal(motif_indices_ref, motif_indices_cmp)
    npt.assert_array_almost_equal(motif_subspaces_ref, motif_subspaces_cmp)
    npt.assert_array_almost_equal(motif_mdls_ref, motif_mdls_cmp)


@pytest.mark.parametrize("T", test_data)
def test_aamp_mmotifs_more_motif_pairs_cutoffs_as_list(T):
    motif_distances_ref = np.array([[0.0, 1.41421356]])
    motif_indices_ref = np.array([[2, 9]])
    motif_subspaces_ref = [np.array([3])]
    motif_mdls_ref = [np.array([244.0, 260.67970001, 279.86313714, 281.35940001])]

    m = 4
    cutoffs = [2, 3, 4, 5]
    excl_zone = int(np.ceil(m / config.STUMPY_EXCL_ZONE_DENOM))
    P, I = naive.maamp(T, m, excl_zone)
    (
        motif_distances_cmp,
        motif_indices_cmp,
        motif_subspaces_cmp,
        motif_mdls_cmp,
    ) = aamp_mmotifs(
        T, P, I, max_distance=np.inf, cutoffs=cutoffs, max_matches=2, max_motifs=10
    )

    npt.assert_array_almost_equal(motif_distances_ref, motif_distances_cmp)
    npt.assert_array_almost_equal(motif_indices_ref, motif_indices_cmp)
    npt.assert_array_almost_equal(motif_subspaces_ref, motif_subspaces_cmp)
    npt.assert_array_almost_equal(motif_mdls_ref, motif_mdls_cmp)


@pytest.mark.parametrize("T", test_data)
def test_aamp_mmotifs_two_motif_pairs_max_motifs_2(T):
    motif_distances_ref = np.array([[0.0, 1.41421356], [0.0, 2.06639783]])
    motif_indices_ref = np.array([[2, 9], [0, 5]])
    motif_subspaces_ref = [np.array([3]), np.array([2])]
    motif_mdls_ref = [
        np.array([244.0, 260.67970001, 279.86313714, 281.35940001]),
        np.array([254.33985, 260.67970001, 279.86313714, 291.20703549]),
    ]

    m = 4
    excl_zone = int(np.ceil(m / config.STUMPY_EXCL_ZONE_DENOM))
    P, I = naive.maamp(T, m, excl_zone)
    (
        motif_distances_cmp,
        motif_indices_cmp,
        motif_subspaces_cmp,
        motif_mdls_cmp,
    ) = aamp_mmotifs(
        T,
        P,
        I,
        max_distance=np.inf,
        cutoffs=np.inf,
        max_matches=2,
        max_motifs=2,
    )

    npt.assert_array_almost_equal(motif_distances_ref, motif_distances_cmp)
    npt.assert_array_almost_equal(motif_indices_ref, motif_indices_cmp)
    npt.assert_array_almost_equal(motif_subspaces_ref, motif_subspaces_cmp)
    npt.assert_array_almost_equal(motif_mdls_ref, motif_mdls_cmp)
