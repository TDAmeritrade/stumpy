import naive
import numpy as np
import numpy.testing as npt
import pytest

from stumpy import config, mmotifs

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


@pytest.mark.parametrize("T", test_data)
def test_mmotifs_with_default_parameters(T):
    motif_distances_ref = np.array([[0.0000000e00, 1.1151008e-07]])
    motif_indices_ref = np.array([[2, 9]])
    motif_subspaces_ref = [np.array([1])]
    motif_mdls_ref = [np.array([232.0, 250.57542476, 260.0, 271.3509059])]

    m = 4
    excl_zone = int(np.ceil(m / config.STUMPY_EXCL_ZONE_DENOM))
    P, I = naive.mstump(T, m, excl_zone)
    (
        motif_distances_cmp,
        motif_indices_cmp,
        motif_subspaces_cmp,
        motif_mdls_cmp,
    ) = mmotifs(T, P, I)

    npt.assert_array_almost_equal(motif_distances_ref, motif_distances_cmp)
    npt.assert_array_almost_equal(motif_indices_ref, motif_indices_cmp)
    npt.assert_array_almost_equal(motif_subspaces_ref, motif_subspaces_cmp)
    npt.assert_array_almost_equal(motif_mdls_ref, motif_mdls_cmp)


@pytest.mark.parametrize("T", test_data)
def test_mmotifs_max_matches_none(T):
    motif_distances_ref = np.array([[0.0000000e00, 1.1151008e-07]])
    motif_indices_ref = np.array([[2, 9]])
    motif_subspaces_ref = [np.array([1])]
    motif_mdls_ref = [np.array([232.0, 250.57542476, 260.0, 271.3509059])]

    m = 4
    excl_zone = int(np.ceil(m / config.STUMPY_EXCL_ZONE_DENOM))
    P, I = naive.mstump(T, m, excl_zone)
    (
        motif_distances_cmp,
        motif_indices_cmp,
        motif_subspaces_cmp,
        motif_mdls_cmp,
    ) = mmotifs(T, P, I, max_matches=None)

    npt.assert_array_almost_equal(motif_distances_ref, motif_distances_cmp)
    npt.assert_array_almost_equal(motif_indices_ref, motif_indices_cmp)
    npt.assert_array_almost_equal(motif_subspaces_ref, motif_subspaces_cmp)
    npt.assert_array_almost_equal(motif_mdls_ref, motif_mdls_cmp)


@pytest.mark.parametrize("T", test_data)
def test_mmotifs_more_motifs_when_cutoffs_3(T):
    motif_distances_ref = np.array([[0.0000000e00, 1.1151008e-07]])
    motif_indices_ref = np.array([[2, 9]])
    motif_subspaces_ref = [np.array([1])]
    motif_mdls_ref = [np.array([232.0, 250.57542476, 260.0, 271.3509059])]

    m = 4
    excl_zone = int(np.ceil(m / config.STUMPY_EXCL_ZONE_DENOM))
    P, I = naive.mstump(T, m, excl_zone)
    (
        motif_distances_cmp,
        motif_indices_cmp,
        motif_subspaces_cmp,
        motif_mdls_cmp,
    ) = mmotifs(T, P, I, cutoffs=3, max_motifs=10)

    npt.assert_array_almost_equal(motif_distances_ref, motif_distances_cmp)
    npt.assert_array_almost_equal(motif_indices_ref, motif_indices_cmp)
    npt.assert_array_almost_equal(motif_subspaces_ref, motif_subspaces_cmp)
    npt.assert_array_almost_equal(motif_mdls_ref, motif_mdls_cmp)


@pytest.mark.parametrize("T", test_data)
def test_mmotifs_more_motifs_cutoffs_is_list(T):
    motif_distances_ref = np.array([[0.0000000e00, 1.1151008e-07]])
    motif_indices_ref = np.array([[2, 9]])
    motif_subspaces_ref = [np.array([1])]
    motif_mdls_ref = [np.array([232.0, 250.57542476, 260.0, 271.3509059])]

    m = 4
    cutoffs = [2, 3, 4, 5]
    excl_zone = int(np.ceil(m / config.STUMPY_EXCL_ZONE_DENOM))
    P, I = naive.mstump(T, m, excl_zone)
    (
        motif_distances_cmp,
        motif_indices_cmp,
        motif_subspaces_cmp,
        motif_mdls_cmp,
    ) = mmotifs(T, P, I, cutoffs=cutoffs, max_motifs=10)

    npt.assert_array_almost_equal(motif_distances_ref, motif_distances_cmp)
    npt.assert_array_almost_equal(motif_indices_ref, motif_indices_cmp)
    npt.assert_array_almost_equal(motif_subspaces_ref, motif_subspaces_cmp)
    npt.assert_array_almost_equal(motif_mdls_ref, motif_mdls_cmp)


@pytest.mark.parametrize("T", test_data)
def test_mmotifs_max_matches_2_k_1(T):
    motif_distances_ref = np.array([[0.0, 0.20948156]])
    motif_indices_ref = np.array([[2, 9]])
    motif_subspaces_ref = [np.array([1, 3])]
    motif_mdls_ref = [np.array([232.0, 250.57542476, 260.0, 271.3509059])]

    m = 4
    excl_zone = int(np.ceil(m / config.STUMPY_EXCL_ZONE_DENOM))
    P, I = naive.mstump(T, m, excl_zone)
    (
        motif_distances_cmp,
        motif_indices_cmp,
        motif_subspaces_cmp,
        motif_mdls_cmp,
    ) = mmotifs(T, P, I, max_distance=np.inf, max_matches=2, k=1)

    npt.assert_array_almost_equal(motif_distances_ref, motif_distances_cmp)
    npt.assert_array_almost_equal(motif_indices_ref, motif_indices_cmp)
    npt.assert_array_almost_equal(motif_subspaces_ref, motif_subspaces_cmp)
    npt.assert_array_almost_equal(motif_mdls_ref, motif_mdls_cmp)


@pytest.mark.parametrize("T", test_data)
def test_mmotifs_two_motif_pairs_max_motifs_2(T):
    motif_distances_ref = np.array(
        [[0.00000000e00, 1.11510080e-07], [1.68587394e-07, 2.58694429e-01]]
    )
    motif_indices_ref = np.array([[2, 9], [6, 1]])
    motif_subspaces_ref = [np.array([1]), np.array([2])]
    motif_mdls_ref = [
        np.array([232.0, 250.57542476, 260.0, 271.3509059]),
        np.array([264.0, 280.0, 299.01955001, 310.51024953]),
    ]

    m = 4
    excl_zone = int(np.ceil(m / config.STUMPY_EXCL_ZONE_DENOM))
    P, I = naive.mstump(T, m, excl_zone)
    (
        motif_distances_cmp,
        motif_indices_cmp,
        motif_subspaces_cmp,
        motif_mdls_cmp,
    ) = mmotifs(
        T, P, I, cutoffs=np.inf, max_motifs=2, max_distance=np.inf, max_matches=2
    )

    npt.assert_array_almost_equal(motif_distances_ref, motif_distances_cmp)
    npt.assert_array_almost_equal(motif_indices_ref, motif_indices_cmp)
    npt.assert_array_almost_equal(motif_subspaces_ref, motif_subspaces_cmp)
    npt.assert_array_almost_equal(motif_mdls_ref, motif_mdls_cmp)


@pytest.mark.parametrize("T", test_data)
def test_mmotifs_with_default_parameters_with_isconstant(T):
    motif_distances_ref = np.array([[0.0000000e00, 1.1151008e-07]])
    motif_indices_ref = np.array([[2, 9]])
    motif_subspaces_ref = [np.array([1])]
    motif_mdls_ref = [np.array([232.0, 250.57542476, 260.0, 271.3509059])]

    m = 4
    excl_zone = int(np.ceil(m / config.STUMPY_EXCL_ZONE_DENOM))

    # The following `T_subseq_isconstant` is basically equivalent to
    # `T_subseq_isconstant=None` (default). The goal is to test its
    # functionality.
    T_subseq_isconstant = [
        None,
        naive.rolling_isconstant(T[1], m),
        None,
        naive.is_ptp_zero_1d,
    ]

    P, I = naive.mstump(T, m, excl_zone, T_subseq_isconstant=T_subseq_isconstant)
    (
        motif_distances_cmp,
        motif_indices_cmp,
        motif_subspaces_cmp,
        motif_mdls_cmp,
    ) = mmotifs(T, P, I, T_subseq_isconstant=T_subseq_isconstant)

    npt.assert_array_almost_equal(motif_distances_ref, motif_distances_cmp)
    npt.assert_array_almost_equal(motif_indices_ref, motif_indices_cmp)
    npt.assert_array_almost_equal(motif_subspaces_ref, motif_subspaces_cmp)
    npt.assert_array_almost_equal(motif_mdls_ref, motif_mdls_cmp)
