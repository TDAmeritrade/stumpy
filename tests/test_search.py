import numpy as np
import numpy.testing as npt
import pytest

from stumpy import search, stump, mstump, core

import naive


test_data_motifs = [
    (
        np.array([0, 1, 3, 2, 9, 1, 14, 15, 1, 2, 2, 10, 7], dtype=float),
        4,
        1,
        [[1, 8]],
        [0.28570485146990254],
    ),
    (
        np.array(
            [0, -10, 0, 0, 1, 3, 2, 9, 1, 14, 15, 1, 2, 2, 10, 7, 0, -10, 0, 0],
            dtype=float,
        ),
        4,
        2,
        [[0, 16], [4, 11]],
        [0, 0.28570485146990254],
    ),
]

test_data_motifs_multidimensional = [
    (
        np.array(
            [
                [0, 1, 3, 2, 9, 1, 14, 15, 1, 2, 2, 10, 7],
                [0, 1, 3, 2, 9, 1, 14, 15, 1, 2, 2, 10, 7],
            ],
            dtype=float,
        ),
        4,
        1,
        [[1, 8]],
        [0.28570485146990254],
    ),
    (
        np.array(
            [
                [0, 1, 3, 2, 9, 1, 14, 15, 1, 2, 2, 10, 7],
                [-2, 1, 3, 2, 9, 1, 15, 14, 1, 2, 2, 8, 7],
            ],
            dtype=float,
        ),
        4,
        1,
        [[1, 8]],
        [0.2676986029003686],
    ),
    (
        np.array(
            [
                [0, -10, 0, 0, 1, 3, 2, 9, 1, 14, 15, 1, 2, 2, 10, 7, 0, -10, 0, 0],
                [0, -10, 0, 0, 1, 3, 2, 9, 1, 14, 15, 1, 2, 2, 10, 7, 0, -10, 0, 0],
            ],
            dtype=float,
        ),
        4,
        2,
        [[0, 16], [4, 11]],
        [0, 0.28570485146990254],
    ),
]

test_data_pattern = [
    (
        np.array([0, -10, 0, 0], dtype=float),
        np.array(
            [
                0,
                -10,
                0,
                0,
                1,
                3,
                2,
                9,
                1,
                14,
                15,
                1,
                2,
                2,
                10,
                7,
                0,
                -10,
                0,
                0,
                0,
                -11,
                0,
                0,
            ],
            dtype=float,
        ),
    ),
    (
        np.array([0, -10, 0, 0], dtype=float),
        np.array(
            [
                0,
                -10,
                0,
                0,
                1,
                3,
                2,
                9,
                1,
                14,
                15,
                np.nan,
                2,
                2,
                10,
                7,
                0,
                -10,
                0,
                0,
                0,
                -11,
                0,
                0,
            ],
            dtype=float,
        ),
    ),
    (
        np.array([[0, -10, 0, 0], [0, -10, 0, 0]], dtype=float),
        np.array(
            [
                [
                    0,
                    -10,
                    0,
                    0,
                    1,
                    3,
                    2,
                    9,
                    1,
                    14,
                    15,
                    1,
                    2,
                    2,
                    10,
                    7,
                    0,
                    -10,
                    0,
                    0,
                    0,
                    -11,
                    0,
                    0,
                ],
                [
                    0,
                    -10,
                    0,
                    0,
                    1,
                    3,
                    2,
                    9,
                    1,
                    14,
                    15,
                    1,
                    2,
                    2,
                    10,
                    7,
                    0,
                    -10,
                    0,
                    0,
                    0,
                    -9,
                    0,
                    0,
                ],
            ],
            dtype=float,
        ),
    ),
]


@pytest.mark.parametrize("T, m, k, left_indices, left_profile_values", test_data_motifs)
def test_motifs(T, m, k, left_indices, left_profile_values):
    P = stump(T, m, ignore_trivial=True)
    right_indices, right_profile_values = search.motifs(T, P[:, 0], k=k, atol=0.001)

    npt.assert_array_equal(left_indices, right_indices)
    npt.assert_almost_equal(left_profile_values, right_profile_values)


@pytest.mark.parametrize(
    "T, m, k, left_indices, left_profile_values", test_data_motifs_multidimensional
)
def test_motifs_multidimensional(T, m, k, left_indices, left_profile_values):
    P, _ = mstump(T, m)
    right_indices, right_profile_values = search.motifs(T, P, k=k, atol=0.001)

    npt.assert_array_equal(left_indices, right_indices)
    npt.assert_almost_equal(left_profile_values, right_profile_values)


@pytest.mark.parametrize("Q, T", test_data_pattern)
def test_pattern(Q, T):
    m = Q.shape[0]
    excl_zone = int(np.ceil(m / 4))
    rtol = 1
    atol = 0.01
    profile_value = 0.1
    aamp = False

    left = naive.pattern(Q, T, excl_zone, profile_value, atol, rtol, aamp)
    right = search.pattern(
        Q,
        T,
        excl_zone=excl_zone,
        max_occurrences=None,
        profile_value=profile_value,
        atol=atol,
        rtol=rtol,
        aamp=aamp,
    )

    npt.assert_array_equal(left, right)


@pytest.mark.parametrize("Q, T", test_data_pattern)
def test_pattern_aamp(Q, T):
    m = Q.shape[0]
    excl_zone = int(np.ceil(m / 4))
    rtol = 1
    atol = 0.01
    profile_value = 0.1
    aamp = True

    left = naive.pattern(Q, T, excl_zone, profile_value, atol, rtol, aamp)
    right = search.pattern(
        Q,
        T,
        excl_zone=excl_zone,
        max_occurrences=None,
        profile_value=profile_value,
        atol=atol,
        rtol=rtol,
        aamp=aamp,
    )

    npt.assert_array_equal(left, right)
