import numpy as np
import numpy.testing as npt
from stumpy import stamp, core
import pytest
import utils

test_data = [
    (
        np.array([9, 8100, -60, 7], dtype=np.float64),
        np.array([584, -11, 23, 79, 1001, 0, -19], dtype=np.float64),
    ),
    (
        np.random.uniform(-1000, 1000, [8]).astype(np.float64),
        np.random.uniform(-1000, 1000, [64]).astype(np.float64),
    ),
]

substitution_values = [np.nan, np.inf]


@pytest.mark.parametrize("T_A, T_B", test_data)
def test_stamp_self_join(T_A, T_B):
    m = 3
    zone = int(np.ceil(m / 2))
    left = np.array(
        [
            utils.naive_mass(Q, T_B, m, i, zone, ignore_trivial=True)
            for i, Q in enumerate(core.rolling_window(T_B, m))
        ],
        dtype=object,
    )
    right = stamp.stamp(T_B, T_B, m, ignore_trivial=True)
    utils.replace_inf(left)
    utils.replace_inf(right)
    npt.assert_almost_equal(left[:, :2], right)


@pytest.mark.parametrize("T_A, T_B", test_data)
def test_stamp_A_B_join(T_A, T_B):
    m = 3
    left = np.array(
        [utils.naive_mass(Q, T_A, m) for Q in core.rolling_window(T_B, m)], dtype=object
    )
    right = stamp.stamp(T_A, T_B, m)
    utils.replace_inf(left)
    utils.replace_inf(right)
    npt.assert_almost_equal(left[:, :2], right)


@pytest.mark.parametrize("T_A, T_B", test_data)
@pytest.mark.parametrize("substitute_B", substitution_values)
def test_stamp_nan_inf_self_join(T_A, T_B, substitute_B):
    m = 3

    T_B_sub = T_B.copy()

    substitution_locations = [slice(0, 0), 0, -1, slice(1, 3), [0, 3]]
    for substitution_location_B in substitution_locations:
        T_B_sub[:] = T_B[:]
        T_B_sub[substitution_location_B] = substitute_B

        zone = int(np.ceil(m / 2))
        left = np.array(
            [
                utils.naive_mass(Q, T_B_sub, m, i, zone, ignore_trivial=True)
                for i, Q in enumerate(core.rolling_window(T_B_sub, m))
            ],
            dtype=object,
        )
        right = stamp.stamp(T_B_sub, T_B_sub, m, ignore_trivial=True)
        utils.replace_inf(left)
        utils.replace_inf(right)
        npt.assert_almost_equal(left[:, :2], right)


@pytest.mark.parametrize("T_A, T_B", test_data)
@pytest.mark.parametrize("substitute_A", substitution_values)
@pytest.mark.parametrize("substitute_B", substitution_values)
def test_stamp_nan_inf_A_B_join(T_A, T_B, substitute_A, substitute_B):
    m = 3

    T_A_sub = T_A.copy()
    T_B_sub = T_B.copy()

    substitution_locations = [slice(0, 0), 0, -1, slice(1, 3), [0, 3]]
    for substitution_location_B in substitution_locations:
        for substitution_location_A in substitution_locations:
            T_A_sub[:] = T_A[:]
            T_B_sub[:] = T_B[:]
            T_A_sub[substitution_location_A] = substitute_A
            T_B_sub[substitution_location_B] = substitute_B

            left = np.array(
                [
                    utils.naive_mass(Q, T_A_sub, m)
                    for Q in core.rolling_window(T_B_sub, m)
                ],
                dtype=object,
            )
            right = stamp.stamp(T_A_sub, T_B_sub, m)
            utils.replace_inf(left)
            utils.replace_inf(right)
            npt.assert_almost_equal(left[:, :2], right)
