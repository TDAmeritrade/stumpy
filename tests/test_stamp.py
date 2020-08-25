import numpy as np
import numpy.testing as npt
from stumpy import stamp, core
import pytest
import naive

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
substitution_locations = [(slice(0, 0), 0, -1, slice(1, 3), [0, 3])]


@pytest.mark.parametrize("T_A, T_B", test_data)
def test_stamp_mass_PI(T_A, T_B):
    m = 3
    trivial_idx = 2
    zone = int(np.ceil(m / 2))
    Q = T_B[trivial_idx : trivial_idx + m]
    M_T, Σ_T = core.compute_mean_std(T_B, m)
    left_P, left_I, left_left_I, left_right_I = naive.mass(
        Q, T_B, m, trivial_idx=trivial_idx, excl_zone=zone, ignore_trivial=True
    )
    right_P, right_I = stamp._mass_PI(
        Q, T_B, M_T, Σ_T, trivial_idx=trivial_idx, excl_zone=zone
    )

    npt.assert_almost_equal(left_P, right_P)
    npt.assert_almost_equal(left_I, right_I)

    right_left_P, right_left_I = stamp._mass_PI(
        Q, T_B, M_T, Σ_T, trivial_idx=trivial_idx, excl_zone=zone, left=True
    )

    npt.assert_almost_equal(left_left_I, right_left_I)

    right_right_P, right_right_I = stamp._mass_PI(
        Q, T_B, M_T, Σ_T, trivial_idx=trivial_idx, excl_zone=zone, right=True
    )

    npt.assert_almost_equal(left_right_I, right_right_I)


@pytest.mark.parametrize("T_A, T_B", test_data)
def test_stamp_self_join(T_A, T_B):
    m = 3
    zone = int(np.ceil(m / 2))
    left = naive.stamp(T_B, m, exclusion_zone=zone)
    right = stamp.stamp(T_B, T_B, m, ignore_trivial=True)
    naive.replace_inf(left)
    naive.replace_inf(right)
    npt.assert_almost_equal(left[:, :2], right)


@pytest.mark.parametrize("T_A, T_B", test_data)
def test_stamp_A_B_join(T_A, T_B):
    m = 3
    left = naive.stamp(T_A, m, T_B=T_B)
    right = stamp.stamp(T_A, T_B, m)
    naive.replace_inf(left)
    naive.replace_inf(right)
    npt.assert_almost_equal(left[:, :2], right)


@pytest.mark.parametrize("T_A, T_B", test_data)
@pytest.mark.parametrize("substitute_B", substitution_values)
@pytest.mark.parametrize("substitution_locations", substitution_locations)
def test_stamp_nan_inf_self_join(T_A, T_B, substitute_B, substitution_locations):
    m = 3

    T_B_sub = T_B.copy()

    for substitution_location_B in substitution_locations:
        T_B_sub[:] = T_B[:]
        T_B_sub[substitution_location_B] = substitute_B

        zone = int(np.ceil(m / 2))
        left = naive.stamp(T_B_sub, m, exclusion_zone=zone)
        right = stamp.stamp(T_B_sub, T_B_sub, m, ignore_trivial=True)
        naive.replace_inf(left)
        naive.replace_inf(right)
        npt.assert_almost_equal(left[:, :2], right)


@pytest.mark.parametrize("T_A, T_B", test_data)
@pytest.mark.parametrize("substitute_A", substitution_values)
@pytest.mark.parametrize("substitute_B", substitution_values)
@pytest.mark.parametrize("substitution_locations", substitution_locations)
def test_stamp_nan_inf_A_B_join(
    T_A, T_B, substitute_A, substitute_B, substitution_locations
):
    m = 3

    T_A_sub = T_A.copy()
    T_B_sub = T_B.copy()

    for substitution_location_B in substitution_locations:
        for substitution_location_A in substitution_locations:
            T_A_sub[:] = T_A[:]
            T_B_sub[:] = T_B[:]
            T_A_sub[substitution_location_A] = substitute_A
            T_B_sub[substitution_location_B] = substitute_B

            left = naive.stamp(T_A_sub, m, T_B=T_B_sub)
            right = stamp.stamp(T_A_sub, T_B_sub, m)
            naive.replace_inf(left)
            naive.replace_inf(right)
            npt.assert_almost_equal(left[:, :2], right)


def test_stamp_nan_zero_mean_self_join():
    T = np.array([-1, 0, 1, np.inf, 1, 0, -1])
    m = 3

    zone = int(np.ceil(m / 2))
    left = naive.stamp(T, m, exclusion_zone=zone)
    right = stamp.stamp(T, T, m, ignore_trivial=True)

    naive.replace_inf(left)
    naive.replace_inf(right)
    npt.assert_almost_equal(left[:, :2], right)
