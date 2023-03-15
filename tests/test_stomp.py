import naive
import numpy as np
import numpy.testing as npt
import pytest

from stumpy import stomp

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

window_size = [8, 16, 32]
substitution_locations = [(slice(0, 0), 0, -1, slice(1, 3), [0, 3])]
substitution_values = [np.nan, np.inf]


def test_stomp_int_input():
    with pytest.raises(TypeError):
        stomp(np.arange(10), 5, ignore_trivial=True)


@pytest.mark.parametrize("T_A, T_B", test_data)
def test_stomp_self_join(T_A, T_B):
    m = 3
    zone = int(np.ceil(m / 4))
    ref_mp = naive.stump(T_B, m, exclusion_zone=zone, row_wise=True)
    comp_mp = stomp._stomp(T_B, m, ignore_trivial=True)
    naive.replace_inf(ref_mp)
    naive.replace_inf(comp_mp)
    npt.assert_almost_equal(ref_mp, comp_mp)


@pytest.mark.parametrize("T_A, T_B", test_data)
@pytest.mark.parametrize("m", window_size)
def test_stump_self_join_larger_window(T_A, T_B, m):
    if len(T_B) > m:
        zone = int(np.ceil(m / 4))
        ref_mp = naive.stump(T_B, m, exclusion_zone=zone, row_wise=True)
        comp_mp = stomp._stomp(T_B, m, ignore_trivial=True)
        naive.replace_inf(ref_mp)
        naive.replace_inf(comp_mp)

        npt.assert_almost_equal(ref_mp, comp_mp)


@pytest.mark.parametrize("T_A, T_B", test_data)
def test_stomp_A_B_join(T_A, T_B):
    m = 3
    ref_mp = naive.stump(T_A, m, T_B=T_B, row_wise=True)
    comp_mp = stomp._stomp(T_A, m, T_B, ignore_trivial=False)
    naive.replace_inf(ref_mp)
    naive.replace_inf(comp_mp)
    npt.assert_almost_equal(ref_mp, comp_mp)


@pytest.mark.parametrize("T_A, T_B", test_data)
@pytest.mark.parametrize("substitute_B", substitution_values)
@pytest.mark.parametrize("substitution_locations", substitution_locations)
def test_stomp_nan_inf_self_join(T_A, T_B, substitute_B, substitution_locations):
    m = 3

    T_B_sub = T_B.copy()

    for substitution_location_B in substitution_locations:
        T_B_sub[:] = T_B[:]
        T_B_sub[substitution_location_B] = substitute_B

        zone = int(np.ceil(m / 4))
        ref_mp = naive.stump(T_B_sub, m, exclusion_zone=zone, row_wise=True)
        comp_mp = stomp._stomp(T_B_sub, m, ignore_trivial=True)
        naive.replace_inf(ref_mp)
        naive.replace_inf(comp_mp)
        npt.assert_almost_equal(ref_mp, comp_mp)


@pytest.mark.parametrize("T_A, T_B", test_data)
@pytest.mark.parametrize("substitute_A", substitution_values)
@pytest.mark.parametrize("substitute_B", substitution_values)
@pytest.mark.parametrize("substitution_locations", substitution_locations)
def test_stomp_nan_inf_A_B_join(
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

            ref_mp = naive.stump(T_A_sub, m, T_B=T_B_sub, row_wise=True)
            comp_mp = stomp._stomp(T_A_sub, m, T_B_sub, ignore_trivial=False)
            naive.replace_inf(ref_mp)
            naive.replace_inf(comp_mp)
            npt.assert_almost_equal(ref_mp, comp_mp)


def test_stomp_nan_zero_mean_self_join():
    T = np.array([-1, 0, 1, np.inf, 1, 0, -1])
    m = 3

    zone = int(np.ceil(m / 4))
    ref_mp = naive.stump(T, m, exclusion_zone=zone, row_wise=True)
    comp_mp = stomp._stomp(T, m, ignore_trivial=True)

    naive.replace_inf(ref_mp)
    naive.replace_inf(comp_mp)
    npt.assert_almost_equal(ref_mp, comp_mp)
