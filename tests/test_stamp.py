import functools

import naive
import numpy as np
import numpy.testing as npt
import pytest

from stumpy import core, stamp

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
    ref_P, ref_I, ref_left_I, ref_right_I = naive.mass_PI(
        Q, T_B, m, trivial_idx=trivial_idx, excl_zone=zone, ignore_trivial=True
    )
    comp_P, comp_I = stamp._mass_PI(
        Q, T_B, M_T, Σ_T, trivial_idx=trivial_idx, excl_zone=zone
    )

    npt.assert_almost_equal(ref_P, comp_P)
    npt.assert_almost_equal(ref_I, comp_I)

    comp_left_P, comp_left_I = stamp._mass_PI(
        Q,
        T_B,
        M_T,
        Σ_T,
        trivial_idx=trivial_idx,
        excl_zone=zone,
        left=True,
    )

    npt.assert_almost_equal(ref_left_I, comp_left_I)

    comp_right_P, comp_right_I = stamp._mass_PI(
        Q,
        T_B,
        M_T,
        Σ_T,
        trivial_idx=trivial_idx,
        excl_zone=zone,
        right=True,
    )

    npt.assert_almost_equal(ref_right_I, comp_right_I)


def test_stamp_int_input():
    with pytest.raises(TypeError):
        T = np.arange(10)
        stamp(T, T, 5, ignore_trivial=True)


@pytest.mark.parametrize("T_A, T_B", test_data)
def test_stamp_self_join(T_A, T_B):
    m = 3
    zone = int(np.ceil(m / 2))
    ref_mp = naive.stump(T_B, m, exclusion_zone=zone, row_wise=True)
    comp_mp = stamp.stamp(T_B, T_B, m, ignore_trivial=True)
    naive.replace_inf(ref_mp)
    naive.replace_inf(comp_mp)
    npt.assert_almost_equal(ref_mp[:, :2], comp_mp)


@pytest.mark.parametrize("T_A, T_B", test_data)
def test_stamp_A_B_join(T_A, T_B):
    m = 3
    ref_mp = naive.stump(T_A, m, T_B=T_B, row_wise=True)
    comp_mp = stamp.stamp(T_A, T_B, m)
    naive.replace_inf(ref_mp)
    naive.replace_inf(comp_mp)
    npt.assert_almost_equal(ref_mp[:, :2], comp_mp)


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
        ref_mp = naive.stump(T_B_sub, m, exclusion_zone=zone, row_wise=True)
        comp_mp = stamp.stamp(T_B_sub, T_B_sub, m, ignore_trivial=True)
        naive.replace_inf(ref_mp)
        naive.replace_inf(comp_mp)
        npt.assert_almost_equal(ref_mp[:, :2], comp_mp)


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

            ref_mp = naive.stump(T_A_sub, m, T_B=T_B_sub, row_wise=True)
            comp_mp = stamp.stamp(T_A_sub, T_B_sub, m)
            naive.replace_inf(ref_mp)
            naive.replace_inf(comp_mp)
            npt.assert_almost_equal(ref_mp[:, :2], comp_mp)


def test_stamp_nan_zero_mean_self_join():
    T = np.array([-1, 0, 1, np.inf, 1, 0, -1])
    m = 3

    zone = int(np.ceil(m / 2))
    ref_mp = naive.stump(T, m, exclusion_zone=zone, row_wise=True)
    comp_mp = stamp.stamp(T, T, m, ignore_trivial=True)

    naive.replace_inf(ref_mp)
    naive.replace_inf(comp_mp)
    npt.assert_almost_equal(ref_mp[:, :2], comp_mp)


def test_stamp_mass_PI_with_isconstant_case1():
    # case1: The query `Q` is not constant
    T_B = np.random.uniform(-1, 1, [64])
    isconstant_custom_func = functools.partial(
        naive.isconstant_func_stddev_threshold, stddev_threshold=0.5
    )

    m = 3
    zone = int(np.ceil(m / 2))

    T_B_subseq_isconstant = naive.rolling_isconstant(T_B, m, isconstant_custom_func)
    M_T, Σ_T = core.compute_mean_std(T_B, m)

    trivial_idx = np.random.choice(np.flatnonzero(~T_B_subseq_isconstant))
    Q = T_B[trivial_idx : trivial_idx + m]

    ref_P, ref_I, ref_left_I, ref_right_I = naive.mass_PI(
        Q,
        T_B,
        m,
        trivial_idx=trivial_idx,
        excl_zone=zone,
        ignore_trivial=True,
        T_subseq_isconstant=isconstant_custom_func,
        Q_subseq_isconstant=isconstant_custom_func,
    )
    comp_P, comp_I = stamp._mass_PI(
        Q,
        T_B,
        M_T,
        Σ_T,
        trivial_idx=trivial_idx,
        excl_zone=zone,
        T_subseq_isconstant=isconstant_custom_func,
        Q_subseq_isconstant=isconstant_custom_func,
    )

    npt.assert_almost_equal(ref_P, comp_P)
    npt.assert_almost_equal(ref_I, comp_I)

    comp_left_P, comp_left_I = stamp._mass_PI(
        Q,
        T_B,
        M_T,
        Σ_T,
        trivial_idx=trivial_idx,
        excl_zone=zone,
        left=True,
        T_subseq_isconstant=isconstant_custom_func,
        Q_subseq_isconstant=isconstant_custom_func,
    )

    npt.assert_almost_equal(ref_left_I, comp_left_I)

    comp_right_P, comp_right_I = stamp._mass_PI(
        Q,
        T_B,
        M_T,
        Σ_T,
        trivial_idx=trivial_idx,
        excl_zone=zone,
        right=True,
        T_subseq_isconstant=isconstant_custom_func,
        Q_subseq_isconstant=isconstant_custom_func,
    )

    npt.assert_almost_equal(ref_right_I, comp_right_I)


def test_stamp_mass_PI_with_isconstant_case2():
    # case2: The query `Q` is constant
    T_B = np.random.uniform(-1, 1, [64])
    isconstant_custom_func = functools.partial(
        naive.isconstant_func_stddev_threshold, stddev_threshold=0.5
    )

    m = 3
    zone = int(np.ceil(m / 2))

    T_B_subseq_isconstant = naive.rolling_isconstant(T_B, m, isconstant_custom_func)
    M_T, Σ_T = core.compute_mean_std(T_B, m)

    trivial_idx = np.random.choice(np.flatnonzero(T_B_subseq_isconstant))
    Q = T_B[trivial_idx : trivial_idx + m]

    ref_P, ref_I, ref_left_I, ref_right_I = naive.mass_PI(
        Q,
        T_B,
        m,
        trivial_idx=trivial_idx,
        excl_zone=zone,
        ignore_trivial=True,
        T_subseq_isconstant=isconstant_custom_func,
        Q_subseq_isconstant=isconstant_custom_func,
    )
    comp_P, comp_I = stamp._mass_PI(
        Q,
        T_B,
        M_T,
        Σ_T,
        trivial_idx=trivial_idx,
        excl_zone=zone,
        T_subseq_isconstant=isconstant_custom_func,
        Q_subseq_isconstant=isconstant_custom_func,
    )

    npt.assert_almost_equal(ref_P, comp_P)
    npt.assert_almost_equal(ref_I, comp_I)

    comp_left_P, comp_left_I = stamp._mass_PI(
        Q,
        T_B,
        M_T,
        Σ_T,
        trivial_idx=trivial_idx,
        excl_zone=zone,
        left=True,
        T_subseq_isconstant=isconstant_custom_func,
        Q_subseq_isconstant=isconstant_custom_func,
    )

    npt.assert_almost_equal(ref_left_I, comp_left_I)

    comp_right_P, comp_right_I = stamp._mass_PI(
        Q,
        T_B,
        M_T,
        Σ_T,
        trivial_idx=trivial_idx,
        excl_zone=zone,
        right=True,
        T_subseq_isconstant=isconstant_custom_func,
        Q_subseq_isconstant=isconstant_custom_func,
    )

    npt.assert_almost_equal(ref_right_I, comp_right_I)


@pytest.mark.parametrize("T_A, T_B", test_data)
def test_stamp_self_join_with_isconstant(T_A, T_B):
    m = 3
    zone = int(np.ceil(m / 2))
    isconstant_custom_func = functools.partial(
        naive.isconstant_func_stddev_threshold, quantile_threshold=0.05
    )

    ref_mp = naive.stump(
        T_B,
        m,
        exclusion_zone=zone,
        row_wise=True,
        T_A_subseq_isconstant=isconstant_custom_func,
    )
    comp_mp = stamp.stamp(
        T_B,
        T_B,
        m,
        ignore_trivial=True,
        T_A_subseq_isconstant=isconstant_custom_func,
        T_B_subseq_isconstant=isconstant_custom_func,
    )
    naive.replace_inf(ref_mp)
    naive.replace_inf(comp_mp)
    npt.assert_almost_equal(ref_mp[:, :2], comp_mp)
