import functools

import naive
import numpy as np
import numpy.testing as npt
import pytest

from stumpy import config, scrump, stump
from stumpy.scrump import prescrump

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
percentages = [(0.01, 0.1, 1.0)]


@pytest.mark.parametrize("T_A, T_B", test_data)
def test_prescrump_self_join(T_A, T_B):
    m = 3
    zone = int(np.ceil(m / 4))
    for s in range(1, zone + 1):
        seed = np.random.randint(100000)

        np.random.seed(seed)
        ref_P, ref_I = naive.prescrump(T_B, m, T_B, s=s, exclusion_zone=zone)

        np.random.seed(seed)
        comp_P, comp_I = prescrump(T_B, m, s=s)

        npt.assert_almost_equal(ref_P, comp_P)
        npt.assert_almost_equal(ref_I, comp_I)


@pytest.mark.parametrize("T_A, T_B", test_data)
def test_prescrump_A_B_join(T_A, T_B):
    m = 3
    zone = int(np.ceil(m / 4))
    for s in range(1, zone + 1):
        seed = np.random.randint(100000)

        np.random.seed(seed)
        ref_P, ref_I = naive.prescrump(T_A, m, T_B, s=s)

        np.random.seed(seed)
        comp_P, comp_I = prescrump(T_A, m, T_B=T_B, s=s)

        npt.assert_almost_equal(ref_P, comp_P)
        npt.assert_almost_equal(ref_I, comp_I)


@pytest.mark.parametrize("T_A, T_B", test_data)
def test_prescrump_A_B_join_swap(T_A, T_B):
    m = 3
    zone = int(np.ceil(m / 4))
    for s in range(1, zone + 1):
        seed = np.random.randint(100000)

        np.random.seed(seed)
        ref_P, ref_I = naive.prescrump(T_B, m, T_A, s=s)

        np.random.seed(seed)
        comp_P, comp_I = prescrump(T_B, m, T_B=T_A, s=s)

        npt.assert_almost_equal(ref_P, comp_P)
        npt.assert_almost_equal(ref_I, comp_I)


@pytest.mark.parametrize("T_A, T_B", test_data)
@pytest.mark.parametrize("m", window_size)
def test_prescrump_self_join_larger_window(T_A, T_B, m):
    if len(T_B) > m:
        zone = int(np.ceil(m / 4))
        for s in range(1, zone + 1):
            seed = np.random.randint(100000)

            np.random.seed(seed)
            ref_P, ref_I = naive.prescrump(T_B, m, T_B, s=s, exclusion_zone=zone)

            np.random.seed(seed)
            comp_P, comp_I = prescrump(T_B, m, s=s)

            npt.assert_almost_equal(ref_P, comp_P)
            npt.assert_almost_equal(ref_I, comp_I)


@pytest.mark.parametrize("T_A, T_B", test_data)
def test_prescrump_self_join_with_isconstant(T_A, T_B):
    isconstant_custom_func = functools.partial(
        naive.isconstant_func_stddev_threshold, quantile_threshold=0.05
    )

    m = 3
    zone = int(np.ceil(m / 4))
    for s in range(1, zone + 1):
        seed = np.random.randint(100000)

        np.random.seed(seed)
        ref_P, ref_I = naive.prescrump(
            T_B,
            m,
            T_B,
            s=s,
            exclusion_zone=zone,
            T_A_subseq_isconstant=isconstant_custom_func,
            T_B_subseq_isconstant=isconstant_custom_func,
        )

        np.random.seed(seed)
        comp_P, comp_I = prescrump(
            T_B, m, s=s, T_A_subseq_isconstant=isconstant_custom_func
        )

        npt.assert_almost_equal(ref_P, comp_P)
        npt.assert_almost_equal(ref_I, comp_I)


def test_scrump_int_input():
    with pytest.raises(TypeError):
        scrump(np.arange(10), 5, ignore_trivial=True, percentage=1.0, pre_scrump=False)


@pytest.mark.parametrize("T_A, T_B", test_data)
@pytest.mark.parametrize("percentages", percentages)
def test_scrump_self_join(T_A, T_B, percentages):
    m = 3
    zone = int(np.ceil(m / 4))

    for percentage in percentages:
        seed = np.random.randint(100000)

        np.random.seed(seed)
        ref_P, ref_I, ref_left_I, ref_right_I = naive.scrump(
            T_B, m, T_B, percentage, zone, False, None
        )

        np.random.seed(seed)
        approx = scrump(
            T_B, m, ignore_trivial=True, percentage=percentage, pre_scrump=False
        )
        approx.update()
        comp_P = approx.P_
        comp_I = approx.I_
        comp_left_I = approx.left_I_
        comp_right_I = approx.right_I_

        naive.replace_inf(ref_P)
        naive.replace_inf(comp_P)
        npt.assert_almost_equal(ref_P, comp_P)
        npt.assert_almost_equal(ref_I, comp_I)
        npt.assert_almost_equal(ref_left_I, comp_left_I)
        npt.assert_almost_equal(ref_right_I, comp_right_I)


@pytest.mark.parametrize("T_A, T_B", test_data)
@pytest.mark.parametrize("percentages", percentages)
def test_scrump_A_B_join(T_A, T_B, percentages):
    m = 3

    for percentage in percentages:
        seed = np.random.randint(100000)

        np.random.seed(seed)
        ref_P, ref_I, ref_left_I, ref_right_I = naive.scrump(
            T_A, m, T_B, percentage, None, False, None
        )

        np.random.seed(seed)
        approx = scrump(
            T_A, m, T_B, ignore_trivial=False, percentage=percentage, pre_scrump=False
        )
        approx.update()
        comp_P = approx.P_
        comp_I = approx.I_
        comp_left_I = approx.left_I_
        comp_right_I = approx.right_I_

        naive.replace_inf(ref_P)
        naive.replace_inf(comp_P)

        npt.assert_almost_equal(ref_P, comp_P)
        npt.assert_almost_equal(ref_I, comp_I)
        npt.assert_almost_equal(ref_left_I, comp_left_I)
        npt.assert_almost_equal(ref_right_I, comp_right_I)


@pytest.mark.parametrize("T_A, T_B", test_data)
@pytest.mark.parametrize("percentages", percentages)
def test_scrump_A_B_join_swap(T_A, T_B, percentages):
    m = 3

    for percentage in percentages:
        seed = np.random.randint(100000)

        np.random.seed(seed)
        ref_P, _, ref_left_I, ref_right_I = naive.scrump(
            T_B, m, T_A, percentage, None, False, None
        )

        np.random.seed(seed)
        approx = scrump(
            T_B, m, T_A, ignore_trivial=False, percentage=percentage, pre_scrump=False
        )
        approx.update()
        comp_P = approx.P_
        # comp_I = approx.I_
        comp_left_I = approx.left_I_
        comp_right_I = approx.right_I_

        naive.replace_inf(ref_P)
        naive.replace_inf(comp_P)

        npt.assert_almost_equal(ref_P, comp_P)
        npt.assert_almost_equal(ref_P, comp_P)
        npt.assert_almost_equal(ref_left_I, comp_left_I)
        npt.assert_almost_equal(ref_right_I, comp_right_I)


@pytest.mark.parametrize("T_A, T_B", test_data)
@pytest.mark.parametrize("m", window_size)
@pytest.mark.parametrize("percentages", percentages)
def test_scrump_self_join_larger_window(T_A, T_B, m, percentages):
    if len(T_B) > m:
        zone = int(np.ceil(m / 4))

        for percentage in percentages:
            seed = np.random.randint(100000)

            np.random.seed(seed)
            ref_P, ref_I, ref_left_I, ref_right_I = naive.scrump(
                T_B, m, T_B, percentage, zone, False, None
            )

            np.random.seed(seed)
            approx = scrump(
                T_B, m, ignore_trivial=True, percentage=percentage, pre_scrump=False
            )
            approx.update()
            comp_P = approx.P_
            comp_I = approx.I_
            comp_left_I = approx.left_I_
            comp_right_I = approx.right_I_

            naive.replace_inf(ref_P)
            naive.replace_inf(comp_P)

            npt.assert_almost_equal(ref_P, comp_P)
            npt.assert_almost_equal(ref_I, comp_I)
            npt.assert_almost_equal(ref_left_I, comp_left_I)
            npt.assert_almost_equal(ref_right_I, comp_right_I)


@pytest.mark.parametrize("T_A, T_B", test_data)
@pytest.mark.parametrize("percentages", percentages)
def test_scrump_self_join_with_isconstant(T_A, T_B, percentages):
    isconstant_custom_func = functools.partial(
        naive.isconstant_func_stddev_threshold, quantile_threshold=0.05
    )

    m = 3
    zone = int(np.ceil(m / 4))

    for percentage in percentages:
        seed = np.random.randint(100000)

        np.random.seed(seed)
        ref_P, ref_I, ref_left_I, ref_right_I = naive.scrump(
            T_B,
            m,
            T_B,
            percentage,
            zone,
            False,
            None,
            T_A_subseq_isconstant=isconstant_custom_func,
            T_B_subseq_isconstant=isconstant_custom_func,
        )

        np.random.seed(seed)
        approx = scrump(
            T_B,
            m,
            ignore_trivial=True,
            percentage=percentage,
            pre_scrump=False,
            T_A_subseq_isconstant=isconstant_custom_func,
        )
        approx.update()
        comp_P = approx.P_
        comp_I = approx.I_
        comp_left_I = approx.left_I_
        comp_right_I = approx.right_I_

        naive.replace_inf(ref_P)
        naive.replace_inf(comp_P)
        npt.assert_almost_equal(ref_P, comp_P)
        npt.assert_almost_equal(ref_I, comp_I)
        npt.assert_almost_equal(ref_left_I, comp_left_I)
        npt.assert_almost_equal(ref_right_I, comp_right_I)


@pytest.mark.parametrize("T_A, T_B", test_data)
def test_scrump_self_join_full(T_A, T_B):
    m = 3
    zone = int(np.ceil(m / 4))

    ref_mp = naive.stump(T_B, m, exclusion_zone=zone, row_wise=True)
    ref_P = ref_mp[:, 0]
    ref_I = ref_mp[:, 1]
    ref_left_I = ref_mp[:, 2]
    ref_right_I = ref_mp[:, 3]

    approx = scrump(T_B, m, ignore_trivial=True, percentage=1.0, pre_scrump=False)
    approx.update()
    comp_P = approx.P_
    comp_I = approx.I_
    comp_left_I = approx.left_I_
    comp_right_I = approx.right_I_

    naive.replace_inf(ref_P)
    naive.replace_inf(comp_P)

    npt.assert_almost_equal(ref_P, comp_P)
    npt.assert_almost_equal(ref_I, comp_I)
    npt.assert_almost_equal(ref_left_I, comp_left_I)
    npt.assert_almost_equal(ref_right_I, comp_right_I)

    ref_mp = stump(T_B, m, ignore_trivial=True)
    ref_P = ref_mp[:, 0]
    ref_I = ref_mp[:, 1]
    ref_left_I = ref_mp[:, 2]
    ref_right_I = ref_mp[:, 3]

    npt.assert_almost_equal(ref_P, comp_P)
    npt.assert_almost_equal(ref_I, comp_I)
    npt.assert_almost_equal(ref_left_I, comp_left_I)
    npt.assert_almost_equal(ref_right_I, comp_right_I)


@pytest.mark.parametrize("T_A, T_B", test_data)
def test_scrump_A_B_join_full(T_A, T_B):
    m = 3

    ref_mp = naive.stump(T_A, m, T_B=T_B, row_wise=True)
    ref_P = ref_mp[:, 0]
    ref_I = ref_mp[:, 1]
    ref_left_I = ref_mp[:, 2]
    ref_right_I = ref_mp[:, 3]

    approx = scrump(T_A, m, T_B, ignore_trivial=False, percentage=1.0, pre_scrump=False)
    approx.update()
    comp_P = approx.P_
    comp_I = approx.I_
    comp_left_I = approx.left_I_
    comp_right_I = approx.right_I_

    naive.replace_inf(ref_P)
    naive.replace_inf(comp_P)

    npt.assert_almost_equal(ref_P, comp_P)
    npt.assert_almost_equal(ref_I, comp_I)
    npt.assert_almost_equal(ref_left_I, comp_left_I)
    npt.assert_almost_equal(ref_right_I, comp_right_I)

    ref_mp = stump(T_A, m, T_B=T_B, ignore_trivial=False)
    ref_P = ref_mp[:, 0]
    ref_I = ref_mp[:, 1]
    ref_left_I = ref_mp[:, 2]
    ref_right_I = ref_mp[:, 3]

    npt.assert_almost_equal(ref_P, comp_P)
    npt.assert_almost_equal(ref_I, comp_I)
    npt.assert_almost_equal(ref_left_I, comp_left_I)
    npt.assert_almost_equal(ref_right_I, comp_right_I)


@pytest.mark.parametrize("T_A, T_B", test_data)
def test_scrump_A_B_join_full_swap(T_A, T_B):
    m = 3

    ref_mp = naive.stump(T_B, m, T_B=T_A, row_wise=True)
    ref_P = ref_mp[:, 0]
    ref_I = ref_mp[:, 1]
    ref_left_I = ref_mp[:, 2]
    ref_right_I = ref_mp[:, 3]

    approx = scrump(T_B, m, T_A, ignore_trivial=False, percentage=1.0, pre_scrump=False)
    approx.update()
    comp_P = approx.P_
    comp_I = approx.I_
    comp_left_I = approx.left_I_
    comp_right_I = approx.right_I_

    naive.replace_inf(ref_P)
    naive.replace_inf(comp_P)

    npt.assert_almost_equal(ref_P, comp_P)
    npt.assert_almost_equal(ref_I, comp_I)
    npt.assert_almost_equal(ref_left_I, comp_left_I)
    npt.assert_almost_equal(ref_right_I, comp_right_I)


@pytest.mark.parametrize("T_A, T_B", test_data)
@pytest.mark.parametrize("m", window_size)
def test_scrump_self_join_full_larger_window(T_A, T_B, m):
    if len(T_B) > m:
        zone = int(np.ceil(m / 4))

        ref_mp = naive.stump(T_B, m, exclusion_zone=zone, row_wise=True)
        ref_P = ref_mp[:, 0]
        ref_I = ref_mp[:, 1]
        ref_left_I = ref_mp[:, 2]
        ref_right_I = ref_mp[:, 3]

        approx = scrump(T_B, m, ignore_trivial=True, percentage=1.0, pre_scrump=False)
        approx.update()
        comp_P = approx.P_
        comp_I = approx.I_
        comp_left_I = approx.left_I_
        comp_right_I = approx.right_I_

        naive.replace_inf(ref_P)
        naive.replace_inf(comp_P)

        npt.assert_almost_equal(ref_P, comp_P)
        npt.assert_almost_equal(ref_I, comp_I)
        npt.assert_almost_equal(ref_left_I, comp_left_I)
        npt.assert_almost_equal(ref_right_I, comp_right_I)


@pytest.mark.parametrize("T_A, T_B", test_data)
@pytest.mark.parametrize("percentages", percentages)
def test_scrump_plus_plus_self_join(T_A, T_B, percentages):
    m = 3
    zone = int(np.ceil(m / 4))

    for s in range(1, zone + 1):
        for percentage in percentages:
            seed = np.random.randint(100000)

            np.random.seed(seed)
            ref_P, ref_I = naive.prescrump(T_B, m, T_B, s=s, exclusion_zone=zone, k=1)
            ref_P_aux, ref_I_aux, _, _ = naive.scrump(
                T_B, m, T_B, percentage, zone, True, s, k=1
            )

            naive.merge_topk_PI(ref_P, ref_P_aux, ref_I, ref_I_aux)

            np.random.seed(seed)
            approx = scrump(
                T_B, m, ignore_trivial=True, percentage=percentage, pre_scrump=True, s=s
            )
            approx.update()
            comp_P = approx.P_
            comp_I = approx.I_

            naive.replace_inf(ref_P)
            naive.replace_inf(comp_P)

            ref_P = ref_P
            ref_I = ref_I
            npt.assert_almost_equal(ref_P, comp_P)
            npt.assert_almost_equal(ref_I, comp_I)


@pytest.mark.parametrize("T_A, T_B", test_data)
@pytest.mark.parametrize("percentages", percentages)
def test_scrump_plus_plus_A_B_join(T_A, T_B, percentages):
    m = 3
    zone = int(np.ceil(m / 4))

    for s in range(1, zone + 1):
        for percentage in percentages:
            seed = np.random.randint(100000)

            np.random.seed(seed)
            ref_P, ref_I = naive.prescrump(T_A, m, T_B, s=s, k=1)

            ref_P_aux, ref_I_aux, ref_left_I_aux, ref_right_I_aux = naive.scrump(
                T_A, m, T_B, percentage, None, False, None, k=1
            )

            naive.merge_topk_PI(ref_P, ref_P_aux, ref_I, ref_I_aux)
            ref_left_I = ref_left_I_aux
            ref_right_I = ref_right_I_aux

            approx = scrump(
                T_A,
                m,
                T_B,
                ignore_trivial=False,
                percentage=percentage,
                pre_scrump=True,
                s=s,
            )
            approx.update()
            comp_P = approx.P_
            comp_I = approx.I_
            comp_left_I = approx.left_I_
            comp_right_I = approx.right_I_

            naive.replace_inf(ref_P)
            naive.replace_inf(comp_P)

            ref_P = ref_P
            ref_I = ref_I
            npt.assert_almost_equal(ref_P, comp_P)
            npt.assert_almost_equal(ref_I, comp_I)
            npt.assert_almost_equal(ref_left_I, comp_left_I)
            npt.assert_almost_equal(ref_right_I, comp_right_I)


@pytest.mark.parametrize("T_A, T_B", test_data)
def test_scrump_plus_plus_self_join_full(T_A, T_B):
    m = 3
    zone = int(np.ceil(m / 4))

    ref_mp = naive.stump(T_B, m, exclusion_zone=zone, row_wise=True)
    ref_P = ref_mp[:, 0]
    ref_I = ref_mp[:, 1]
    ref_left_I = ref_mp[:, 2]
    ref_right_I = ref_mp[:, 3]

    approx = scrump(
        T_B, m, ignore_trivial=True, percentage=1.0, pre_scrump=True, s=zone
    )
    approx.update()
    comp_P = approx.P_
    comp_I = approx.I_
    comp_left_I = approx.left_I_
    comp_right_I = approx.right_I_

    naive.replace_inf(ref_P)
    naive.replace_inf(comp_P)

    npt.assert_almost_equal(ref_P, comp_P)
    npt.assert_almost_equal(ref_I, comp_I)
    npt.assert_almost_equal(ref_left_I, comp_left_I)
    npt.assert_almost_equal(ref_right_I, comp_right_I)


@pytest.mark.parametrize("T_A, T_B", test_data)
def test_scrump_plus_plus_A_B_join_full(T_A, T_B):
    m = 3
    zone = int(np.ceil(m / 4))

    ref_mp = naive.stump(T_A, m, T_B=T_B, row_wise=True)
    ref_P = ref_mp[:, 0]
    ref_I = ref_mp[:, 1]
    ref_left_I = ref_mp[:, 2]
    ref_right_I = ref_mp[:, 3]

    approx = scrump(
        T_A, m, T_B=T_B, ignore_trivial=False, percentage=1.0, pre_scrump=True, s=zone
    )
    approx.update()
    comp_P = approx.P_
    comp_I = approx.I_
    comp_left_I = approx.left_I_
    comp_right_I = approx.right_I_

    naive.replace_inf(ref_P)
    naive.replace_inf(comp_P)

    npt.assert_almost_equal(ref_P, comp_P)
    npt.assert_almost_equal(ref_I, comp_I)
    npt.assert_almost_equal(ref_left_I, comp_left_I)
    npt.assert_almost_equal(ref_right_I, comp_right_I)


@pytest.mark.parametrize("T_A, T_B", test_data)
def test_scrump_plus_plus_A_B_join_full_swap(T_A, T_B):
    m = 3
    zone = int(np.ceil(m / 4))

    ref_mp = naive.stump(T_B, m, T_B=T_A, row_wise=True)
    ref_P = ref_mp[:, 0]
    ref_I = ref_mp[:, 1]
    ref_left_I = ref_mp[:, 2]
    ref_right_I = ref_mp[:, 3]

    approx = scrump(
        T_B, m, T_B=T_A, ignore_trivial=False, percentage=1.0, pre_scrump=True, s=zone
    )
    approx.update()
    comp_P = approx.P_
    comp_I = approx.I_
    comp_left_I = approx.left_I_
    comp_right_I = approx.right_I_

    naive.replace_inf(ref_P)
    naive.replace_inf(comp_P)

    npt.assert_almost_equal(ref_P, comp_P)
    npt.assert_almost_equal(ref_I, comp_I)
    npt.assert_almost_equal(ref_left_I, comp_left_I)
    npt.assert_almost_equal(ref_right_I, comp_right_I)


@pytest.mark.parametrize("percentages", percentages)
def test_scrump_constant_subsequence_self_join(percentages):
    T = np.concatenate((np.zeros(20, dtype=np.float64), np.ones(5, dtype=np.float64)))

    m = 3
    zone = int(np.ceil(m / 4))

    for percentage in percentages:
        seed = np.random.randint(100000)

        np.random.seed(seed)
        ref_P, ref_I, ref_left_I, ref_right_I = naive.scrump(
            T, m, T, percentage, zone, False, None
        )

        np.random.seed(seed)
        approx = scrump(
            T, m, ignore_trivial=True, percentage=percentage, pre_scrump=False
        )
        approx.update()
        comp_P = approx.P_
        comp_I = approx.I_
        comp_left_I = approx.left_I_
        comp_right_I = approx.right_I_

        naive.replace_inf(ref_P)
        naive.replace_inf(comp_P)

        npt.assert_almost_equal(ref_P, comp_P)
        npt.assert_almost_equal(ref_I, comp_I)
        npt.assert_almost_equal(ref_left_I, comp_left_I)
        npt.assert_almost_equal(ref_right_I, comp_right_I)


@pytest.mark.parametrize("percentages", percentages)
def test_scrump_identical_subsequence_self_join(percentages):
    identical = np.random.rand(8)
    T = np.random.rand(20)
    T[1 : 1 + identical.shape[0]] = identical
    T[11 : 11 + identical.shape[0]] = identical
    m = 3
    zone = int(np.ceil(m / 4))

    for percentage in percentages:
        seed = np.random.randint(100000)

        np.random.seed(seed)
        ref_P, _, _, _ = naive.scrump(T, m, T, percentage, zone, False, None)

        np.random.seed(seed)
        approx = scrump(
            T, m, ignore_trivial=True, percentage=percentage, pre_scrump=False
        )
        approx.update()
        comp_P = approx.P_
        # comp_I = approx.I_
        # comp_left_I = approx.left_I_
        # comp_right_I = approx.right_I_

        naive.replace_inf(ref_P)
        naive.replace_inf(comp_P)

        npt.assert_almost_equal(ref_P, comp_P, decimal=config.STUMPY_TEST_PRECISION)
        # npt.assert_almost_equal(ref_I, comp_I)
        # npt.assert_almost_equal(ref_left_I, comp_left_I)
        # npt.assert_almost_equal(ref_right_I, comp_right_I)


@pytest.mark.parametrize("T_A, T_B", test_data)
@pytest.mark.parametrize("substitute", substitution_values)
@pytest.mark.parametrize("substitution_locations", substitution_locations)
@pytest.mark.parametrize("percentages", percentages)
def test_scrump_nan_inf_self_join(
    T_A, T_B, substitute, substitution_locations, percentages
):
    m = 3

    T_B_sub = T_B.copy()

    for substitution_location in substitution_locations:
        T_B_sub[:] = T_B[:]
        T_B_sub[substitution_location] = substitute

        zone = int(np.ceil(m / 4))

        for percentage in percentages:
            seed = np.random.randint(100000)

            np.random.seed(seed)
            ref_P, ref_I, ref_left_I, ref_right_I = naive.scrump(
                T_B_sub, m, T_B_sub, percentage, zone, False, None
            )

            np.random.seed(seed)
            approx = scrump(T_B_sub, m, percentage=percentage, pre_scrump=False)
            approx.update()
            comp_P = approx.P_
            comp_I = approx.I_
            comp_left_I = approx.left_I_
            comp_right_I = approx.right_I_

            naive.replace_inf(ref_P)
            naive.replace_inf(comp_P)

            npt.assert_almost_equal(ref_P, comp_P)
            npt.assert_almost_equal(ref_I, comp_I)
            npt.assert_almost_equal(ref_left_I, comp_left_I)
            npt.assert_almost_equal(ref_right_I, comp_right_I)


@pytest.mark.parametrize("percentages", percentages)
def test_scrump_nan_zero_mean_self_join(percentages):
    T = np.array([-1, 0, 1, np.inf, 1, 0, -1])

    m = 3
    zone = int(np.ceil(m / 4))

    for percentage in percentages:
        seed = np.random.randint(100000)

        np.random.seed(seed)
        ref_P, ref_I, ref_left_I, ref_right_I = naive.scrump(
            T, m, T, percentage, zone, False, None
        )

        np.random.seed(seed)
        approx = scrump(T, m, percentage=percentage, pre_scrump=False)
        approx.update()
        comp_P = approx.P_
        comp_I = approx.I_
        comp_left_I = approx.left_I_
        comp_right_I = approx.right_I_

        naive.replace_inf(ref_P)
        naive.replace_inf(comp_P)

        npt.assert_almost_equal(ref_P, comp_P)
        npt.assert_almost_equal(ref_I, comp_I)
        npt.assert_almost_equal(ref_left_I, comp_left_I)
        npt.assert_almost_equal(ref_right_I, comp_right_I)


@pytest.mark.parametrize("T_A, T_B", test_data)
def test_prescrump_A_B_join_larger_window(T_A, T_B):
    m = 5
    zone = int(np.ceil(m / 4))
    if len(T_A) > m and len(T_B) > m:
        for s in range(1, zone + 1):
            seed = np.random.randint(100000)

            np.random.seed(seed)
            ref_P, ref_I = naive.prescrump(T_A, m, T_B, s=s)

            np.random.seed(seed)
            comp_P, comp_I = prescrump(T_A, m, T_B, s=s)

            npt.assert_almost_equal(ref_P, comp_P)
            npt.assert_almost_equal(ref_I, comp_I)


@pytest.mark.parametrize("T_A, T_B", test_data)
def test_prescrump_self_join_KNN(T_A, T_B):
    m = 3
    zone = int(np.ceil(m / 4))
    for k in range(2, 4):
        for s in range(1, zone + 1):
            seed = np.random.randint(100000)

            np.random.seed(seed)
            ref_P, ref_I = naive.prescrump(T_B, m, T_B, s=s, exclusion_zone=zone, k=k)

            np.random.seed(seed)
            comp_P, comp_I = prescrump(T_B, m, s=s, k=k)

            npt.assert_almost_equal(ref_I, comp_I)
            npt.assert_almost_equal(ref_P, comp_P)


@pytest.mark.parametrize("T_A, T_B", test_data)
def test_prescrump_A_B_join_KNN(T_A, T_B):
    m = 3
    zone = int(np.ceil(m / 4))
    for k in range(2, 4):
        for s in range(1, zone + 1):
            seed = np.random.randint(100000)

            np.random.seed(seed)
            ref_P, ref_I = naive.prescrump(T_A, m, T_B, s=s)

            np.random.seed(seed)
            comp_P, comp_I = prescrump(T_A, m, T_B=T_B, s=s)

            npt.assert_almost_equal(ref_P, comp_P)
            npt.assert_almost_equal(ref_I, comp_I)


@pytest.mark.parametrize("T_A, T_B", test_data)
@pytest.mark.parametrize("percentages", percentages)
def test_scrump_self_join_KNN(T_A, T_B, percentages):
    m = 3
    zone = int(np.ceil(m / 4))

    for k in range(2, 4):
        for percentage in percentages:
            seed = np.random.randint(100000)

            np.random.seed(seed)
            ref_P, ref_I, ref_left_I, ref_right_I = naive.scrump(
                T_B, m, T_B, percentage, zone, False, None, k=k
            )

            np.random.seed(seed)
            approx = scrump(
                T_B,
                m,
                ignore_trivial=True,
                percentage=percentage,
                pre_scrump=False,
                k=k,
            )
            approx.update()
            comp_P = approx.P_
            comp_I = approx.I_
            comp_left_I = approx.left_I_
            comp_right_I = approx.right_I_

            naive.replace_inf(ref_P)
            naive.replace_inf(comp_P)
            npt.assert_almost_equal(ref_P, comp_P)
            npt.assert_almost_equal(ref_I, comp_I)
            npt.assert_almost_equal(ref_left_I, comp_left_I)
            npt.assert_almost_equal(ref_right_I, comp_right_I)


@pytest.mark.parametrize("T_A, T_B", test_data)
@pytest.mark.parametrize("percentages", percentages)
def test_scrump_A_B_join_KNN(T_A, T_B, percentages):
    m = 3
    for k in range(2, 4):
        for percentage in percentages:
            seed = np.random.randint(100000)

            np.random.seed(seed)
            ref_P, ref_I, ref_left_I, ref_right_I = naive.scrump(
                T_A, m, T_B, percentage, None, False, None, k=k
            )

            np.random.seed(seed)
            approx = scrump(
                T_A,
                m,
                T_B,
                ignore_trivial=False,
                percentage=percentage,
                pre_scrump=False,
                k=k,
            )
            approx.update()
            comp_P = approx.P_
            comp_I = approx.I_
            comp_left_I = approx.left_I_
            comp_right_I = approx.right_I_

            naive.replace_inf(ref_P)
            naive.replace_inf(comp_P)

            npt.assert_almost_equal(ref_P, comp_P)
            npt.assert_almost_equal(ref_I, comp_I)
            npt.assert_almost_equal(ref_left_I, comp_left_I)
            npt.assert_almost_equal(ref_right_I, comp_right_I)


@pytest.mark.parametrize("T_A, T_B", test_data)
@pytest.mark.parametrize("percentages", percentages)
def test_scrump_plus_plus_self_join_KNN(T_A, T_B, percentages):
    m = 3
    zone = int(np.ceil(m / 4))

    for k in range(2, 4):
        for s in range(1, zone + 1):
            for percentage in percentages:
                seed = np.random.randint(100000)

                np.random.seed(seed)
                ref_P, ref_I = naive.prescrump(
                    T_B, m, T_B, s=s, exclusion_zone=zone, k=k
                )
                ref_P_aux, ref_I_aux, _, _ = naive.scrump(
                    T_B, m, T_B, percentage, zone, True, s, k=k
                )
                naive.merge_topk_PI(ref_P, ref_P_aux, ref_I, ref_I_aux)

                np.random.seed(seed)
                approx = scrump(
                    T_B,
                    m,
                    ignore_trivial=True,
                    percentage=percentage,
                    pre_scrump=True,
                    s=s,
                    k=k,
                )
                approx.update()
                comp_P = approx.P_
                comp_I = approx.I_

                naive.replace_inf(ref_P)
                naive.replace_inf(comp_P)

                npt.assert_almost_equal(ref_P, comp_P)
                npt.assert_almost_equal(ref_I, comp_I)


@pytest.mark.parametrize("T_A, T_B", test_data)
def test_prescrump_self_join_larger_window_m_5_k_5(T_A, T_B):
    m = 5
    k = 5
    zone = int(np.ceil(m / 4))

    if len(T_B) > m:
        for s in range(1, zone + 1):
            seed = np.random.randint(100000)

            np.random.seed(seed)
            ref_P, ref_I = naive.prescrump(T_B, m, T_B, s=s, exclusion_zone=zone, k=k)

            np.random.seed(seed)
            comp_P, comp_I = prescrump(T_B, m, s=s, k=k)

            npt.assert_almost_equal(ref_P, comp_P)
            npt.assert_almost_equal(ref_I, comp_I)


@pytest.mark.parametrize("T_A, T_B", test_data)
def test_prescrump_A_B_join_larger_window_m_5_k_5(T_A, T_B):
    m = 5
    k = 5
    zone = int(np.ceil(m / 4))
    if len(T_A) > m and len(T_B) > m:
        for s in range(1, zone + 1):
            seed = np.random.randint(100000)

            np.random.seed(seed)
            ref_P, ref_I = naive.prescrump(T_A, m, T_B, s=s, k=k)

            np.random.seed(seed)
            comp_P, comp_I = prescrump(T_A, m, T_B, s=s, k=k)

            npt.assert_almost_equal(ref_P, comp_P)
            npt.assert_almost_equal(ref_I, comp_I)


def test_prescrump_self_join_KNN_no_overlap():
    # This test is particularly designed to raise error in a rare case described
    # as follows: Let's denote `I[i]` as the array with length `k` that contains
    # the start indices of the best-so-far top-k nearest neighbors of `subseq i`,
    # (`S_i`). Also, we denote `P[i]` as their corresponding distances sorted in
    # ascending order. Let's denote `d` as the distance between `S_i` and `S_j`. P[i]
    # and I[i] must be updated if (1) `j` is not in I[i] and (2) `d` < P[i,-1].
    # Regarding the former condition, one needs to check the whole array I[i]. Checking
    # the array I[i, :idx], where `idx = np.searchsorted(P[i], 'd', side='right')` is
    # not completely correct and that is due to imprecision in numerical calculation.
    # It may happen that `j` is not in `I[i, :idx]`, but it is in fact at `I[i, idx]`
    # (or any other position in array I[i]). And, its corresponding distance, i.e
    # P[i, idx], is d + 1e-5, for instance. In theory, this should be exactly `d`.
    #  However, due to imprecision, we may calculated a slightly different value
    # for such distance in one of previous iterations in function prescrump. This
    #  test results in error if someone tries to change the performant code of prescrump
    # function and check `I[i, :idx]` rather than the full array `I[i]`.
    T = np.array(
        [
            -916.64703784,
            -327.42056679,
            379.19386284,
            -281.80427628,
            -189.85401773,
            -38.69610569,
            187.89889345,
            578.65862523,
            528.09687811,
            -667.42973795,
            -285.27749324,
            -211.28930925,
            -703.93802657,
            -820.53780562,
            -955.91174663,
            383.65471851,
            932.08809422,
            -563.57569746,
            784.0546579,
            -343.14886064,
            -612.72329848,
            -270.09273091,
            -448.39346549,
            578.03202014,
            867.15436674,
            -783.55167049,
            -494.78062922,
            -311.18567747,
            522.70052256,
            933.45474094,
            192.34822368,
            -162.11374908,
            -612.95359279,
            -449.62297051,
            -351.79138459,
            -77.70189101,
            -439.46519487,
            -660.48431174,
            548.69362177,
            485.36004744,
            -535.3566627,
            -568.0955257,
            755.26647273,
            736.1079588,
            -597.65672557,
            379.3299783,
            731.38211912,
            247.34827447,
            545.41888454,
            644.94300763,
            20.99042666,
            788.19859515,
            -898.24325898,
            -929.47841134,
            -738.45875181,
            66.01030291,
            512.945841,
            -44.07720164,
            302.97141464,
            -696.95271302,
            662.98385163,
            -712.3807531,
            -43.62688539,
            74.16927482,
        ]
    )

    # test_cases: dict() with `key: value` pair, where key is `(m, k)`, and value
    # is a list of random `seeds`
    test_cases = {
        (3, 2): [4279, 9133, 8190],
        (3, 5): [1267, 4016, 4046],
        (5, 2): [6327, 4926, 3712],
        (5, 5): [3032, 3032, 8117],
    }
    for (m, k), specified_seeds in test_cases.items():
        zone = int(np.ceil(m / 4))
        for seed in specified_seeds:
            np.random.seed(seed)
            ref_P, ref_I = naive.prescrump(T, m, T, s=1, exclusion_zone=zone, k=k)
            comp_P, comp_I = prescrump(T, m, s=1, k=k)

            npt.assert_almost_equal(ref_P, comp_P)
            npt.assert_almost_equal(ref_I, comp_I)


@pytest.mark.parametrize("T_A, T_B", test_data)
def test_prescrump_self_join_larger_window_m_5_k_5_with_isconstant(T_A, T_B):
    isconstant_custom_func = functools.partial(
        naive.isconstant_func_stddev_threshold, quantile_threshold=0.05
    )

    m = 5
    k = 5
    zone = int(np.ceil(m / 4))

    if len(T_B) > m:
        for s in range(1, zone + 1):
            seed = np.random.randint(100000)

            np.random.seed(seed)
            ref_P, ref_I = naive.prescrump(
                T_B,
                m,
                T_B,
                s=s,
                exclusion_zone=zone,
                k=k,
                T_A_subseq_isconstant=isconstant_custom_func,
                T_B_subseq_isconstant=isconstant_custom_func,
            )

            np.random.seed(seed)
            comp_P, comp_I = prescrump(
                T_B, m, s=s, k=k, T_A_subseq_isconstant=isconstant_custom_func
            )

            npt.assert_almost_equal(ref_P, comp_P)
            npt.assert_almost_equal(ref_I, comp_I)
