import naive
import numpy as np
import numpy.testing as npt
import pytest

from stumpy import aamp, config, scraamp
from stumpy.scraamp import prescraamp

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
def test_prescraamp_self_join(T_A, T_B):
    for p in [1.0, 2.0, 3.0]:
        m = 3
        zone = int(np.ceil(m / 4))
        for s in range(1, zone + 1):
            seed = np.random.randint(100000)

            np.random.seed(seed)
            ref_P, ref_I = naive.prescraamp(T_B, m, T_B, s=s, exclusion_zone=zone, p=p)

            np.random.seed(seed)
            comp_P, comp_I = prescraamp(T_B, m, s=s, p=p)

            npt.assert_almost_equal(ref_P, comp_P)
            npt.assert_almost_equal(ref_I, comp_I)


@pytest.mark.parametrize("T_A, T_B", test_data)
def test_prescraamp_A_B_join(T_A, T_B):
    for p in [1.0, 2.0, 3.0]:
        m = 3
        zone = int(np.ceil(m / 4))
        for s in range(1, zone + 1):
            seed = np.random.randint(100000)

            np.random.seed(seed)
            ref_P, ref_I = naive.prescraamp(T_A, m, T_B, s=s, p=p)

            np.random.seed(seed)
            comp_P, comp_I = prescraamp(T_A, m, T_B=T_B, s=s, p=p)

            npt.assert_almost_equal(ref_P, comp_P)
            npt.assert_almost_equal(ref_I, comp_I)


@pytest.mark.parametrize("T_A, T_B", test_data)
def test_prescraamp_A_B_join_swap(T_A, T_B):
    m = 3
    zone = int(np.ceil(m / 4))
    for s in range(1, zone + 1):
        seed = np.random.randint(100000)

        np.random.seed(seed)
        ref_P, ref_I = naive.prescraamp(T_B, m, T_A, s=s)

        np.random.seed(seed)
        comp_P, comp_I = prescraamp(T_B, m, T_B=T_A, s=s)

        npt.assert_almost_equal(ref_P, comp_P)
        npt.assert_almost_equal(ref_I, comp_I)


@pytest.mark.parametrize("T_A, T_B", test_data)
@pytest.mark.parametrize("m", window_size)
def test_prescraamp_self_join_larger_window(T_A, T_B, m):
    if len(T_B) > m:
        zone = int(np.ceil(m / 4))
        for s in range(1, zone + 1):
            seed = np.random.randint(100000)

            np.random.seed(seed)
            ref_P, ref_I = naive.prescraamp(T_B, m, T_B, s=s, exclusion_zone=zone)

            np.random.seed(seed)
            comp_P, comp_I = prescraamp(T_B, m, s=s)

            npt.assert_almost_equal(ref_P, comp_P)
            npt.assert_almost_equal(ref_I, comp_I)


def test_scraamp_int_input():
    with pytest.raises(TypeError):
        scraamp(
            np.arange(10), 5, ignore_trivial=True, percentage=1.0, pre_scraamp=False
        )


@pytest.mark.parametrize("T_A, T_B", test_data)
@pytest.mark.parametrize("percentages", percentages)
def test_scraamp_self_join(T_A, T_B, percentages):
    m = 3
    zone = int(np.ceil(m / 4))

    for p in [1.0, 2.0, 3.0]:
        for percentage in percentages:
            seed = np.random.randint(100000)

            np.random.seed(seed)
            ref_P, ref_I, ref_left_I, ref_right_I = naive.scraamp(
                T_B, m, T_B, percentage, zone, False, None, p=p
            )

            np.random.seed(seed)
            approx = scraamp(
                T_B,
                m,
                ignore_trivial=True,
                percentage=percentage,
                pre_scraamp=False,
                p=p,
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
def test_scraamp_A_B_join(T_A, T_B, percentages):
    m = 3

    for p in [1.0, 2.0, 3.0]:
        for percentage in percentages:
            seed = np.random.randint(100000)

            np.random.seed(seed)
            ref_P, ref_I, ref_left_I, ref_right_I = naive.scraamp(
                T_A, m, T_B, percentage, None, False, None, p=p
            )

            np.random.seed(seed)
            approx = scraamp(
                T_A,
                m,
                T_B,
                ignore_trivial=False,
                percentage=percentage,
                pre_scraamp=False,
                p=p,
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
def test_scraamp_A_B_join_swap(T_A, T_B, percentages):
    m = 3

    for percentage in percentages:
        seed = np.random.randint(100000)

        np.random.seed(seed)
        ref_P, _, ref_left_I, ref_right_I = naive.scraamp(
            T_B, m, T_A, percentage, None, False, None
        )

        np.random.seed(seed)
        approx = scraamp(
            T_B, m, T_A, ignore_trivial=False, percentage=percentage, pre_scraamp=False
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
def test_scraamp_self_join_larger_window(T_A, T_B, m, percentages):
    if len(T_B) > m:
        zone = int(np.ceil(m / 4))

        for percentage in percentages:
            seed = np.random.randint(100000)

            np.random.seed(seed)
            ref_P, ref_I, ref_left_I, ref_right_I = naive.scraamp(
                T_B, m, T_B, percentage, zone, False, None
            )

            np.random.seed(seed)
            approx = scraamp(
                T_B, m, ignore_trivial=True, percentage=percentage, pre_scraamp=False
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
def test_scraamp_self_join_full(T_A, T_B):
    m = 3
    zone = int(np.ceil(m / 4))

    ref_mp = naive.aamp(T_B, m, exclusion_zone=zone)
    ref_P = ref_mp[:, 0]
    ref_I = ref_mp[:, 1]
    ref_left_I = ref_mp[:, 2]
    ref_right_I = ref_mp[:, 3]

    approx = scraamp(T_B, m, ignore_trivial=True, percentage=1.0, pre_scraamp=False)
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

    ref_mp = aamp(T_B, m, ignore_trivial=True)
    ref_P = ref_mp[:, 0]
    ref_I = ref_mp[:, 1]
    ref_left_I = ref_mp[:, 2]
    ref_right_I = ref_mp[:, 3]

    npt.assert_almost_equal(ref_P, comp_P)
    npt.assert_almost_equal(ref_I, comp_I)
    npt.assert_almost_equal(ref_left_I, comp_left_I)
    npt.assert_almost_equal(ref_right_I, comp_right_I)


@pytest.mark.parametrize("T_A, T_B", test_data)
def test_scraamp_A_B_join_full(T_A, T_B):
    m = 3

    ref_mp = naive.aamp(T_A, m, T_B=T_B)
    ref_P = ref_mp[:, 0]
    ref_I = ref_mp[:, 1]
    ref_left_I = ref_mp[:, 2]
    ref_right_I = ref_mp[:, 3]

    approx = scraamp(
        T_A, m, T_B, ignore_trivial=False, percentage=1.0, pre_scraamp=False
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

    ref_mp = aamp(T_A, m, T_B=T_B, ignore_trivial=False)
    ref_P = ref_mp[:, 0]
    ref_I = ref_mp[:, 1]
    ref_left_I = ref_mp[:, 2]
    ref_right_I = ref_mp[:, 3]

    npt.assert_almost_equal(ref_P, comp_P)
    npt.assert_almost_equal(ref_I, comp_I)
    npt.assert_almost_equal(ref_left_I, comp_left_I)
    npt.assert_almost_equal(ref_right_I, comp_right_I)


@pytest.mark.parametrize("T_A, T_B", test_data)
def test_scraamp_A_B_join_full_swap(T_A, T_B):
    m = 3

    ref_mp = naive.aamp(T_B, m, T_B=T_A)
    ref_P = ref_mp[:, 0]
    ref_I = ref_mp[:, 1]
    ref_left_I = ref_mp[:, 2]
    ref_right_I = ref_mp[:, 3]

    approx = scraamp(
        T_B, m, T_A, ignore_trivial=False, percentage=1.0, pre_scraamp=False
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
@pytest.mark.parametrize("m", window_size)
def test_scraamp_self_join_full_larger_window(T_A, T_B, m):
    if len(T_B) > m:
        zone = int(np.ceil(m / 4))

        ref_mp = naive.aamp(T_B, m, exclusion_zone=zone)
        ref_P = ref_mp[:, 0]
        ref_I = ref_mp[:, 1]
        ref_left_I = ref_mp[:, 2]
        ref_right_I = ref_mp[:, 3]

        approx = scraamp(T_B, m, ignore_trivial=True, percentage=1.0, pre_scraamp=False)
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
def test_scraamp_plus_plus_self_join(T_A, T_B, percentages):
    m = 3
    zone = int(np.ceil(m / 4))

    for p in [1.0, 2.0, 3.0]:
        for s in range(1, zone + 1):
            for percentage in percentages:
                seed = np.random.randint(100000)

                np.random.seed(seed)
                ref_P, ref_I = naive.prescraamp(
                    T_B, m, T_B, s=s, exclusion_zone=zone, p=p
                )
                ref_P_aux, ref_I_aux, _, _ = naive.scraamp(
                    T_B, m, T_B, percentage, zone, True, s, p=p
                )

                naive.merge_topk_PI(ref_P, ref_P_aux, ref_I, ref_I_aux)

                np.random.seed(seed)
                approx = scraamp(
                    T_B,
                    m,
                    ignore_trivial=True,
                    percentage=percentage,
                    pre_scraamp=True,
                    s=s,
                    p=p,
                )
                approx.update()
                comp_P = approx.P_
                comp_I = approx.I_
                # comp_left_I = approx.left_I_
                # comp_right_I = approx.right_I_

                naive.replace_inf(ref_P)
                naive.replace_inf(comp_P)

                npt.assert_almost_equal(ref_P, comp_P)
                npt.assert_almost_equal(ref_I, comp_I)
                # npt.assert_almost_equal(ref_left_I, comp_left_I)
                # npt.assert_almost_equal(ref_right_I, comp_right_I)


@pytest.mark.parametrize("T_A, T_B", test_data)
@pytest.mark.parametrize("percentages", percentages)
def test_scraamp_plus_plus_A_B_join(T_A, T_B, percentages):
    m = 3
    zone = int(np.ceil(m / 4))

    for p in [1.0, 2.0, 3.0]:
        for s in range(1, zone + 1):
            for percentage in percentages:
                seed = np.random.randint(100000)

                np.random.seed(seed)
                ref_P, ref_I = naive.prescraamp(T_A, m, T_B, s=s, p=p)
                ref_P_aux, ref_I_aux, ref_left_I_aux, ref_right_I_aux = naive.scraamp(
                    T_A, m, T_B, percentage, None, False, None, p=p, k=1
                )

                naive.merge_topk_PI(ref_P, ref_P_aux, ref_I, ref_I_aux)
                ref_left_I = ref_left_I_aux
                ref_right_I = ref_right_I_aux

                approx = scraamp(
                    T_A,
                    m,
                    T_B,
                    ignore_trivial=False,
                    percentage=percentage,
                    pre_scraamp=True,
                    s=s,
                    p=p,
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
def test_scraamp_plus_plus_self_join_full(T_A, T_B):
    m = 3
    zone = int(np.ceil(m / 4))

    ref_mp = naive.aamp(T_B, m, exclusion_zone=zone)
    ref_P = ref_mp[:, 0]
    ref_I = ref_mp[:, 1]
    ref_left_I = ref_mp[:, 2]
    ref_right_I = ref_mp[:, 3]

    approx = scraamp(
        T_B, m, ignore_trivial=True, percentage=1.0, pre_scraamp=True, s=zone
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
def test_scraamp_plus_plus_A_B_join_full(T_A, T_B):
    m = 3
    zone = int(np.ceil(m / 4))

    ref_mp = naive.aamp(T_A, m, T_B=T_B)
    ref_P = ref_mp[:, 0]
    ref_I = ref_mp[:, 1]
    ref_left_I = ref_mp[:, 2]
    ref_right_I = ref_mp[:, 3]

    approx = scraamp(
        T_A, m, T_B=T_B, ignore_trivial=False, percentage=1.0, pre_scraamp=True, s=zone
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
def test_scraamp_plus_plus_A_B_join_full_swap(T_A, T_B):
    m = 3
    zone = int(np.ceil(m / 4))

    ref_mp = naive.aamp(T_B, m, T_B=T_A)
    ref_P = ref_mp[:, 0]
    ref_I = ref_mp[:, 1]
    ref_left_I = ref_mp[:, 2]
    ref_right_I = ref_mp[:, 3]

    approx = scraamp(
        T_B, m, T_B=T_A, ignore_trivial=False, percentage=1.0, pre_scraamp=True, s=zone
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
def test_scraamp_constant_subsequence_self_join(percentages):
    T = np.concatenate((np.zeros(20, dtype=np.float64), np.ones(5, dtype=np.float64)))

    m = 3
    zone = int(np.ceil(m / 4))

    for percentage in percentages:
        seed = np.random.randint(100000)

        np.random.seed(seed)
        ref_P, _, _, _ = naive.scraamp(T, m, T, percentage, zone, False, None)

        np.random.seed(seed)
        approx = scraamp(
            T, m, ignore_trivial=True, percentage=percentage, pre_scraamp=False
        )
        approx.update()
        comp_P = approx.P_
        # comp_I = approx.I_
        # comp_left_I = approx.left_I_
        # comp_right_I = approx.right_I_

        naive.replace_inf(ref_P)
        naive.replace_inf(comp_P)

        npt.assert_almost_equal(ref_P, comp_P)
        # npt.assert_almost_equal(ref_I, comp_I)
        # npt.assert_almost_equal(ref_left_I, comp_left_I)
        # npt.assert_almost_equal(ref_right_I, comp_right_I)


@pytest.mark.parametrize("percentages", percentages)
def test_scraamp_identical_subsequence_self_join(percentages):
    identical = np.random.rand(8)
    T = np.random.rand(20)
    T[1 : 1 + identical.shape[0]] = identical
    T[11 : 11 + identical.shape[0]] = identical
    m = 3
    zone = int(np.ceil(m / 4))

    for percentage in percentages:
        seed = np.random.randint(100000)

        np.random.seed(seed)
        ref_P, _, _, _ = naive.scraamp(T, m, T, percentage, zone, False, None)

        np.random.seed(seed)
        approx = scraamp(
            T, m, ignore_trivial=True, percentage=percentage, pre_scraamp=False
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
def test_scraamp_nan_inf_self_join(
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
            ref_P, ref_I, ref_left_I, ref_right_I = naive.scraamp(
                T_B_sub, m, T_B_sub, percentage, zone, False, None
            )

            np.random.seed(seed)
            approx = scraamp(T_B_sub, m, percentage=percentage, pre_scraamp=False)
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
def test_scraamp_nan_zero_mean_self_join(percentages):
    T = np.array([-1, 0, 1, np.inf, 1, 0, -1])

    m = 3
    zone = int(np.ceil(m / 4))

    for percentage in percentages:
        seed = np.random.randint(100000)

        np.random.seed(seed)
        ref_P, ref_I, ref_left_I, ref_right_I = naive.scraamp(
            T, m, T, percentage, zone, False, None
        )

        np.random.seed(seed)
        approx = scraamp(T, m, percentage=percentage, pre_scraamp=False)
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
def test_prescraamp_A_B_join_larger_window(T_A, T_B):
    m = 5
    zone = int(np.ceil(m / 4))
    if len(T_A) > m and len(T_B) > m:
        for s in range(1, zone + 1):
            seed = np.random.randint(100000)

            np.random.seed(seed)
            ref_P, ref_I = naive.prescraamp(T_A, m, T_B, s=s)

            np.random.seed(seed)
            comp_P, comp_I = prescraamp(T_A, m, T_B, s=s)

            npt.assert_almost_equal(ref_P, comp_P)
            npt.assert_almost_equal(ref_I, comp_I)


@pytest.mark.parametrize("T_A, T_B", test_data)
def test_prescraamp_self_join_KNN(T_A, T_B):
    m = 3
    zone = int(np.ceil(m / 4))
    for k in range(2, 4):
        for p in [1.0, 2.0, 3.0]:
            for s in range(1, zone + 1):
                seed = np.random.randint(100000)

                np.random.seed(seed)
                ref_P, ref_I = naive.prescraamp(
                    T_B, m, T_B, s=s, exclusion_zone=zone, p=p, k=k
                )

                np.random.seed(seed)
                comp_P, comp_I = prescraamp(T_B, m, s=s, p=p, k=k)

                npt.assert_almost_equal(ref_P, comp_P)
                npt.assert_almost_equal(ref_I, comp_I)


@pytest.mark.parametrize("T_A, T_B", test_data)
def test_prescraamp_A_B_join_KNN(T_A, T_B):
    m = 3
    zone = int(np.ceil(m / 4))
    for k in range(2, 4):
        for p in [1.0, 2.0, 3.0]:
            for s in range(1, zone + 1):
                seed = np.random.randint(100000)

                np.random.seed(seed)
                ref_P, ref_I = naive.prescraamp(T_A, m, T_B, s=s, p=p, k=k)

                np.random.seed(seed)
                comp_P, comp_I = prescraamp(T_A, m, T_B=T_B, s=s, p=p, k=k)

                npt.assert_almost_equal(ref_P, comp_P)
                npt.assert_almost_equal(ref_I, comp_I)


@pytest.mark.parametrize("T_A, T_B", test_data)
@pytest.mark.parametrize("percentages", percentages)
def test_scraamp_self_join_KNN(T_A, T_B, percentages):
    m = 3
    zone = int(np.ceil(m / 4))

    for k in range(2, 4):
        for p in [1.0, 2.0, 3.0]:
            for percentage in percentages:
                seed = np.random.randint(100000)

                np.random.seed(seed)
                ref_P, ref_I, ref_left_I, ref_right_I = naive.scraamp(
                    T_B, m, T_B, percentage, zone, False, None, p=p, k=k
                )

                np.random.seed(seed)
                approx = scraamp(
                    T_B,
                    m,
                    ignore_trivial=True,
                    percentage=percentage,
                    pre_scraamp=False,
                    p=p,
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
def test_scraamp_A_B_join_KNN(T_A, T_B, percentages):
    m = 3
    for k in range(2, 4):
        for p in [1.0, 2.0, 3.0]:
            for percentage in percentages:
                seed = np.random.randint(100000)

                np.random.seed(seed)
                ref_P, ref_I, ref_left_I, ref_right_I = naive.scraamp(
                    T_A, m, T_B, percentage, None, False, None, p=p, k=k
                )

                np.random.seed(seed)
                approx = scraamp(
                    T_A,
                    m,
                    T_B,
                    ignore_trivial=False,
                    percentage=percentage,
                    pre_scraamp=False,
                    p=p,
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
def test_scraamp_plus_plus_self_join_KNN(T_A, T_B, percentages):
    m = 3
    zone = int(np.ceil(m / 4))
    for k in range(2, 4):
        for p in [1.0, 2.0, 3.0]:
            for s in range(1, zone + 1):
                for percentage in percentages:
                    seed = np.random.randint(100000)

                    np.random.seed(seed)
                    ref_P, ref_I = naive.prescraamp(
                        T_B, m, T_B, s=s, exclusion_zone=zone, p=p, k=k
                    )
                    ref_P_aux, ref_I_aux, _, _ = naive.scraamp(
                        T_B, m, T_B, percentage, zone, True, s, p=p, k=k
                    )

                    naive.merge_topk_PI(ref_P, ref_P_aux, ref_I, ref_I_aux)

                    np.random.seed(seed)
                    approx = scraamp(
                        T_B,
                        m,
                        ignore_trivial=True,
                        percentage=percentage,
                        pre_scraamp=True,
                        s=s,
                        p=p,
                        k=k,
                    )
                    approx.update()
                    comp_P = approx.P_
                    comp_I = approx.I_
                    # comp_left_I = approx.left_I_
                    # comp_right_I = approx.right_I_

                    naive.replace_inf(ref_P)
                    naive.replace_inf(comp_P)

                    npt.assert_almost_equal(ref_P, comp_P)
                    npt.assert_almost_equal(ref_I, comp_I)
                    # npt.assert_almost_equal(ref_left_I, comp_left_I)
                    # npt.assert_almost_equal(ref_right_I, comp_right_I)


@pytest.mark.parametrize("T_A, T_B", test_data)
@pytest.mark.parametrize("m", window_size)
def test_prescraamp_self_join_larger_window_m_5_k_5(T_A, T_B, m):
    m = 5
    k = 5
    zone = int(np.ceil(m / 4))

    if len(T_B) > m:
        for s in range(1, zone + 1):
            seed = np.random.randint(100000)

            np.random.seed(seed)
            ref_P, ref_I = naive.prescraamp(T_B, m, T_B, s=s, exclusion_zone=zone, k=k)

            np.random.seed(seed)
            comp_P, comp_I = prescraamp(T_B, m, s=s, k=k)

            npt.assert_almost_equal(ref_P, comp_P)
            npt.assert_almost_equal(ref_I, comp_I)


@pytest.mark.parametrize("T_A, T_B", test_data)
def test_prescraamp_A_B_join_larger_window_m_5_k_5(T_A, T_B):
    m = 5
    k = 5
    zone = int(np.ceil(m / 4))

    if len(T_A) > m and len(T_B) > m:
        for s in range(1, zone + 1):
            seed = np.random.randint(100000)

            np.random.seed(seed)
            ref_P, ref_I = naive.prescraamp(T_A, m, T_B, s=s, k=k)

            np.random.seed(seed)
            comp_P, comp_I = prescraamp(T_A, m, T_B, s=s, k=k)

            npt.assert_almost_equal(ref_P, comp_P)
            npt.assert_almost_equal(ref_I, comp_I)
