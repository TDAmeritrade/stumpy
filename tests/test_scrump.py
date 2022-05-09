import numpy as np
import numpy.testing as npt
from stumpy import scrump, stump, config
from stumpy.scrump import prescrump
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
        ref_mp = naive.scrump(T_B, m, T_B, percentage, zone, False, None)
        ref_P = ref_mp[:, 0]
        ref_I = ref_mp[:, 1]
        ref_left_I = ref_mp[:, 2]
        ref_right_I = ref_mp[:, 3]

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
        ref_mp = naive.scrump(T_A, m, T_B, percentage, None, False, None)
        ref_P = ref_mp[:, 0]
        ref_I = ref_mp[:, 1]
        ref_left_I = ref_mp[:, 2]
        ref_right_I = ref_mp[:, 3]

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
        ref_mp = naive.scrump(T_B, m, T_A, percentage, None, False, None)
        ref_P = ref_mp[:, 0]
        # ref_I = ref_mp[:, 1]
        ref_left_I = ref_mp[:, 2]
        ref_right_I = ref_mp[:, 3]

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
            ref_mp = naive.scrump(T_B, m, T_B, percentage, zone, False, None)
            ref_P = ref_mp[:, 0]
            ref_I = ref_mp[:, 1]
            ref_left_I = ref_mp[:, 2]
            ref_right_I = ref_mp[:, 3]

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
            ref_P, ref_I = naive.prescrump(T_B, m, T_B, s=s, exclusion_zone=zone)
            ref_mp = naive.scrump(T_B, m, T_B, percentage, zone, True, s)
            for i in range(ref_mp.shape[0]):
                if ref_P[i] < ref_mp[i, 0]:
                    ref_mp[i, 0] = ref_P[i]
                    ref_mp[i, 1] = ref_I[i]
            ref_P = ref_mp[:, 0]
            ref_I = ref_mp[:, 1]
            # ref_left_I = ref_mp[:, 2]
            # ref_right_I = ref_mp[:, 3]

            np.random.seed(seed)
            approx = scrump(
                T_B, m, ignore_trivial=True, percentage=percentage, pre_scrump=True, s=s
            )
            approx.update()
            comp_P = approx.P_
            comp_I = approx.I_
            # comp_left_I = approx.left_I_
            # comp_right_I = approx.right_I_

            naive.replace_inf(ref_P)
            naive.replace_inf(comp_I)

            npt.assert_almost_equal(ref_P, comp_P)
            npt.assert_almost_equal(ref_I, comp_I)
            # npt.assert_almost_equal(ref_left_I, comp_left_I)
            # npt.assert_almost_equal(ref_right_I, comp_right_I)


@pytest.mark.parametrize("T_A, T_B", test_data)
@pytest.mark.parametrize("percentages", percentages)
def test_scrump_plus_plus_A_B_join(T_A, T_B, percentages):
    m = 3
    zone = int(np.ceil(m / 4))

    for s in range(1, zone + 1):
        for percentage in percentages:
            seed = np.random.randint(100000)

            np.random.seed(seed)
            ref_P, ref_I = naive.prescrump(T_A, m, T_B, s=s)
            ref_mp = naive.scrump(T_A, m, T_B, percentage, None, False, None)
            for i in range(ref_mp.shape[0]):
                if ref_P[i] < ref_mp[i, 0]:
                    ref_mp[i, 0] = ref_P[i]
                    ref_mp[i, 1] = ref_I[i]
            ref_P = ref_mp[:, 0]
            ref_I = ref_mp[:, 1]
            ref_left_I = ref_mp[:, 2]
            ref_right_I = ref_mp[:, 3]

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
        ref_mp = naive.scrump(T, m, T, percentage, zone, False, None)
        ref_P = ref_mp[:, 0]
        ref_I = ref_mp[:, 1]
        ref_left_I = ref_mp[:, 2]
        ref_right_I = ref_mp[:, 3]

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
        ref_mp = naive.scrump(T, m, T, percentage, zone, False, None)
        ref_P = ref_mp[:, 0]
        # ref_I = ref_mp[:, 1]
        # ref_left_I = ref_mp[:, 2]
        # ref_right_I = ref_mp[:, 3]

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
            ref_mp = naive.scrump(T_B_sub, m, T_B_sub, percentage, zone, False, None)
            ref_P = ref_mp[:, 0]
            ref_I = ref_mp[:, 1]
            ref_left_I = ref_mp[:, 2]
            ref_right_I = ref_mp[:, 3]

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
        ref_mp = naive.scrump(T, m, T, percentage, zone, False, None)
        ref_P = ref_mp[:, 0]
        ref_I = ref_mp[:, 1]
        ref_left_I = ref_mp[:, 2]
        ref_right_I = ref_mp[:, 3]

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
