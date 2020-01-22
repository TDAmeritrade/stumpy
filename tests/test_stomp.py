import numpy as np
import numpy.testing as npt
from stumpy import stomp, core
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


@pytest.mark.parametrize("T_A, T_B", test_data)
def test_stomp_self_join(T_A, T_B):
    m = 3
    zone = int(np.ceil(m / 4))
    left = np.array(
        [
            utils.naive_mass(Q, T_B, m, i, zone, True)
            for i, Q in enumerate(core.rolling_window(T_B, m))
        ],
        dtype=object,
    )
    right = stomp(T_B, m, ignore_trivial=True)
    utils.replace_inf(left)
    utils.replace_inf(right)
    npt.assert_almost_equal(left, right)


@pytest.mark.parametrize("T_A, T_B", test_data)
def test_stump_self_join_larger_window(T_A, T_B):
    for m in [8, 16, 32]:
        if len(T_B) > m:
            zone = int(np.ceil(m / 4))
            left = np.array(
                [
                    utils.naive_mass(Q, T_B, m, i, zone, True)
                    for i, Q in enumerate(core.rolling_window(T_B, m))
                ],
                dtype=object,
            )
            right = stomp(T_B, m, ignore_trivial=True)
            utils.replace_inf(left)
            utils.replace_inf(right)

            npt.assert_almost_equal(left, right)


@pytest.mark.parametrize("T_A, T_B", test_data)
def test_stomp_A_B_join(T_A, T_B):
    m = 3
    left = np.array(
        [utils.naive_mass(Q, T_A, m) for Q in core.rolling_window(T_B, m)], dtype=object
    )
    right = stomp(T_A, m, T_B, ignore_trivial=False)
    utils.replace_inf(left)
    utils.replace_inf(right)
    npt.assert_almost_equal(left, right)


def test_stomp_nan_self_join_beginning():
    m = 3
    T = np.array([np.nan, 1, 0, 0, 1, 0, 0])

    zone = int(np.ceil(m / 4))
    left = np.array(
        [
            utils.naive_mass(Q, T, m, i, zone, ignore_trivial=True)
            for i, Q in enumerate(core.rolling_window(T, m))
        ],
        dtype=object,
    )

    right = stomp(T, m, ignore_trivial=True)

    utils.replace_inf(left)
    utils.replace_inf(right)
    npt.assert_almost_equal(left[:, 0], right[:, 0])


def test_stomp_inf_self_join_beginning():
    m = 3
    T = np.array([np.inf, 1, 0, 0, 1, 0, 0])

    zone = int(np.ceil(m / 4))
    left = np.array(
        [
            utils.naive_mass(Q, T, m, i, zone, ignore_trivial=True)
            for i, Q in enumerate(core.rolling_window(T, m))
        ],
        dtype=object,
    )

    right = stomp(T, m, ignore_trivial=True)

    utils.replace_inf(left)
    utils.replace_inf(right)
    npt.assert_almost_equal(left[:, 0], right[:, 0])


@pytest.mark.parametrize("T_A, T_B", test_data)
def test_stomp_nan_inf_self_join(T_A, T_B):
    m = 3
    nan_inf_sample_B = np.random.permutation(np.arange(len(T_B)))
    T_B[nan_inf_sample_B[: len(T_B) // 20]] = np.nan
    T_B[nan_inf_sample_B[-len(T_B) // 20 :]] = np.inf

    zone = int(np.ceil(m / 4))
    left = np.array(
        [
            utils.naive_mass(Q, T_B, m, i, zone, ignore_trivial=True)
            for i, Q in enumerate(core.rolling_window(T_B, m))
        ],
        dtype=object,
    )

    right = stomp(T_B, m, ignore_trivial=True)

    utils.replace_inf(left)
    utils.replace_inf(right)
    npt.assert_almost_equal(left, right)


@pytest.mark.parametrize("T_A, T_B", test_data)
def test_stomp_nan_inf_A_B_join(T_A, T_B):
    m = 3
    nan_inf_sample_A = np.random.permutation(np.arange(len(T_A)))
    nan_inf_sample_B = np.random.permutation(np.arange(len(T_B)))
    T_A[nan_inf_sample_A[: len(T_A) // 20]] = np.nan
    T_A[nan_inf_sample_A[-len(T_A) // 20 :]] = np.inf
    T_B[nan_inf_sample_B[: len(T_B) // 20]] = np.nan
    T_B[nan_inf_sample_B[-len(T_B) // 20 :]] = np.inf

    left = np.array(
        [utils.naive_mass(Q, T_A, m) for Q in core.rolling_window(T_B, m)], dtype=object
    )

    right = stomp(T_A, m, T_B, ignore_trivial=False)

    utils.replace_inf(left)
    utils.replace_inf(right)
    npt.assert_almost_equal(left, right)
