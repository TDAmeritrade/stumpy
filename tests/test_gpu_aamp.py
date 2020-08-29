import numpy as np
import numpy.testing as npt
import pandas as pd
from stumpy import gpu_aamp
from stumpy import core, config
from numba import cuda
import math
import pytest
import naive

config.THREADS_PER_BLOCK = 10

if not cuda.is_available():
    pytest.skip("Skipping Tests No GPUs Available", allow_module_level=True)


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


@pytest.mark.parametrize("T_A, T_B", test_data)
def test_gpu_aamp_self_join(T_A, T_B):
    m = 3
    zone = int(np.ceil(m / 4))
    left = naive.aamp(T_B, m, exclusion_zone=zone)
    right = gpu_aamp(T_B, m, ignore_trivial=True)
    naive.replace_inf(left)
    naive.replace_inf(right)
    npt.assert_almost_equal(left, right)

    right = gpu_aamp(pd.Series(T_B), m, ignore_trivial=True)
    naive.replace_inf(right)
    npt.assert_almost_equal(left, right)


@pytest.mark.parametrize("T_A, T_B", test_data)
@pytest.mark.parametrize("m", window_size)
def test_gpu_aamp_self_join_larger_window(T_A, T_B, m):
    if len(T_B) > m:
        zone = int(np.ceil(m / 4))
        left = naive.aamp(T_B, m, exclusion_zone=zone)
        right = gpu_aamp(T_B, m, ignore_trivial=True)
        naive.replace_inf(left)
        naive.replace_inf(right)

        npt.assert_almost_equal(left, right)

        # right = gpu_aamp(
        #     pd.Series(T_B),
        #     m,
        #     ignore_trivial=True,
        # )
        # naive.replace_inf(right)
        # npt.assert_almost_equal(left, right)


@pytest.mark.parametrize("T_A, T_B", test_data)
def test_gpu_aamp_A_B_join(T_A, T_B):
    m = 3
    left = naive.aamp(T_A, m, T_B=T_B)
    right = gpu_aamp(T_A, m, T_B, ignore_trivial=False)
    naive.replace_inf(left)
    naive.replace_inf(right)
    npt.assert_almost_equal(left, right)

    right = gpu_aamp(pd.Series(T_A), m, pd.Series(T_B), ignore_trivial=False)
    naive.replace_inf(right)
    npt.assert_almost_equal(left, right)


@pytest.mark.parametrize("T_A, T_B", test_data)
def test_parallel_gpu_aamp_self_join(T_A, T_B):
    device_ids = [device.id for device in cuda.list_devices()]
    if len(T_B) > 10:
        m = 3
        zone = int(np.ceil(m / 4))
        left = naive.aamp(T_B, m, exclusion_zone=zone)
        right = gpu_aamp(
            T_B,
            m,
            ignore_trivial=True,
            device_id=device_ids,
        )
        naive.replace_inf(left)
        naive.replace_inf(right)
        npt.assert_almost_equal(left, right)

        # right = gpu_aamp(
        #     pd.Series(T_B),
        #     m,
        #     ignore_trivial=True,
        #     device_id=device_ids,
        # )
        # naive.replace_inf(right)
        # npt.assert_almost_equal(left, right)


@pytest.mark.parametrize("T_A, T_B", test_data)
def test_parallel_gpu_aamp_A_B_join(T_A, T_B):
    device_ids = [device.id for device in cuda.list_devices()]
    if len(T_B) > 10:
        m = 3
        left = naive.aamp(T_A, m, T_B=T_B)
        right = gpu_aamp(
            T_A,
            m,
            T_B,
            ignore_trivial=False,
            device_id=device_ids,
        )
        naive.replace_inf(left)
        naive.replace_inf(right)
        npt.assert_almost_equal(left, right)

        # right = gpu_aamp(
        #     pd.Series(T_A),
        #     m,
        #     pd.Series(T_B),
        #     ignore_trivial=False,
        #     device_id=device_ids,
        # )
        # naive.replace_inf(right)
        # npt.assert_almost_equal(left, right)


def test_gpu_aamp_constant_subsequence_self_join():
    T_A = np.concatenate((np.zeros(20, dtype=np.float64), np.ones(5, dtype=np.float64)))
    m = 3
    zone = int(np.ceil(m / 4))
    left = naive.aamp(T_A, m, exclusion_zone=zone)
    right = gpu_aamp(T_A, m, ignore_trivial=True)
    naive.replace_inf(left)
    naive.replace_inf(right)
    npt.assert_almost_equal(left[:, 0], right[:, 0])  # ignore indices

    # right = gpu_aamp(pd.Series(T_A), m, ignore_trivial=True)
    # naive.replace_inf(right)
    # npt.assert_almost_equal(left[:, 0], right[:, 0])  # ignore indices


def test_gpu_aamp_one_constant_subsequence_A_B_join():
    T_A = np.random.rand(20)
    T_B = np.concatenate((np.zeros(20, dtype=np.float64), np.ones(5, dtype=np.float64)))
    m = 3
    left = naive.aamp(T_A, m, T_B=T_B)
    right = gpu_aamp(T_A, m, T_B, ignore_trivial=False)
    naive.replace_inf(left)
    naive.replace_inf(right)
    npt.assert_almost_equal(left[:, 0], right[:, 0])  # ignore indices

    # right = gpu_aamp(pd.Series(T_A), m, pd.Series(T_B), ignore_trivial=False)
    # naive.replace_inf(right)
    # npt.assert_almost_equal(left[:, 0], right[:, 0])  # ignore indices

    # Swap inputs
    left = naive.aamp(T_B, m, T_B=T_A)
    right = gpu_aamp(T_B, m, T_A, ignore_trivial=False)
    naive.replace_inf(left)
    naive.replace_inf(right)
    npt.assert_almost_equal(left[:, 0], right[:, 0])  # ignore indices

    # right = gpu_aamp(pd.Series(T_B), m, pd.Series(T_A), ignore_trivial=False)
    # naive.replace_inf(right)
    # npt.assert_almost_equal(left[:, 0], right[:, 0])  # ignore indices


def test_gpu_aamp_two_constant_subsequences_A_B_join():
    T_A = np.array([0, 0, 0, 0, 0, 1], dtype=np.float64)
    T_B = np.concatenate((np.zeros(20, dtype=np.float64), np.ones(5, dtype=np.float64)))
    m = 3
    left = naive.aamp(T_A, m, T_B=T_B)
    right = gpu_aamp(T_A, m, T_B, ignore_trivial=False)
    naive.replace_inf(left)
    naive.replace_inf(right)
    npt.assert_almost_equal(left[:, 0], right[:, 0])  # ignore indices

    # right = gpu_aamp(pd.Series(T_A), m, pd.Series(T_B), ignore_trivial=False)
    # naive.replace_inf(right)
    # npt.assert_almost_equal(left[:, 0], right[:, 0])  # ignore indices

    # Swap inputs
    left = naive.aamp(T_B, m, T_B=T_A)
    right = gpu_aamp(T_B, m, T_A, ignore_trivial=False)
    naive.replace_inf(left)
    naive.replace_inf(right)
    npt.assert_almost_equal(left[:, 0], right[:, 0])  # ignore indices

    # right = gpu_aamp(pd.Series(T_B), m, pd.Series(T_A), ignore_trivial=False)
    # naive.replace_inf(right)
    # npt.assert_almost_equal(left[:, 0], right[:, 0])  # ignore indices


def test_gpu_aamp_identical_subsequence_self_join():
    identical = np.random.rand(8)
    T_A = np.random.rand(20)
    T_A[1 : 1 + identical.shape[0]] = identical
    T_A[11 : 11 + identical.shape[0]] = identical
    m = 3
    zone = int(np.ceil(m / 4))
    left = naive.aamp(T_A, m, exclusion_zone=zone)
    right = gpu_aamp(T_A, m, ignore_trivial=True)
    naive.replace_inf(left)
    naive.replace_inf(right)
    npt.assert_almost_equal(
        left[:, 0], right[:, 0], decimal=config.STUMPY_TEST_PRECISION
    )  # ignore indices

    # right = gpu_aamp(pd.Series(T_A), m, ignore_trivial=True)
    # naive.replace_inf(right)
    # npt.assert_almost_equal(
    #     left[:, 0], right[:, 0], decimal=config.STUMPY_TEST_PRECISION
    # )  # ignore indices


def test_gpu_aamp_identical_subsequence_A_B_join():
    identical = np.random.rand(8)
    T_A = np.random.rand(20)
    T_B = np.random.rand(20)
    T_A[1 : 1 + identical.shape[0]] = identical
    T_B[11 : 11 + identical.shape[0]] = identical
    m = 3
    left = naive.aamp(T_A, m, T_B=T_B)
    right = gpu_aamp(T_A, m, T_B, ignore_trivial=False)
    naive.replace_inf(left)
    naive.replace_inf(right)
    npt.assert_almost_equal(
        left[:, 0], right[:, 0], decimal=config.STUMPY_TEST_PRECISION
    )  # ignore indices

    # right = gpu_aamp(pd.Series(T_A), m, pd.Series(T_B), ignore_trivial=False)
    # naive.replace_inf(right)
    # npt.assert_almost_equal(
    #     left[:, 0], right[:, 0], decimal=config.STUMPY_TEST_PRECISION
    # )  # ignore indices

    # Swap inputs
    left = naive.aamp(T_B, m, T_B=T_A)
    right = gpu_aamp(T_B, m, T_A, ignore_trivial=False)
    naive.replace_inf(left)
    naive.replace_inf(right)
    npt.assert_almost_equal(
        left[:, 0], right[:, 0], decimal=config.STUMPY_TEST_PRECISION
    )  # ignore indices

    # right = gpu_aamp(pd.Series(T_B), m, pd.Series(T_A), ignore_trivial=False)
    # naive.replace_inf(right)
    # npt.assert_almost_equal(
    #     left[:, 0], right[:, 0], decimal=config.STUMPY_TEST_PRECISION
    # )  # ignore indices


@pytest.mark.parametrize("T_A, T_B", test_data)
@pytest.mark.parametrize("substitute_B", substitution_values)
@pytest.mark.parametrize("substitution_locations", substitution_locations)
def test_gpu_aamp_nan_inf_self_join(T_A, T_B, substitute_B, substitution_locations):
    m = 3
    stop = 16
    T_B_sub = T_B.copy()[:stop]

    for substitution_location_B in substitution_locations:
        T_B_sub[:] = T_B[:stop]
        T_B_sub[substitution_location_B] = substitute_B

        zone = int(np.ceil(m / 4))
        left = naive.aamp(T_B_sub, m, exclusion_zone=zone)
        right = gpu_aamp(T_B_sub, m, ignore_trivial=True)
        naive.replace_inf(left)
        naive.replace_inf(right)
        npt.assert_almost_equal(left, right)

        # right = gpu_aamp(pd.Series(T_B_sub), m, ignore_trivial=True)
        # naive.replace_inf(right)
        # npt.assert_almost_equal(left, right)


@pytest.mark.parametrize("T_A, T_B", test_data)
@pytest.mark.parametrize("substitute_A", substitution_values)
@pytest.mark.parametrize("substitute_B", substitution_values)
@pytest.mark.parametrize("substitution_locations", substitution_locations)
def test_gpu_aamp_nan_inf_A_B_join(
    T_A, T_B, substitute_A, substitute_B, substitution_locations
):
    m = 3
    stop = 16
    T_A_sub = T_A.copy()
    T_B_sub = T_B.copy()[:stop]

    for substitution_location_B in substitution_locations:
        for substitution_location_A in substitution_locations:
            T_A_sub[:] = T_A
            T_B_sub[:] = T_B[:stop]
            T_A_sub[substitution_location_A] = substitute_A
            T_B_sub[substitution_location_B] = substitute_B

            left = naive.aamp(T_A_sub, m, T_B=T_B_sub)
            right = gpu_aamp(T_A_sub, m, T_B_sub, ignore_trivial=False)
            naive.replace_inf(left)
            naive.replace_inf(right)
            npt.assert_almost_equal(left, right)

            # right = gpu_aamp(
            #     pd.Series(T_A_sub), m, pd.Series(T_B_sub), ignore_trivial=False
            # )
            # naive.replace_inf(right)
            # npt.assert_almost_equal(left, right)


def test_gpu_aamp_nan_zero_mean_self_join():
    T = np.array([-1, 0, 1, np.inf, 1, 0, -1])
    m = 3

    zone = int(np.ceil(m / 4))
    left = naive.aamp(T, m, exclusion_zone=zone)
    right = gpu_aamp(T, m, ignore_trivial=True)

    naive.replace_inf(left)
    naive.replace_inf(right)
    npt.assert_almost_equal(left, right)
