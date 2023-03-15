from unittest.mock import patch

import numpy as np
import numpy.testing as npt
import pandas as pd
from numba import cuda

from stumpy import config, gpu_aamp

try:
    from numba.errors import NumbaPerformanceWarning
except ModuleNotFoundError:
    from numba.core.errors import NumbaPerformanceWarning

import naive
import pytest

TEST_THREADS_PER_BLOCK = 10

if not cuda.is_available():  # pragma: no cover
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


@patch("stumpy.config.STUMPY_THREADS_PER_BLOCK", TEST_THREADS_PER_BLOCK)
def test_gpu_aamp_int_input():
    with pytest.raises(TypeError):
        gpu_aamp(np.arange(10), 5, ignore_trivial=True)


@pytest.mark.filterwarnings("ignore", category=NumbaPerformanceWarning)
@pytest.mark.parametrize("T_A, T_B", test_data)
@patch("stumpy.config.STUMPY_THREADS_PER_BLOCK", TEST_THREADS_PER_BLOCK)
def test_gpu_aamp_self_join(T_A, T_B):
    m = 3
    zone = int(np.ceil(m / 4))
    for p in [1.0, 2.0, 3.0]:
        ref_mp = naive.aamp(T_B, m, exclusion_zone=zone, p=p, row_wise=True)
        comp_mp = gpu_aamp(T_B, m, ignore_trivial=True, p=p)
        naive.replace_inf(ref_mp)
        naive.replace_inf(comp_mp)
        npt.assert_almost_equal(ref_mp, comp_mp)

        comp_mp = gpu_aamp(pd.Series(T_B), m, ignore_trivial=True, p=p)
        naive.replace_inf(comp_mp)
        npt.assert_almost_equal(ref_mp, comp_mp)


@pytest.mark.filterwarnings("ignore", category=NumbaPerformanceWarning)
@pytest.mark.parametrize("T_A, T_B", test_data)
@pytest.mark.parametrize("m", window_size)
@patch("stumpy.config.STUMPY_THREADS_PER_BLOCK", TEST_THREADS_PER_BLOCK)
def test_gpu_aamp_self_join_larger_window(T_A, T_B, m):
    if len(T_B) > m:
        zone = int(np.ceil(m / 4))
        ref_mp = naive.aamp(T_B, m, exclusion_zone=zone)
        comp_mp = gpu_aamp(T_B, m, ignore_trivial=True)
        naive.replace_inf(ref_mp)
        naive.replace_inf(comp_mp)

        npt.assert_almost_equal(ref_mp, comp_mp)

        # comp_mp = gpu_aamp(
        #     pd.Series(T_B),
        #     m,
        #     ignore_trivial=True,
        # )
        # naive.replace_inf(comp_mp)
        # npt.assert_almost_equal(ref_mp, comp_mp)


@pytest.mark.filterwarnings("ignore", category=NumbaPerformanceWarning)
@pytest.mark.parametrize("T_A, T_B", test_data)
@patch("stumpy.config.STUMPY_THREADS_PER_BLOCK", TEST_THREADS_PER_BLOCK)
def test_gpu_aamp_A_B_join(T_A, T_B):
    m = 3
    for p in [1.0, 2.0, 3.0]:
        ref_mp = naive.aamp(T_B, m, T_B=T_A, p=p, row_wise=True)
        comp_mp = gpu_aamp(T_B, m, T_A, ignore_trivial=False, p=p)
        naive.replace_inf(ref_mp)
        naive.replace_inf(comp_mp)
        npt.assert_almost_equal(ref_mp, comp_mp)

        # comp_mp = gpu_aamp(
        #     pd.Series(T_B), m, pd.Series(T_A), ignore_trivial=False, p=p
        # )
        # naive.replace_inf(comp_mp)
        # npt.assert_almost_equal(ref_mp, comp_mp)


@pytest.mark.filterwarnings("ignore", category=NumbaPerformanceWarning)
@pytest.mark.parametrize("T_A, T_B", test_data)
@patch("stumpy.config.STUMPY_THREADS_PER_BLOCK", TEST_THREADS_PER_BLOCK)
def test_parallel_gpu_aamp_self_join(T_A, T_B):
    device_ids = [device.id for device in cuda.list_devices()]
    if len(T_B) > 10:
        m = 3
        zone = int(np.ceil(m / 4))
        ref_mp = naive.aamp(T_B, m, exclusion_zone=zone)
        comp_mp = gpu_aamp(
            T_B,
            m,
            ignore_trivial=True,
            device_id=device_ids,
        )
        naive.replace_inf(ref_mp)
        naive.replace_inf(comp_mp)
        npt.assert_almost_equal(ref_mp, comp_mp)

        # comp_mp = gpu_aamp(
        #     pd.Series(T_B),
        #     m,
        #     ignore_trivial=True,
        #     device_id=device_ids,
        # )
        # naive.replace_inf(comp_mp)
        # npt.assert_almost_equal(ref_mp, comp_mp)


@pytest.mark.filterwarnings("ignore", category=NumbaPerformanceWarning)
@pytest.mark.parametrize("T_A, T_B", test_data)
@patch("stumpy.config.STUMPY_THREADS_PER_BLOCK", TEST_THREADS_PER_BLOCK)
def test_parallel_gpu_aamp_A_B_join(T_A, T_B):
    device_ids = [device.id for device in cuda.list_devices()]
    if len(T_B) > 10:
        m = 3
        ref_mp = naive.aamp(T_B, m, T_B=T_A)
        comp_mp = gpu_aamp(
            T_B,
            m,
            T_A,
            ignore_trivial=False,
            device_id=device_ids,
        )
        naive.replace_inf(ref_mp)
        naive.replace_inf(comp_mp)
        npt.assert_almost_equal(ref_mp, comp_mp)

        # comp_mp = gpu_aamp(
        #     pd.Series(T_B),
        #     m,
        #     pd.Series(T_A),
        #     ignore_trivial=False,
        #     device_id=device_ids,
        # )
        # naive.replace_inf(comp_mp)
        # npt.assert_almost_equal(ref_mp, comp_mp)


@pytest.mark.filterwarnings("ignore", category=NumbaPerformanceWarning)
@patch("stumpy.config.STUMPY_THREADS_PER_BLOCK", TEST_THREADS_PER_BLOCK)
def test_gpu_aamp_constant_subsequence_self_join():
    T_A = np.concatenate((np.zeros(20, dtype=np.float64), np.ones(5, dtype=np.float64)))
    m = 3
    zone = int(np.ceil(m / 4))
    ref_mp = naive.aamp(T_A, m, exclusion_zone=zone)
    comp_mp = gpu_aamp(T_A, m, ignore_trivial=True)
    naive.replace_inf(ref_mp)
    naive.replace_inf(comp_mp)
    npt.assert_almost_equal(ref_mp[:, 0], comp_mp[:, 0])  # ignore indices

    # comp_mp = gpu_aamp(pd.Series(T_A), m, ignore_trivial=True)
    # naive.replace_inf(comp_mp)
    # npt.assert_almost_equal(ref_mp[:, 0], comp_mp[:, 0])  # ignore indices


@pytest.mark.filterwarnings("ignore", category=NumbaPerformanceWarning)
@patch("stumpy.config.STUMPY_THREADS_PER_BLOCK", TEST_THREADS_PER_BLOCK)
def test_gpu_aamp_one_constant_subsequence_A_B_join():
    T_A = np.random.rand(20)
    T_B = np.concatenate((np.zeros(20, dtype=np.float64), np.ones(5, dtype=np.float64)))
    m = 3
    ref_mp = naive.aamp(T_B, m, T_B=T_A)
    comp_mp = gpu_aamp(T_B, m, T_A, ignore_trivial=False)
    naive.replace_inf(ref_mp)
    naive.replace_inf(comp_mp)
    npt.assert_almost_equal(ref_mp[:, 0], comp_mp[:, 0])  # ignore indices

    # comp_mp = gpu_aamp(pd.Series(T_B), m, pd.Series(T_A), ignore_trivial=False)
    # naive.replace_inf(comp_mp)
    # npt.assert_almost_equal(ref_mp[:, 0], comp_mp[:, 0])  # ignore indices

    # Swap inputs
    ref_mp = naive.aamp(T_A, m, T_B=T_B)
    comp_mp = gpu_aamp(T_A, m, T_B, ignore_trivial=False)
    naive.replace_inf(ref_mp)
    naive.replace_inf(comp_mp)
    npt.assert_almost_equal(ref_mp[:, 0], comp_mp[:, 0])  # ignore indices

    # comp_mp = gpu_aamp(pd.Series(T_A), m, pd.Series(T_B), ignore_trivial=False)
    # naive.replace_inf(comp_mp)
    # npt.assert_almost_equal(ref_mp[:, 0], comp_mp[:, 0])  # ignore indices


@pytest.mark.filterwarnings("ignore", category=NumbaPerformanceWarning)
@patch("stumpy.config.STUMPY_THREADS_PER_BLOCK", TEST_THREADS_PER_BLOCK)
def test_gpu_aamp_two_constant_subsequences_A_B_join():
    T_A = np.array([0, 0, 0, 0, 0, 1], dtype=np.float64)
    T_B = np.concatenate((np.zeros(20, dtype=np.float64), np.ones(5, dtype=np.float64)))
    m = 3
    ref_mp = naive.aamp(T_B, m, T_B=T_A)
    comp_mp = gpu_aamp(T_B, m, T_A, ignore_trivial=False)
    naive.replace_inf(ref_mp)
    naive.replace_inf(comp_mp)
    npt.assert_almost_equal(ref_mp[:, 0], comp_mp[:, 0])  # ignore indices

    # comp_mp = gpu_aamp(pd.Series(T_B), m, pd.Series(T_A), ignore_trivial=False)
    # naive.replace_inf(comp_mp)
    # npt.assert_almost_equal(ref_mp[:, 0], comp_mp[:, 0])  # ignore indices

    # Swap inputs
    ref_mp = naive.aamp(T_A, m, T_B=T_B)
    comp_mp = gpu_aamp(T_A, m, T_B, ignore_trivial=False)
    naive.replace_inf(ref_mp)
    naive.replace_inf(comp_mp)
    npt.assert_almost_equal(ref_mp[:, 0], comp_mp[:, 0])  # ignore indices

    # comp_mp = gpu_aamp(pd.Series(T_A), m, pd.Series(T_B), ignore_trivial=False)
    # naive.replace_inf(comp_mp)
    # npt.assert_almost_equal(ref_mp[:, 0], comp_mp[:, 0])  # ignore indices


@pytest.mark.filterwarnings("ignore", category=NumbaPerformanceWarning)
@patch("stumpy.config.STUMPY_THREADS_PER_BLOCK", TEST_THREADS_PER_BLOCK)
def test_gpu_aamp_identical_subsequence_self_join():
    identical = np.random.rand(8)
    T_A = np.random.rand(20)
    T_A[1 : 1 + identical.shape[0]] = identical
    T_A[11 : 11 + identical.shape[0]] = identical
    m = 3
    zone = int(np.ceil(m / 4))
    ref_mp = naive.aamp(T_A, m, exclusion_zone=zone)
    comp_mp = gpu_aamp(T_A, m, ignore_trivial=True)
    naive.replace_inf(ref_mp)
    naive.replace_inf(comp_mp)
    npt.assert_almost_equal(
        ref_mp[:, 0], comp_mp[:, 0], decimal=config.STUMPY_TEST_PRECISION
    )  # ignore indices

    # comp_mp = gpu_aamp(pd.Series(T_A), m, ignore_trivial=True)
    # naive.replace_inf(comp_mp)
    # npt.assert_almost_equal(
    #     ref_mp[:, 0], comp_mp[:, 0], decimal=config.STUMPY_TEST_PRECISION
    # )  # ignore indices


@pytest.mark.filterwarnings("ignore", category=NumbaPerformanceWarning)
@patch("stumpy.config.STUMPY_THREADS_PER_BLOCK", TEST_THREADS_PER_BLOCK)
def test_gpu_aamp_identical_subsequence_A_B_join():
    identical = np.random.rand(8)
    T_A = np.random.rand(20)
    T_B = np.random.rand(20)
    T_A[1 : 1 + identical.shape[0]] = identical
    T_B[11 : 11 + identical.shape[0]] = identical
    m = 3
    ref_mp = naive.aamp(T_A, m, T_B=T_B)
    comp_mp = gpu_aamp(T_A, m, T_B, ignore_trivial=False)
    naive.replace_inf(ref_mp)
    naive.replace_inf(comp_mp)
    npt.assert_almost_equal(
        ref_mp[:, 0], comp_mp[:, 0], decimal=config.STUMPY_TEST_PRECISION
    )  # ignore indices

    # comp_mp = gpu_aamp(pd.Series(T_A), m, pd.Series(T_B), ignore_trivial=False)
    # naive.replace_inf(comp_mp)
    # npt.assert_almost_equal(
    #     ref_mp[:, 0], comp_mp[:, 0], decimal=config.STUMPY_TEST_PRECISION
    # )  # ignore indices

    # Swap inputs
    ref_mp = naive.aamp(T_B, m, T_B=T_A)
    comp_mp = gpu_aamp(T_B, m, T_A, ignore_trivial=False)
    naive.replace_inf(ref_mp)
    naive.replace_inf(comp_mp)
    npt.assert_almost_equal(
        ref_mp[:, 0], comp_mp[:, 0], decimal=config.STUMPY_TEST_PRECISION
    )  # ignore indices

    # comp_mp = gpu_aamp(pd.Series(T_B), m, pd.Series(T_A), ignore_trivial=False)
    # naive.replace_inf(comp_mp)
    # npt.assert_almost_equal(
    #     ref_mp[:, 0], comp_mp[:, 0], decimal=config.STUMPY_TEST_PRECISION
    # )  # ignore indices


@pytest.mark.filterwarnings("ignore", category=NumbaPerformanceWarning)
@pytest.mark.parametrize("T_A, T_B", test_data)
@pytest.mark.parametrize("substitute_B", substitution_values)
@pytest.mark.parametrize("substitution_locations", substitution_locations)
@patch("stumpy.config.STUMPY_THREADS_PER_BLOCK", TEST_THREADS_PER_BLOCK)
def test_gpu_aamp_nan_inf_self_join(T_A, T_B, substitute_B, substitution_locations):
    m = 3
    stop = 16
    T_B_sub = T_B.copy()[:stop]

    for substitution_location_B in substitution_locations:
        T_B_sub[:] = T_B[:stop]
        T_B_sub[substitution_location_B] = substitute_B

        zone = int(np.ceil(m / 4))
        ref_mp = naive.aamp(T_B_sub, m, exclusion_zone=zone)
        comp_mp = gpu_aamp(T_B_sub, m, ignore_trivial=True)
        naive.replace_inf(ref_mp)
        naive.replace_inf(comp_mp)
        npt.assert_almost_equal(ref_mp, comp_mp)

        # comp_mp = gpu_aamp(pd.Series(T_B_sub), m, ignore_trivial=True)
        # naive.replace_inf(comp_mp)
        # npt.assert_almost_equal(ref_mp, comp_mp)


@pytest.mark.filterwarnings("ignore", category=NumbaPerformanceWarning)
@pytest.mark.parametrize("T_A, T_B", test_data)
@pytest.mark.parametrize("substitute_A", substitution_values)
@pytest.mark.parametrize("substitute_B", substitution_values)
@pytest.mark.parametrize("substitution_locations", substitution_locations)
@patch("stumpy.config.STUMPY_THREADS_PER_BLOCK", TEST_THREADS_PER_BLOCK)
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

            ref_mp = naive.aamp(T_B_sub, m, T_B=T_A_sub)
            comp_mp = gpu_aamp(T_B_sub, m, T_A_sub, ignore_trivial=False)
            naive.replace_inf(ref_mp)
            naive.replace_inf(comp_mp)
            npt.assert_almost_equal(ref_mp, comp_mp)

            # comp_mp = gpu_aamp(
            #     pd.Series(T_B_sub), m, pd.Series(T_A_sub), ignore_trivial=False
            # )
            # naive.replace_inf(comp_mp)
            # npt.assert_almost_equal(ref_mp, comp_mp)


@pytest.mark.filterwarnings("ignore", category=NumbaPerformanceWarning)
@patch("stumpy.config.STUMPY_THREADS_PER_BLOCK", TEST_THREADS_PER_BLOCK)
def test_gpu_aamp_nan_zero_mean_self_join():
    T = np.array([-1, 0, 1, np.inf, 1, 0, -1])
    m = 3

    zone = int(np.ceil(m / 4))
    ref_mp = naive.aamp(T, m, exclusion_zone=zone)
    comp_mp = gpu_aamp(T, m, ignore_trivial=True)

    naive.replace_inf(ref_mp)
    naive.replace_inf(comp_mp)
    npt.assert_almost_equal(ref_mp, comp_mp)


@pytest.mark.filterwarnings("ignore", category=NumbaPerformanceWarning)
@pytest.mark.parametrize("T_A, T_B", test_data)
def test_gpu_aamp_self_join_KNN(T_A, T_B):
    m = 3
    zone = int(np.ceil(m / 4))
    for k in range(2, 4):
        for p in [1.0, 2.0, 3.0]:
            ref_mp = naive.aamp(T_B, m, exclusion_zone=zone, p=p, k=k)
            comp_mp = gpu_aamp(T_B, m, ignore_trivial=True, p=p, k=k)
            naive.replace_inf(ref_mp)
            naive.replace_inf(comp_mp)
            npt.assert_almost_equal(ref_mp, comp_mp)

            comp_mp = gpu_aamp(pd.Series(T_B), m, ignore_trivial=True, p=p, k=k)
            naive.replace_inf(comp_mp)
            npt.assert_almost_equal(ref_mp, comp_mp)


@pytest.mark.filterwarnings("ignore", category=NumbaPerformanceWarning)
@pytest.mark.parametrize("T_A, T_B", test_data)
def test_gpu_aamp_A_B_join_KNN(T_A, T_B):
    m = 3
    for k in range(2, 4):
        for p in [1.0, 2.0, 3.0]:
            ref_mp = naive.aamp(T_B, m, T_B=T_A, p=p, k=k)
            comp_mp = gpu_aamp(T_B, m, T_A, ignore_trivial=False, p=p, k=k)
            naive.replace_inf(ref_mp)
            naive.replace_inf(comp_mp)
            npt.assert_almost_equal(ref_mp, comp_mp)
