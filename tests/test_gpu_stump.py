import numpy as np
import numpy.testing as npt
import pandas as pd
from stumpy import (
    gpu_stump,
    _get_QT_kernel,
    _ignore_trivial_kernel,
    _calculate_squared_distance_kernel,
    _update_PI_kernel,
)
from stumpy import core, _get_QT
from numba import cuda
import math
import pytest

THREADS_PER_BLOCK = 1

if not cuda.is_available():
    pytest.skip("Skipping Tests No GPUs Available", allow_module_level=True)


def naive_mass(Q, T, m, trivial_idx=None, excl_zone=0, ignore_trivial=False):
    D = np.linalg.norm(
        core.z_norm(core.rolling_window(T, m), 1) - core.z_norm(Q), axis=1
    )
    if ignore_trivial:
        start = max(0, trivial_idx - excl_zone)
        stop = min(T.shape[0] - Q.shape[0] + 1, trivial_idx + excl_zone)
        D[start:stop] = np.inf
    I = np.argmin(D)
    P = D[I]

    # Get left and right matrix profiles for self-joins
    if ignore_trivial and trivial_idx > 0:
        PL = np.inf
        IL = -1
        for i in range(trivial_idx):
            if D[i] < PL:
                IL = i
                PL = D[i]
        if start <= IL < stop:
            IL = -1
    else:
        IL = -1

    if ignore_trivial and trivial_idx + 1 < D.shape[0]:
        PR = np.inf
        IR = -1
        for i in range(trivial_idx + 1, D.shape[0]):
            if D[i] < PR:
                IR = i
                PR = D[i]
        if start <= IR < stop:
            IR = -1
    else:
        IR = -1

    return P, I, IL, IR


def replace_inf(x, value=0):
    x[x == np.inf] = value
    x[x == -np.inf] = value
    return


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
def test_get_QT_kernel(T_A, T_B):
    m = 3
    M_T, Σ_T = core.compute_mean_std(T_B, m)
    μ_Q, σ_Q = core.compute_mean_std(T_A, m)
    QT, QT_first = _get_QT(0, T_B, T_A, m)

    device_T_A = cuda.to_device(T_B)
    device_T_B = cuda.to_device(T_A)
    device_M_T = cuda.to_device(M_T)
    device_Σ_T = cuda.to_device(Σ_T)
    device_QT_odd = cuda.to_device(QT)
    device_QT_even = cuda.to_device(QT)
    device_QT_first = cuda.to_device(QT_first)

    threads_per_block = THREADS_PER_BLOCK
    blocks_per_grid = math.ceil(QT_first.shape[0] / threads_per_block)

    for i in range(1, QT_first.shape[0]):
        left = core.sliding_dot_product(T_A[i : i + m], T_B)

        _get_QT_kernel[blocks_per_grid, threads_per_block](
            i, device_T_A, device_T_B, m, device_QT_even, device_QT_odd, device_QT_first
        )

        if i % 2 == 0:
            right = device_QT_even.copy_to_host()
            npt.assert_almost_equal(left, right)
        else:
            right = device_QT_odd.copy_to_host()
            npt.assert_almost_equal(left, right)


@pytest.mark.parametrize("T_A, T_B", test_data)
def test_calculate_squared_distance_kernel(T_A, T_B):
    m = 3
    for i in range(T_A.shape[0] - m + 1):
        Q = T_A[i:i+m]
        left = np.linalg.norm(
            core.z_norm(core.rolling_window(T_B, m), 1) - core.z_norm(Q), axis=1
        )
        left = np.square(left)
        M_T, Σ_T = core.compute_mean_std(T_B, m)
        QT = core.sliding_dot_product(Q, T_B)
        μ_Q, σ_Q = core.compute_mean_std(T_A, m)

        device_M_T = cuda.to_device(M_T)
        device_Σ_T = cuda.to_device(Σ_T)
        device_QT_even = cuda.to_device(QT)
        device_QT_odd = cuda.to_device(QT)
        device_QT_first = cuda.to_device(QT)
        device_μ_Q = cuda.to_device(μ_Q)
        device_σ_Q = cuda.to_device(σ_Q)
        device_D = cuda.device_array(QT.shape, dtype=np.float64)
        device_denom = cuda.device_array(QT.shape, dtype=np.float64)

        threads_per_block = THREADS_PER_BLOCK
        blocks_per_grid = math.ceil(QT.shape[0] / threads_per_block)

        _calculate_squared_distance_kernel[blocks_per_grid, threads_per_block](
            i,
            m,
            device_M_T,
            device_Σ_T,
            device_QT_even,
            device_QT_odd,
            device_μ_Q,
            device_σ_Q,
            device_D,
            device_denom,
        )

        right = device_D.copy_to_host()
        npt.assert_almost_equal(left, right)


def test_ignore_trivial_kernel():
    D = np.random.rand(10)

    start_stop = [(0, 3), (4, 6), (7, 9)]

    for start, stop in start_stop:
        left = D.copy()
        left[start:stop] = np.inf

        device_D = cuda.to_device(D)
        _ignore_trivial_kernel(device_D, start, stop)
        right = device_D.copy_to_host()

        npt.assert_almost_equal(left, right)


def test_update_PI_kernel():
    D = np.random.rand(5, 10)
    profile = np.empty((10, 3))
    profile[:, :] = np.inf
    indices = np.ones((10, 3)) * -1

    ignore_trivial = False

    left_profile = profile.copy()
    left_profile[:, 0] = np.min(D, axis=0)
    left_indices = np.ones((10, 3)) * -1
    left_indices[:, 0] = np.argmin(D, axis=0)

    device_profile = cuda.to_device(profile)
    device_indices = cuda.to_device(indices)

    for i in range(D.shape[0]):
        device_D = cuda.to_device(D[i])
        _update_PI_kernel(i, device_D, ignore_trivial, device_profile, device_indices)

    right_profile = device_profile.copy_to_host()
    right_indices = device_indices.copy_to_host()

    npt.assert_almost_equal(left_profile, right_profile)

    ignore_trivial = True

    left_profile = profile.copy()
    left_profile[:, 0] = np.min(D, axis=0)
    # for i in range(1, D.shape[1]):

    for j in range(D.shape[1]):
        for i in range(D.shape[0]):
            if i < j and D[i, j] < left_profile[j, 1]:
                left_profile[j, 1] = D[i, j]
            if i > j and D[i, j] < left_profile[j, 2]:
                left_profile[j, 2] = D[i, j]

    left_indices = np.ones((10, 3)) * -1
    left_indices[:, 0] = np.argmin(D, axis=0)

    device_profile = cuda.to_device(profile)
    device_indices = cuda.to_device(indices)

    for i in range(D.shape[0]):
        device_D = cuda.to_device(D[i])
        _update_PI_kernel(i, device_D, ignore_trivial, device_profile, device_indices)

    right_profile = device_profile.copy_to_host()
    right_indices = device_indices.copy_to_host()

    npt.assert_almost_equal(left_profile[:, 2], right_profile[:, 2])


@pytest.mark.parametrize("T_A, T_B", test_data)
def test_stump_self_join(T_A, T_B):
    m = 3
    zone = int(np.ceil(m / 4))
    left = np.array(
        [
            naive_mass(Q, T_B, m, i, zone, True)
            for i, Q in enumerate(core.rolling_window(T_B, m))
        ],
        dtype=object,
    )
    right = gpu_stump(T_B, m, ignore_trivial=True, threads_per_block=THREADS_PER_BLOCK)
    replace_inf(left)
    replace_inf(right)
    npt.assert_almost_equal(left, right)

    right = gpu_stump(
        pd.Series(T_B), m, ignore_trivial=True, threads_per_block=THREADS_PER_BLOCK
    )
    replace_inf(right)
    npt.assert_almost_equal(left, right)


@pytest.mark.parametrize("T_A, T_B", test_data)
def test_stump_A_B_join(T_A, T_B):
    m = 3
    left = np.array(
        [naive_mass(Q, T_A, m) for Q in core.rolling_window(T_B, m)], dtype=object
    )
    right = gpu_stump(
        T_A, m, T_B, ignore_trivial=False, threads_per_block=THREADS_PER_BLOCK
    )
    replace_inf(left)
    replace_inf(right)
    npt.assert_almost_equal(left, right)

    right = gpu_stump(
        pd.Series(T_A),
        m,
        pd.Series(T_B),
        ignore_trivial=False,
        threads_per_block=THREADS_PER_BLOCK,
    )
    replace_inf(right)
    npt.assert_almost_equal(left, right)
