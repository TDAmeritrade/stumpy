import numpy as np
import numpy.testing as npt
import pandas as pd
from stumpy import gpu_stump
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

    if P == np.inf:
        I = -1

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
def test_gpu_stump_self_join(T_A, T_B):
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
def test_gpu_stump_self_join_larger_window(T_A, T_B):
    for m in [8, 16, 32]:
        if len(T_B) > m:
            zone = int(np.ceil(m / 4))
            left = np.array(
                [
                    naive_mass(Q, T_B, m, i, zone, True)
                    for i, Q in enumerate(core.rolling_window(T_B, m))
                ],
                dtype=object,
            )
            right = gpu_stump(
                T_B, m, ignore_trivial=True, threads_per_block=THREADS_PER_BLOCK
            )
            replace_inf(left)
            replace_inf(right)

            npt.assert_almost_equal(left, right)

            right = gpu_stump(
                pd.Series(T_B),
                m,
                ignore_trivial=True,
                threads_per_block=THREADS_PER_BLOCK,
            )
            replace_inf(right)
            npt.assert_almost_equal(left, right)


@pytest.mark.parametrize("T_A, T_B", test_data)
def test_gpu_stump_A_B_join(T_A, T_B):
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


@pytest.mark.parametrize("T_A, T_B", test_data)
def test_parallel_gpu_stump_self_join(T_A, T_B):
    device_ids = [device.id for device in cuda.list_devices()]
    if len(T_B) > 10:
        m = 3
        zone = int(np.ceil(m / 4))
        left = np.array(
            [
                naive_mass(Q, T_B, m, i, zone, True)
                for i, Q in enumerate(core.rolling_window(T_B, m))
            ],
            dtype=object,
        )
        right = gpu_stump(
            T_B,
            m,
            ignore_trivial=True,
            threads_per_block=THREADS_PER_BLOCK,
            device_id=device_ids,
        )
        replace_inf(left)
        replace_inf(right)
        npt.assert_almost_equal(left, right)

        right = gpu_stump(
            pd.Series(T_B),
            m,
            ignore_trivial=True,
            threads_per_block=THREADS_PER_BLOCK,
            device_id=device_ids,
        )
        replace_inf(right)
        npt.assert_almost_equal(left, right)


@pytest.mark.parametrize("T_A, T_B", test_data)
def test_parallel_gpu_stump_A_B_join(T_A, T_B):
    device_ids = [device.id for device in cuda.list_devices()]
    if len(T_B) > 10:
        m = 3
        left = np.array(
            [naive_mass(Q, T_A, m) for Q in core.rolling_window(T_B, m)], dtype=object
        )
        right = gpu_stump(
            T_A,
            m,
            T_B,
            ignore_trivial=False,
            threads_per_block=THREADS_PER_BLOCK,
            device_id=device_ids,
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
            device_id=device_ids,
        )
        replace_inf(right)
        npt.assert_almost_equal(left, right)
