import numpy as np
import numpy.testing as npt
import pandas as pd
from stumpy import gpu_stump
from stumpy import core, _get_QT
from numba import cuda
import math
import pytest
import utils

THREADS_PER_BLOCK = 1

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


@pytest.mark.parametrize("T_A, T_B", test_data)
def test_gpu_stump_self_join(T_A, T_B):
    m = 3
    zone = int(np.ceil(m / 4))
    left = np.array(
        [
            utils.naive_mass(Q, T_B, m, i, zone, True)
            for i, Q in enumerate(core.rolling_window(T_B, m))
        ],
        dtype=object,
    )
    right = gpu_stump(T_B, m, ignore_trivial=True, threads_per_block=THREADS_PER_BLOCK)
    utils.replace_inf(left)
    utils.replace_inf(right)
    npt.assert_almost_equal(left, right)

    right = gpu_stump(
        pd.Series(T_B), m, ignore_trivial=True, threads_per_block=THREADS_PER_BLOCK
    )
    utils.replace_inf(right)
    npt.assert_almost_equal(left, right)


@pytest.mark.parametrize("T_A, T_B", test_data)
def test_gpu_stump_self_join_larger_window(T_A, T_B):
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
            right = gpu_stump(
                T_B, m, ignore_trivial=True, threads_per_block=THREADS_PER_BLOCK
            )
            utils.replace_inf(left)
            utils.replace_inf(right)

            npt.assert_almost_equal(left, right)

            right = gpu_stump(
                pd.Series(T_B),
                m,
                ignore_trivial=True,
                threads_per_block=THREADS_PER_BLOCK,
            )
            utils.replace_inf(right)
            npt.assert_almost_equal(left, right)


@pytest.mark.parametrize("T_A, T_B", test_data)
def test_gpu_stump_A_B_join(T_A, T_B):
    m = 3
    left = np.array(
        [utils.naive_mass(Q, T_A, m) for Q in core.rolling_window(T_B, m)], dtype=object
    )
    right = gpu_stump(
        T_A, m, T_B, ignore_trivial=False, threads_per_block=THREADS_PER_BLOCK
    )
    utils.replace_inf(left)
    utils.replace_inf(right)
    npt.assert_almost_equal(left, right)

    right = gpu_stump(
        pd.Series(T_A),
        m,
        pd.Series(T_B),
        ignore_trivial=False,
        threads_per_block=THREADS_PER_BLOCK,
    )
    utils.replace_inf(right)
    npt.assert_almost_equal(left, right)


@pytest.mark.parametrize("T_A, T_B", test_data)
def test_parallel_gpu_stump_self_join(T_A, T_B):
    device_ids = [device.id for device in cuda.list_devices()]
    if len(T_B) > 10:
        m = 3
        zone = int(np.ceil(m / 4))
        left = np.array(
            [
                utils.naive_mass(Q, T_B, m, i, zone, True)
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
        utils.replace_inf(left)
        utils.replace_inf(right)
        npt.assert_almost_equal(left, right)

        right = gpu_stump(
            pd.Series(T_B),
            m,
            ignore_trivial=True,
            threads_per_block=THREADS_PER_BLOCK,
            device_id=device_ids,
        )
        utils.replace_inf(right)
        npt.assert_almost_equal(left, right)


@pytest.mark.parametrize("T_A, T_B", test_data)
def test_parallel_gpu_stump_A_B_join(T_A, T_B):
    device_ids = [device.id for device in cuda.list_devices()]
    if len(T_B) > 10:
        m = 3
        left = np.array(
            [utils.naive_mass(Q, T_A, m) for Q in core.rolling_window(T_B, m)],
            dtype=object,
        )
        right = gpu_stump(
            T_A,
            m,
            T_B,
            ignore_trivial=False,
            threads_per_block=THREADS_PER_BLOCK,
            device_id=device_ids,
        )
        utils.replace_inf(left)
        utils.replace_inf(right)
        npt.assert_almost_equal(left, right)

        right = gpu_stump(
            pd.Series(T_A),
            m,
            pd.Series(T_B),
            ignore_trivial=False,
            threads_per_block=THREADS_PER_BLOCK,
            device_id=device_ids,
        )
        utils.replace_inf(right)
        npt.assert_almost_equal(left, right)


def test_constant_subsequence_self_join():
    T_A = np.concatenate((np.zeros(20, dtype=np.float64), np.ones(5, dtype=np.float64)))
    m = 3
    zone = int(np.ceil(m / 4))
    left = np.array(
        [
            utils.naive_mass(Q, T_A, m, i, zone, True)
            for i, Q in enumerate(core.rolling_window(T_A, m))
        ],
        dtype=object,
    )
    right = gpu_stump(T_A, m, ignore_trivial=True)
    utils.replace_inf(left)
    utils.replace_inf(right)
    npt.assert_almost_equal(left[:, 0], right[:, 0])  # ignore indices

    right = gpu_stump(pd.Series(T_A), m, ignore_trivial=True)
    utils.replace_inf(right)
    npt.assert_almost_equal(left[:, 0], right[:, 0])  # ignore indices


def test_constant_subsequence_A_B_join():
    T_A = np.array([0, 0, 0, 0, 0, 1], dtype=np.float64)
    T_B = np.concatenate((np.zeros(20, dtype=np.float64), np.ones(5, dtype=np.float64)))
    m = 3
    left = np.array(
        [utils.naive_mass(Q, T_A, m) for Q in core.rolling_window(T_B, m)], dtype=object
    )
    right = gpu_stump(T_A, m, T_B, ignore_trivial=False)
    utils.replace_inf(left)
    utils.replace_inf(right)
    npt.assert_almost_equal(left[:, 0], right[:, 0])  # ignore indices

    right = gpu_stump(pd.Series(T_A), m, pd.Series(T_B), ignore_trivial=False)
    utils.replace_inf(right)
    npt.assert_almost_equal(left[:, 0], right[:, 0])  # ignore indices

    # Swap inputs
    left = np.array(
        [utils.naive_mass(Q, T_B, m) for Q in core.rolling_window(T_A, m)], dtype=object
    )
    right = gpu_stump(T_B, m, T_A, ignore_trivial=False)
    utils.replace_inf(left)
    utils.replace_inf(right)
    npt.assert_almost_equal(left[:, 0], right[:, 0])  # ignore indices

    right = gpu_stump(pd.Series(T_B), m, pd.Series(T_A), ignore_trivial=False)
    utils.replace_inf(right)
    npt.assert_almost_equal(left[:, 0], right[:, 0])  # ignore indices
