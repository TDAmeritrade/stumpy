import math
import numpy as np
import numpy.testing as npt
import pandas as pd
from stumpy import core, gpu_stump
from stumpy.gpu_stump import _gpu_searchsorted_left, _gpu_searchsorted_right
from stumpy import config
from numba import cuda

try:
    from numba.errors import NumbaPerformanceWarning
except ModuleNotFoundError:
    from numba.core.errors import NumbaPerformanceWarning
import pytest
import naive

config.THREADS_PER_BLOCK = 10

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


def test_gpu_stump_int_input():
    with pytest.raises(TypeError):
        gpu_stump(np.arange(10), 5, ignore_trivial=True)


@cuda.jit("(f8[:, :], f8[:], i8[:], i8, b1, i8[:])")
def _gpu_searchsorted_kernel(A, V, bfs, nlevel, is_left, IDX):
    # A wrapper kernel for calling device function _gpu_searchsorted_left/right.
    i = cuda.grid(1)
    if i < A.shape[0]:
        if is_left:
            IDX[i] = _gpu_searchsorted_left(A[i], V[i], bfs, nlevel)
        else:
            IDX[i] = _gpu_searchsorted_right(A[i], V[i], bfs, nlevel)


def test_gpu_searchsorted():
    n = 5000
    for k in range(1, 21):
        bfs = core._bfs_indices(k, fill_value=-1)
        nlevel = np.floor(np.log2(k) + 1).astype(np.int64)

        A = np.sort(np.random.rand(n, k), axis=1)
        V = np.empty(n)
        col_idx = np.random.randint(0, k, size=n)
        diff = [-0.001, 0, 0.001]
        for i in range(n):  # creating ties between values of PA and PB
            V[i] = np.random.choice(A[i, col_idx[i]], size=1, replace=False)
            V[i] += diff[i % 3]

        device_A = cuda.to_device(A)
        device_V = cuda.to_device(V)
        device_bfs = cuda.to_device(bfs)
        for is_left in [True, False]:
            if is_left:
                side = "left"
            else:
                side = "right"

            ref_IDX = np.full(n, -1, dtype=np.int64)
            for i in range(n):
                ref_IDX[i] = np.searchsorted(A[i], V[i], side=side)

            comp_IDX = np.full(n, -1, dtype=np.int64)
            device_comp_IDX = cuda.to_device(comp_IDX)

            threads_per_block = config.STUMPY_THREADS_PER_BLOCK
            blocks_per_grid = math.ceil(n / threads_per_block)
            _gpu_searchsorted_kernel[blocks_per_grid, threads_per_block](
                device_A, device_V, device_bfs, nlevel, is_left, device_comp_IDX
            )
            comp_IDX = device_comp_IDX.copy_to_host()

            npt.assert_array_equal(ref_IDX, comp_IDX)


@pytest.mark.filterwarnings("ignore", category=NumbaPerformanceWarning)
@pytest.mark.parametrize("T_A, T_B", test_data)
def test_gpu_stump_self_join(T_A, T_B):
    m = 3
    zone = int(np.ceil(m / 4))
    ref_mp = naive.stump(T_B, m, exclusion_zone=zone, row_wise=True)
    comp_mp = gpu_stump(T_B, m, ignore_trivial=True)
    naive.replace_inf(ref_mp)
    naive.replace_inf(comp_mp)
    npt.assert_almost_equal(ref_mp, comp_mp)

    comp_mp = gpu_stump(pd.Series(T_B), m, ignore_trivial=True)
    naive.replace_inf(comp_mp)
    npt.assert_almost_equal(ref_mp, comp_mp)


@pytest.mark.filterwarnings("ignore", category=NumbaPerformanceWarning)
@pytest.mark.parametrize("T_A, T_B", test_data)
@pytest.mark.parametrize("m", window_size)
def test_gpu_stump_self_join_larger_window(T_A, T_B, m):
    if len(T_B) > m:
        zone = int(np.ceil(m / 4))
        ref_mp = naive.stump(T_B, m, exclusion_zone=zone, row_wise=True)
        comp_mp = gpu_stump(T_B, m, ignore_trivial=True)
        naive.replace_inf(ref_mp)
        naive.replace_inf(comp_mp)

        npt.assert_almost_equal(ref_mp, comp_mp)

        # comp_mp = gpu_stump(
        #     pd.Series(T_B),
        #     m,
        #     ignore_trivial=True,
        # )
        # naive.replace_inf(comp_mp)
        # npt.assert_almost_equal(ref_mp, comp_mp)


@pytest.mark.filterwarnings("ignore", category=NumbaPerformanceWarning)
@pytest.mark.parametrize("T_A, T_B", test_data)
def test_gpu_stump_A_B_join(T_A, T_B):
    m = 3
    ref_mp = naive.stump(T_B, m, T_B=T_A, row_wise=True)
    comp_mp = gpu_stump(T_B, m, T_A, ignore_trivial=False)
    naive.replace_inf(ref_mp)
    naive.replace_inf(comp_mp)
    npt.assert_almost_equal(ref_mp, comp_mp)

    # comp_mp = gpu_stump(pd.Series(T_A), m, pd.Series(T_B), ignore_trivial=False)
    # naive.replace_inf(comp_mp)
    # npt.assert_almost_equal(ref_mp, comp_mp)


@pytest.mark.filterwarnings("ignore", category=NumbaPerformanceWarning)
@pytest.mark.parametrize("T_A, T_B", test_data)
def test_parallel_gpu_stump_self_join(T_A, T_B):
    device_ids = [device.id for device in cuda.list_devices()]
    if len(T_B) > 10:
        m = 3
        zone = int(np.ceil(m / 4))
        ref_mp = naive.stump(T_B, m, exclusion_zone=zone, row_wise=True)
        comp_mp = gpu_stump(
            T_B,
            m,
            ignore_trivial=True,
            device_id=device_ids,
        )
        naive.replace_inf(ref_mp)
        naive.replace_inf(comp_mp)
        npt.assert_almost_equal(ref_mp, comp_mp)

        # comp_mp = gpu_stump(
        #     pd.Series(T_B),
        #     m,
        #     ignore_trivial=True,
        #     device_id=device_ids,
        # )
        # naive.replace_inf(comp_mp)
        # npt.assert_almost_equal(ref_mp, comp_mp)


@pytest.mark.filterwarnings("ignore", category=NumbaPerformanceWarning)
@pytest.mark.parametrize("T_A, T_B", test_data)
def test_parallel_gpu_stump_A_B_join(T_A, T_B):
    device_ids = [device.id for device in cuda.list_devices()]
    if len(T_B) > 10:
        m = 3
        ref_mp = naive.stump(T_B, m, T_B=T_A, row_wise=True)
        comp_mp = gpu_stump(
            T_B,
            m,
            T_A,
            ignore_trivial=False,
            device_id=device_ids,
        )
        naive.replace_inf(ref_mp)
        naive.replace_inf(comp_mp)
        npt.assert_almost_equal(ref_mp, comp_mp)

        # comp_mp = gpu_stump(
        #     pd.Series(T_B),
        #     m,
        #     pd.Series(T_A),
        #     ignore_trivial=False,
        #     device_id=device_ids,
        # )
        # naive.replace_inf(comp_mp)
        # npt.assert_almost_equal(ref_mp, comp_mp)


@pytest.mark.filterwarnings("ignore", category=NumbaPerformanceWarning)
def test_gpu_stump_constant_subsequence_self_join():
    T_A = np.concatenate((np.zeros(20, dtype=np.float64), np.ones(5, dtype=np.float64)))
    m = 3
    zone = int(np.ceil(m / 4))
    ref_mp = naive.stump(T_A, m, exclusion_zone=zone, row_wise=True)
    comp_mp = gpu_stump(T_A, m, ignore_trivial=True)
    naive.replace_inf(ref_mp)
    naive.replace_inf(comp_mp)
    npt.assert_almost_equal(ref_mp[:, 0], comp_mp[:, 0])  # ignore indices

    # comp_mp = gpu_stump(pd.Series(T_A), m, ignore_trivial=True)
    # naive.replace_inf(comp_mp)
    # npt.assert_almost_equal(ref_mp[:, 0], comp_mp[:, 0])  # ignore indices


@pytest.mark.filterwarnings("ignore", category=NumbaPerformanceWarning)
def test_gpu_stump_one_constant_subsequence_A_B_join():
    T_A = np.random.rand(20)
    T_B = np.concatenate((np.zeros(20, dtype=np.float64), np.ones(5, dtype=np.float64)))
    m = 3
    ref_mp = naive.stump(T_B, m, T_B=T_A, row_wise=True)
    comp_mp = gpu_stump(T_B, m, T_A, ignore_trivial=False)
    naive.replace_inf(ref_mp)
    naive.replace_inf(comp_mp)
    npt.assert_almost_equal(ref_mp[:, 0], comp_mp[:, 0])  # ignore indices

    # comp_mp = gpu_stump(pd.Series(T_A), m, pd.Series(T_B), ignore_trivial=False)
    # naive.replace_inf(comp_mp)
    # npt.assert_almost_equal(ref_mp[:, 0], comp_mp[:, 0])  # ignore indices

    # Swap inputs
    ref_mp = naive.stump(T_A, m, T_B=T_B, row_wise=True)
    comp_mp = gpu_stump(T_A, m, T_B, ignore_trivial=False)
    naive.replace_inf(ref_mp)
    naive.replace_inf(comp_mp)
    npt.assert_almost_equal(ref_mp[:, 0], comp_mp[:, 0])  # ignore indices

    # comp_mp = gpu_stump(pd.Series(T_A), m, pd.Series(T_B), ignore_trivial=False)
    # naive.replace_inf(comp_mp)
    # npt.assert_almost_equal(ref_mp[:, 0], comp_mp[:, 0])  # ignore indices


@pytest.mark.filterwarnings("ignore", category=NumbaPerformanceWarning)
def test_gpu_stump_two_constant_subsequences_A_B_join():
    T_A = np.array([0, 0, 0, 0, 0, 1], dtype=np.float64)
    T_B = np.concatenate((np.zeros(20, dtype=np.float64), np.ones(5, dtype=np.float64)))
    m = 3
    ref_mp = naive.stump(T_B, m, T_B=T_A, row_wise=True)
    comp_mp = gpu_stump(T_B, m, T_A, ignore_trivial=False)
    naive.replace_inf(ref_mp)
    naive.replace_inf(comp_mp)
    npt.assert_almost_equal(ref_mp[:, 0], comp_mp[:, 0])  # ignore indices

    # comp_mp = gpu_stump(pd.Series(T_B), m, pd.Series(T_A), ignore_trivial=False)
    # naive.replace_inf(comp_mp)
    # npt.assert_almost_equal(ref_mp[:, 0], comp_mp[:, 0])  # ignore indices

    # Swap inputs
    ref_mp = naive.stump(T_A, m, T_B=T_B, row_wise=True)
    comp_mp = gpu_stump(T_A, m, T_B, ignore_trivial=False)
    naive.replace_inf(ref_mp)
    naive.replace_inf(comp_mp)
    npt.assert_almost_equal(ref_mp[:, 0], comp_mp[:, 0])  # ignore indices

    # comp_mp = gpu_stump(pd.Series(T_A), m, pd.Series(T_B), ignore_trivial=False)
    # naive.replace_inf(comp_mp)
    # npt.assert_almost_equal(ref_mp[:, 0], comp_mp[:, 0])  # ignore indices


@pytest.mark.filterwarnings("ignore", category=NumbaPerformanceWarning)
def test_gpu_stump_identical_subsequence_self_join():
    identical = np.random.rand(8)
    T_A = np.random.rand(20)
    T_A[1 : 1 + identical.shape[0]] = identical
    T_A[11 : 11 + identical.shape[0]] = identical
    m = 3
    zone = int(np.ceil(m / 4))
    ref_mp = naive.stump(T_A, m, exclusion_zone=zone, row_wise=True)
    comp_mp = gpu_stump(T_A, m, ignore_trivial=True)
    naive.replace_inf(ref_mp)
    naive.replace_inf(comp_mp)
    npt.assert_almost_equal(
        ref_mp[:, 0], comp_mp[:, 0], decimal=config.STUMPY_TEST_PRECISION
    )  # ignore indices

    # comp_mp = gpu_stump(pd.Series(T_A), m, ignore_trivial=True)
    # naive.replace_inf(comp_mp)
    # npt.assert_almost_equal(
    #     ref_mp[:, 0], comp_mp[:, 0], decimal=config.STUMPY_TEST_PRECISION
    # )  # ignore indices


@pytest.mark.filterwarnings("ignore", category=NumbaPerformanceWarning)
def test_gpu_stump_identical_subsequence_A_B_join():
    identical = np.random.rand(8)
    T_A = np.random.rand(20)
    T_B = np.random.rand(20)
    T_A[1 : 1 + identical.shape[0]] = identical
    T_B[11 : 11 + identical.shape[0]] = identical
    m = 3
    ref_mp = naive.stump(T_B, m, T_B=T_A, row_wise=True)
    comp_mp = gpu_stump(T_B, m, T_A, ignore_trivial=False)
    naive.replace_inf(ref_mp)
    naive.replace_inf(comp_mp)
    npt.assert_almost_equal(
        ref_mp[:, 0], comp_mp[:, 0], decimal=config.STUMPY_TEST_PRECISION
    )  # ignore indices

    # comp_mp = gpu_stump(pd.Series(T_B), m, pd.Series(T_A), ignore_trivial=False)
    # naive.replace_inf(comp_mp)
    # npt.assert_almost_equal(
    #     ref_mp[:, 0], comp_mp[:, 0], decimal=config.STUMPY_TEST_PRECISION
    # )  # ignore indices

    # Swap inputs
    ref_mp = naive.stump(T_A, m, T_B=T_B, row_wise=True)
    comp_mp = gpu_stump(T_A, m, T_B, ignore_trivial=False)
    naive.replace_inf(ref_mp)
    naive.replace_inf(comp_mp)
    npt.assert_almost_equal(
        ref_mp[:, 0], comp_mp[:, 0], decimal=config.STUMPY_TEST_PRECISION
    )  # ignore indices

    # comp_mp = gpu_stump(pd.Series(T_A), m, pd.Series(T_B), ignore_trivial=False)
    # naive.replace_inf(comp_mp)
    # npt.assert_almost_equal(
    #     ref_mp[:, 0], comp_mp[:, 0], decimal=config.STUMPY_TEST_PRECISION
    # )  # ignore indices


@pytest.mark.filterwarnings("ignore", category=NumbaPerformanceWarning)
@pytest.mark.parametrize("T_A, T_B", test_data)
@pytest.mark.parametrize("substitute_B", substitution_values)
@pytest.mark.parametrize("substitution_locations", substitution_locations)
def test_gpu_stump_nan_inf_self_join(T_A, T_B, substitute_B, substitution_locations):
    m = 3
    stop = 16
    T_B_sub = T_B.copy()[:stop]

    for substitution_location_B in substitution_locations:
        T_B_sub[:] = T_B[:stop]
        T_B_sub[substitution_location_B] = substitute_B

        zone = int(np.ceil(m / 4))
        ref_mp = naive.stump(T_B_sub, m, exclusion_zone=zone, row_wise=True)
        comp_mp = gpu_stump(T_B_sub, m, ignore_trivial=True)
        naive.replace_inf(ref_mp)
        naive.replace_inf(comp_mp)
        npt.assert_almost_equal(ref_mp, comp_mp)

        # comp_mp = gpu_stump(pd.Series(T_B_sub), m, ignore_trivial=True)
        # naive.replace_inf(comp_mp)
        # npt.assert_almost_equal(ref_mp, comp_mp)


@pytest.mark.filterwarnings("ignore", category=NumbaPerformanceWarning)
@pytest.mark.parametrize("T_A, T_B", test_data)
@pytest.mark.parametrize("substitute_A", substitution_values)
@pytest.mark.parametrize("substitute_B", substitution_values)
@pytest.mark.parametrize("substitution_locations", substitution_locations)
def test_gpu_stump_nan_inf_A_B_join(
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

            ref_mp = naive.stump(T_B_sub, m, T_B=T_A_sub, row_wise=True)
            comp_mp = gpu_stump(T_B_sub, m, T_A_sub, ignore_trivial=False)
            naive.replace_inf(ref_mp)
            naive.replace_inf(comp_mp)
            npt.assert_almost_equal(ref_mp, comp_mp)

            # comp_mp = gpu_stump(
            #     pd.Series(T_B_sub), m, pd.Series(T_A_sub), ignore_trivial=False
            # )
            # naive.replace_inf(comp_mp)
            # npt.assert_almost_equal(ref_mp, comp_mp)


@pytest.mark.filterwarnings("ignore", category=NumbaPerformanceWarning)
def test_gpu_stump_nan_zero_mean_self_join():
    T = np.array([-1, 0, 1, np.inf, 1, 0, -1])
    m = 3

    zone = int(np.ceil(m / 4))
    ref_mp = naive.stump(T, m, exclusion_zone=zone, row_wise=True)
    comp_mp = gpu_stump(T, m, ignore_trivial=True)

    naive.replace_inf(ref_mp)
    naive.replace_inf(comp_mp)
    npt.assert_almost_equal(ref_mp, comp_mp)


@pytest.mark.filterwarnings("ignore", category=NumbaPerformanceWarning)
@pytest.mark.parametrize("T_A, T_B", test_data)
def test_gpu_stump_self_join_KNN(T_A, T_B):
    m = 3
    for k in range(1, 4):
        zone = int(np.ceil(m / 4))
        ref_mp = naive.stump(T_B, m, exclusion_zone=zone, row_wise=True, k=k)
        comp_mp = gpu_stump(T_B, m, ignore_trivial=True, k=k)
        naive.replace_inf(ref_mp)
        naive.replace_inf(comp_mp)
        npt.assert_almost_equal(ref_mp, comp_mp)

        comp_mp = gpu_stump(pd.Series(T_B), m, ignore_trivial=True, k=k)
        naive.replace_inf(comp_mp)
        npt.assert_almost_equal(ref_mp, comp_mp)


@pytest.mark.filterwarnings("ignore", category=NumbaPerformanceWarning)
@pytest.mark.parametrize("T_A, T_B", test_data)
def test_gpu_stump_A_B_join_KNN(T_A, T_B):
    for k in range(1, 4):
        m = 3
        ref_mp = naive.stump(T_B, m, T_B=T_A, row_wise=True, k=k)
        comp_mp = gpu_stump(T_B, m, T_A, ignore_trivial=False, k=k)
        naive.replace_inf(ref_mp)
        naive.replace_inf(comp_mp)
        npt.assert_almost_equal(ref_mp, comp_mp)
