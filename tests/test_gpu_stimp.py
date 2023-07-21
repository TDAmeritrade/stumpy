import functools
from unittest.mock import patch

import numpy as np
import numpy.testing as npt
from numba import cuda

from stumpy import gpu_stimp

try:
    from numba.errors import NumbaPerformanceWarning
except ModuleNotFoundError:
    from numba.core.errors import NumbaPerformanceWarning

import naive
import pytest

TEST_THREADS_PER_BLOCK = 10

if not cuda.is_available():  # pragma: no cover
    pytest.skip("Skipping Tests No GPUs Available", allow_module_level=True)


T = [
    np.array([584, -11, 23, 79, 1001, 0, -19], dtype=np.float64),
    np.random.uniform(-1000, 1000, [64]).astype(np.float64),
]


@pytest.mark.filterwarnings("ignore", category=NumbaPerformanceWarning)
@pytest.mark.parametrize("T", T)
@patch("stumpy.config.STUMPY_THREADS_PER_BLOCK", TEST_THREADS_PER_BLOCK)
def test_gpu_stimp(T):
    threshold = 0.2
    min_m = 3
    n = T.shape[0] - min_m + 1

    pan = gpu_stimp(
        T,
        min_m=min_m,
        max_m=None,
        step=1,
        device_id=0
        # normalize=True,
    )

    for i in range(n):
        pan.update()

    ref_PAN = np.full((pan.M_.shape[0], T.shape[0]), fill_value=np.inf)

    for idx, m in enumerate(pan.M_[:n]):
        zone = int(np.ceil(m / 4))
        ref_mp = naive.stump(T, m, T_B=None, exclusion_zone=zone)
        ref_PAN[pan._bfs_indices[idx], : ref_mp.shape[0]] = ref_mp[:, 0]

    # Compare raw pan
    cmp_PAN = pan._PAN

    naive.replace_inf(ref_PAN)
    naive.replace_inf(cmp_PAN)

    npt.assert_almost_equal(ref_PAN, cmp_PAN)

    # Compare transformed pan
    cmp_pan = pan.PAN_
    ref_pan = naive.transform_pan(
        pan._PAN, pan._M, threshold, pan._bfs_indices, pan._n_processed
    )

    naive.replace_inf(ref_pan)
    naive.replace_inf(cmp_pan)

    npt.assert_almost_equal(ref_pan, cmp_pan)


def test_gpu_stimp_1_percent_with_isconstant():
    T = np.random.uniform(-1, 1, [64])
    isconstant_func = functools.partial(
        naive.isconstant_func_stddev_threshold, stddev_threshold=0.5
    )

    threshold = 0.2
    percentage = 0.01
    min_m = 3
    n = T.shape[0] - min_m + 1

    seed = np.random.randint(100000)

    np.random.seed(seed)
    pan = gpu_stimp(
        T,
        min_m=min_m,
        max_m=None,
        step=1,
        percentage=percentage,
        pre_scrump=True,
        # normalize=True,
        T_subseq_isconstant_func=isconstant_func,
    )

    for i in range(n):
        pan.update()

    ref_PAN = np.full((pan.M_.shape[0], T.shape[0]), fill_value=np.inf)

    np.random.seed(seed)
    for idx, m in enumerate(pan.M_[:n]):
        zone = int(np.ceil(m / 4))
        s = zone
        tmp_P, tmp_I = naive.prescrump(
            T,
            m,
            T,
            s=s,
            exclusion_zone=zone,
            T_A_subseq_isconstant=isconstant_func,
            T_B_subseq_isconstant=isconstant_func,
        )
        ref_P, ref_I, _, _ = naive.scrump(
            T,
            m,
            T,
            percentage,
            zone,
            True,
            s,
            T_A_subseq_isconstant=isconstant_func,
            T_B_subseq_isconstant=isconstant_func,
        )
        naive.merge_topk_PI(ref_P, tmp_P, ref_I, tmp_I)
        ref_PAN[pan._bfs_indices[idx], : ref_P.shape[0]] = ref_P

    # Compare raw pan
    cmp_PAN = pan._PAN

    naive.replace_inf(ref_PAN)
    naive.replace_inf(cmp_PAN)

    npt.assert_almost_equal(ref_PAN, cmp_PAN)

    # Compare transformed pan
    cmp_pan = pan.PAN_
    ref_pan = naive.transform_pan(
        pan._PAN, pan._M, threshold, pan._bfs_indices, pan._n_processed
    )

    naive.replace_inf(ref_pan)
    naive.replace_inf(cmp_pan)

    npt.assert_almost_equal(ref_pan, cmp_pan)


@pytest.mark.filterwarnings("ignore:\\s+Port 8787 is already in use:UserWarning")
def test_gpu_stimp_with_isconstant():
    T = np.random.uniform(-1, 1, [64])
    isconstant_func = functools.partial(
        naive.isconstant_func_stddev_threshold, stddev_threshold=0.5
    )

    threshold = 0.2
    min_m = 3
    n = T.shape[0] - min_m + 1

    pan = gpu_stimp(
        T,
        min_m=min_m,
        max_m=None,
        step=1,
        # normalize=True,
        T_subseq_isconstant_func=isconstant_func,
    )

    for i in range(n):
        pan.update()

    ref_PAN = np.full((pan.M_.shape[0], T.shape[0]), fill_value=np.inf)

    for idx, m in enumerate(pan.M_[:n]):
        zone = int(np.ceil(m / 4))
        ref_mp = naive.stump(
            T,
            m,
            T_B=None,
            exclusion_zone=zone,
            T_A_subseq_isconstant=isconstant_func,
        )
        ref_PAN[pan._bfs_indices[idx], : ref_mp.shape[0]] = ref_mp[:, 0]

    # Compare raw pan
    cmp_PAN = pan._PAN

    naive.replace_inf(ref_PAN)
    naive.replace_inf(cmp_PAN)

    npt.assert_almost_equal(ref_PAN, cmp_PAN)

    # Compare transformed pan
    cmp_pan = pan.PAN_
    ref_pan = naive.transform_pan(
        pan._PAN, pan._M, threshold, pan._bfs_indices, pan._n_processed
    )

    naive.replace_inf(ref_pan)
    naive.replace_inf(cmp_pan)

    npt.assert_almost_equal(ref_pan, cmp_pan)
