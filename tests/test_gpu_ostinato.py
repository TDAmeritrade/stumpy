import functools

import numpy as np
import numpy.testing as npt
from numba import cuda

try:
    from numba.errors import NumbaPerformanceWarning
except ModuleNotFoundError:
    from numba.core.errors import NumbaPerformanceWarning

from unittest.mock import patch

import naive
import pytest

from stumpy import gpu_ostinato

TEST_THREADS_PER_BLOCK = 10

if not cuda.is_available():  # pragma: no cover
    pytest.skip("Skipping Tests No GPUs Available", allow_module_level=True)


@pytest.mark.filterwarnings("ignore", category=NumbaPerformanceWarning)
@pytest.mark.parametrize(
    "seed", np.random.choice(np.arange(10000), size=2, replace=False)
)
@patch("stumpy.config.STUMPY_THREADS_PER_BLOCK", TEST_THREADS_PER_BLOCK)
def test_random_gpu_ostinato(seed):
    m = 50
    np.random.seed(seed)
    Ts = [np.random.rand(n) for n in [64, 128, 256]]

    ref_radius, ref_Ts_idx, ref_subseq_idx = naive.ostinato(Ts, m)
    comp_radius, comp_Ts_idx, comp_subseq_idx = gpu_ostinato(Ts, m)

    npt.assert_almost_equal(ref_radius, comp_radius)
    npt.assert_almost_equal(ref_Ts_idx, comp_Ts_idx)
    npt.assert_almost_equal(ref_subseq_idx, comp_subseq_idx)


@pytest.mark.filterwarnings("ignore", category=NumbaPerformanceWarning)
@pytest.mark.parametrize("seed", [79, 109])
@patch("stumpy.config.STUMPY_THREADS_PER_BLOCK", TEST_THREADS_PER_BLOCK)
def test_deterministic_gpu_ostinato(seed):
    m = 50
    np.random.seed(seed)
    Ts = [np.random.rand(n) for n in [64, 128, 256]]

    ref_radius, ref_Ts_idx, ref_subseq_idx = naive.ostinato(Ts, m)
    comp_radius, comp_Ts_idx, comp_subseq_idx = gpu_ostinato(Ts, m)

    npt.assert_almost_equal(ref_radius, comp_radius)
    npt.assert_almost_equal(ref_Ts_idx, comp_Ts_idx)
    npt.assert_almost_equal(ref_subseq_idx, comp_subseq_idx)


@pytest.mark.parametrize(
    "seed", np.random.choice(np.arange(10000), size=25, replace=False)
)
def test_random_ostinato_with_isconstant(seed):
    isconstant_custom_func = functools.partial(
        naive.isconstant_func_stddev_threshold, quantile_threshold=0.05
    )

    m = 50
    np.random.seed(seed)
    Ts = [np.random.rand(n) for n in [64, 128, 256]]
    Ts_subseq_isconstant = [isconstant_custom_func for _ in range(len(Ts))]

    ref_radius, ref_Ts_idx, ref_subseq_idx = naive.ostinato(
        Ts, m, Ts_subseq_isconstant=Ts_subseq_isconstant
    )
    comp_radius, comp_Ts_idx, comp_subseq_idx = gpu_ostinato(
        Ts, m, Ts_subseq_isconstant=Ts_subseq_isconstant
    )

    npt.assert_almost_equal(ref_radius, comp_radius)
    npt.assert_almost_equal(ref_Ts_idx, comp_Ts_idx)
    npt.assert_almost_equal(ref_subseq_idx, comp_subseq_idx)


@pytest.mark.parametrize("seed", [79, 109, 112, 133, 151, 161, 251, 275, 309, 355])
def test_deterministic_ostinato_with_isconstant(seed):
    isconstant_custom_func = functools.partial(
        naive.isconstant_func_stddev_threshold, quantile_threshold=0.05
    )

    m = 50
    np.random.seed(seed)
    Ts = [np.random.rand(n) for n in [64, 128, 256]]

    l = 64 - m + 1
    subseq_isconsant = np.full(l, 0, dtype=bool)
    subseq_isconsant[np.random.randint(0, l)] = True
    Ts_subseq_isconstant = [
        subseq_isconsant,
        None,
        isconstant_custom_func,
    ]

    ref_radius, ref_Ts_idx, ref_subseq_idx = naive.ostinato(
        Ts, m, Ts_subseq_isconstant=Ts_subseq_isconstant
    )
    comp_radius, comp_Ts_idx, comp_subseq_idx = gpu_ostinato(
        Ts, m, Ts_subseq_isconstant=Ts_subseq_isconstant
    )

    npt.assert_almost_equal(ref_radius, comp_radius)
    npt.assert_almost_equal(ref_Ts_idx, comp_Ts_idx)
    npt.assert_almost_equal(ref_subseq_idx, comp_subseq_idx)
