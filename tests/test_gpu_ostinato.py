import numpy as np
import numpy.testing as npt
from numba import cuda
from numba.core.errors import NumbaPerformanceWarning
from stumpy import gpu_ostinato, config
import naive
import pytest


config.THREADS_PER_BLOCK = 10

if not cuda.is_available():
    pytest.skip("Skipping Tests No GPUs Available", allow_module_level=True)


@pytest.mark.filterwarnings("ignore", category=NumbaPerformanceWarning)
@pytest.mark.parametrize(
    "seed", np.random.choice(np.arange(10000), size=2, replace=False)
)
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
def test_deterministic_gpu_ostinato(seed):
    m = 50
    np.random.seed(seed)
    Ts = [np.random.rand(n) for n in [64, 128, 256]]

    ref_radius, ref_Ts_idx, ref_subseq_idx = naive.ostinato(Ts, m)
    comp_radius, comp_Ts_idx, comp_subseq_idx = gpu_ostinato(Ts, m)

    npt.assert_almost_equal(ref_radius, comp_radius)
    npt.assert_almost_equal(ref_Ts_idx, comp_Ts_idx)
    npt.assert_almost_equal(ref_subseq_idx, comp_subseq_idx)
