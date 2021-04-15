import numpy as np
import numpy.testing as npt
from stumpy import gpu_mpdist, config
from numba import cuda
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


@pytest.mark.parametrize("T_A, T_B", test_data)
def test_gpu_mpdist(T_A, T_B):
    m = 3
    ref_mpdist = naive.mpdist(T_A, T_B, m)
    comp_mpdist = gpu_mpdist(T_A, T_B, m)

    npt.assert_almost_equal(ref_mpdist, comp_mpdist)
