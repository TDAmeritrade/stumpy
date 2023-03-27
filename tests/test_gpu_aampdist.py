from unittest.mock import patch

import numpy as np
import numpy.testing as npt
from numba import cuda

from stumpy import gpu_aampdist

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


@pytest.mark.filterwarnings("ignore", category=NumbaPerformanceWarning)
@pytest.mark.parametrize("T_A, T_B", test_data)
@patch("stumpy.config.STUMPY_THREADS_PER_BLOCK", TEST_THREADS_PER_BLOCK)
def test_gpu_aampdist(T_A, T_B):
    m = 3
    for p in [1.0, 2.0, 3.0]:
        ref_mpdist = naive.aampdist(T_A, T_B, m, p=p)
        comp_mpdist = gpu_aampdist(T_A, T_B, m, p=p)

        npt.assert_almost_equal(ref_mpdist, comp_mpdist)
