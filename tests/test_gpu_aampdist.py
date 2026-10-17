from unittest.mock import patch

import numpy as np
import numpy.testing as npt

try:
    from numba_cuda_mlir import cuda
except ModuleNotFoundError:
    from numba import cuda

if cuda.is_available():
    from stumpy.gpu_aampdist import gpu_aampdist
else:  # pragma: no cover
    from stumpy.core import _gpu_aampdist_dnf as gpu_aampdist  # noqa: F401

try:
    from numba.errors import NumbaPerformanceWarning
except ModuleNotFoundError:
    from numba.core.errors import NumbaPerformanceWarning

import naive
import pytest

from stumpy import rng

TEST_THREADS_PER_BLOCK = 10

if not cuda.is_available():  # pragma: no cover
    pytest.skip("Skipping Tests No GPUs Available", allow_module_level=True)


test_data = [
    (
        np.array([9, 8100, -60, 7], dtype=np.float64),
        np.array([584, -11, 23, 79, 1001, 0, -19], dtype=np.float64),
    ),
    (
        rng.RNG.uniform(-1000, 1000, [8]).astype(np.float64),
        rng.RNG.uniform(-1000, 1000, [64]).astype(np.float64),
    ),
]


@pytest.mark.filterwarnings("ignore", category=NumbaPerformanceWarning)
@pytest.mark.parametrize("T_A, T_B", test_data)
@patch("stumpy.config.STUMPY_THREADS_PER_BLOCK", TEST_THREADS_PER_BLOCK)
def test_gpu_aampdist(T_A, T_B):
    m = 3
    for p in [1.0, 2.0, 3.0]:
        ref_mpdist = naive.aampdist(T_A, T_B, m, p=p)
        cmp_mpdist = gpu_aampdist(T_A, T_B, m, p=p)

        npt.assert_allclose(cmp_mpdist, ref_mpdist, atol=1.5e-07)
