from unittest.mock import patch

import numpy as np
import numpy.testing as npt
from numba import cuda

try:
    from numba.errors import NumbaPerformanceWarning
except ModuleNotFoundError:
    from numba.core.errors import NumbaPerformanceWarning

import naive
import pytest

from stumpy import core, gpu_aamp_ostinato

TEST_THREADS_PER_BLOCK = 10

if not cuda.is_available():  # pragma: no cover
    pytest.skip("Skipping Tests No GPUs Available", allow_module_level=True)


@pytest.mark.filterwarnings("ignore", category=NumbaPerformanceWarning)
@pytest.mark.parametrize(
    "seed", np.random.choice(np.arange(10000), size=2, replace=False)
)
@patch("stumpy.config.STUMPY_THREADS_PER_BLOCK", TEST_THREADS_PER_BLOCK)
def test_random_gpu_aamp_ostinato(seed):
    m = 50
    np.random.seed(seed)
    Ts = [np.random.rand(n) for n in [64, 128, 256]]

    ref_radius, ref_Ts_idx, ref_subseq_idx = naive.aamp_ostinato(Ts, m)
    comp_radius, comp_Ts_idx, comp_subseq_idx = gpu_aamp_ostinato(Ts, m)

    npt.assert_almost_equal(ref_radius, comp_radius)
    npt.assert_almost_equal(ref_Ts_idx, comp_Ts_idx)
    npt.assert_almost_equal(ref_subseq_idx, comp_subseq_idx)


@pytest.mark.filterwarnings("ignore", category=NumbaPerformanceWarning)
@pytest.mark.parametrize("seed", [41, 88])
@patch("stumpy.config.STUMPY_THREADS_PER_BLOCK", TEST_THREADS_PER_BLOCK)
def test_deterministic_gpu_aamp_ostinato(seed):
    m = 50
    np.random.seed(seed)
    Ts = [np.random.rand(n) for n in [64, 128, 256]]

    for p in [1.0, 2.0, 3.0]:
        ref_radius, ref_Ts_idx, ref_subseq_idx = naive.aamp_ostinato(Ts, m, p=p)
        comp_radius, comp_Ts_idx, comp_subseq_idx = gpu_aamp_ostinato(Ts, m, p=p)

        npt.assert_almost_equal(ref_radius, comp_radius)
        npt.assert_almost_equal(ref_Ts_idx, comp_Ts_idx)
        npt.assert_almost_equal(ref_subseq_idx, comp_subseq_idx)


@pytest.mark.filterwarnings("ignore", category=NumbaPerformanceWarning)
@patch("stumpy.config.STUMPY_THREADS_PER_BLOCK", TEST_THREADS_PER_BLOCK)
def test_input_not_overwritten():
    # gpu_aamp_ostinato preprocesses its input, a list of time series,
    # by replacing nan value with 0 in each time series.
    # This test ensures that the original input is not overwritten
    m = 50
    Ts = [np.random.rand(n) for n in [64, 128, 256]]
    for T in Ts:
        T[0] = np.nan

    # raise error if gpu_aamp_ostinato overwrite its input
    Ts_input = [T.copy() for T in Ts]
    gpu_aamp_ostinato(Ts_input, m)
    for i in range(len(Ts)):
        T_ref = Ts[i]
        T_comp = Ts_input[i]
        npt.assert_almost_equal(T_ref[np.isfinite(T_ref)], T_comp[np.isfinite(T_comp)])


@pytest.mark.filterwarnings("ignore", category=NumbaPerformanceWarning)
@patch("stumpy.config.STUMPY_THREADS_PER_BLOCK", TEST_THREADS_PER_BLOCK)
def test_extract_several_consensus():
    # This test is to further ensure that the function `gpu_aamp_ostinato`
    # does not tamper with the original data.
    Ts = [np.random.rand(n) for n in [64, 128]]
    Ts_ref = [T.copy() for T in Ts]
    Ts_comp = [T.copy() for T in Ts]

    m = 20

    k = 2  # Get the first `k` consensus motifs
    for _ in range(k):
        # Find consensus motif and its NN in each time series in Ts_comp
        # Remove them from Ts_comp as well as Ts_ref, and assert that the
        # two time series are the same
        radius, Ts_idx, subseq_idx = gpu_aamp_ostinato(Ts_comp, m)
        consensus_motif = Ts_comp[Ts_idx][subseq_idx : subseq_idx + m].copy()
        for i in range(len(Ts_comp)):
            if i == Ts_idx:
                query_idx = subseq_idx
            else:
                query_idx = None

            idx = np.argmin(
                core.mass(
                    consensus_motif, Ts_comp[i], normalize=False, query_idx=query_idx
                )
            )
            Ts_comp[i][idx : idx + m] = np.nan
            Ts_ref[i][idx : idx + m] = np.nan

            npt.assert_almost_equal(
                Ts_ref[i][np.isfinite(Ts_ref[i])], Ts_comp[i][np.isfinite(Ts_comp[i])]
            )
