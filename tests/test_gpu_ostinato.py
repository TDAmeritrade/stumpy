import functools

import numpy as np
import numpy.testing as npt

try:
    from numba_cuda_mlir import cuda
except ModuleNotFoundError:
    from numba import cuda

try:
    from numba.errors import NumbaPerformanceWarning
except ModuleNotFoundError:
    from numba.core.errors import NumbaPerformanceWarning

from unittest.mock import patch

import mersenne
import naive
import pytest

from stumpy import core, rng

if cuda.is_available():
    from stumpy.gpu_ostinato import gpu_ostinato
else:  # pragma: no cover
    from stumpy.core import _gpu_ostinato_dnf as gpu_ostinato  # noqa: F401


TEST_THREADS_PER_BLOCK = 10

if not cuda.is_available():  # pragma: no cover
    pytest.skip("Skipping Tests No GPUs Available", allow_module_level=True)


@pytest.mark.filterwarnings("ignore", category=NumbaPerformanceWarning)
@pytest.mark.parametrize("runs", range(2))
@patch("stumpy.config.STUMPY_THREADS_PER_BLOCK", TEST_THREADS_PER_BLOCK)
def test_random_gpu_ostinato(runs):
    m = 50
    Ts = [rng.RNG.rand(n) for n in [64, 128, 256]]

    ref_radius, ref_Ts_idx, ref_subseq_idx = naive.ostinato(Ts, m)
    cmp_radius, cmp_Ts_idx, cmp_subseq_idx = gpu_ostinato(Ts, m)

    npt.assert_allclose(cmp_radius, ref_radius, atol=1.5e-07)
    npt.assert_allclose(cmp_Ts_idx, ref_Ts_idx, atol=1.5e-07)
    npt.assert_allclose(cmp_subseq_idx, ref_subseq_idx, atol=1.5e-07)


@pytest.mark.filterwarnings("ignore", category=NumbaPerformanceWarning)
@pytest.mark.parametrize("seed", [79, 109])
@patch("stumpy.config.STUMPY_THREADS_PER_BLOCK", TEST_THREADS_PER_BLOCK)
def test_deterministic_gpu_ostinato(seed):
    m = 50
    state = mersenne.seed_to_state(seed)
    with rng.fix_state(state):
        Ts = [rng.RNG.rand(n) for n in [64, 128, 256]]

        ref_radius, ref_Ts_idx, ref_subseq_idx = naive.ostinato(Ts, m)
        cmp_radius, cmp_Ts_idx, cmp_subseq_idx = gpu_ostinato(Ts, m)

        npt.assert_allclose(cmp_radius, ref_radius, atol=1.5e-07)
        npt.assert_allclose(cmp_Ts_idx, ref_Ts_idx, atol=1.5e-07)
        npt.assert_allclose(cmp_subseq_idx, ref_subseq_idx, atol=1.5e-07)


@pytest.mark.filterwarnings("ignore", category=NumbaPerformanceWarning)
@pytest.mark.parametrize("runs", range(25))
@patch("stumpy.config.STUMPY_THREADS_PER_BLOCK", TEST_THREADS_PER_BLOCK)
def test_random_gpu_ostinato_with_isconstant(runs):
    isconstant_custom_func = functools.partial(
        naive.isconstant_func_stddev_threshold, quantile_threshold=0.05
    )

    m = 50
    Ts = [rng.RNG.rand(n) for n in [64, 128, 256]]
    Ts_subseq_isconstant = [isconstant_custom_func for _ in range(len(Ts))]

    ref_radius, ref_Ts_idx, ref_subseq_idx = naive.ostinato(
        Ts, m, Ts_subseq_isconstant=Ts_subseq_isconstant
    )
    cmp_radius, cmp_Ts_idx, cmp_subseq_idx = gpu_ostinato(
        Ts, m, Ts_subseq_isconstant=Ts_subseq_isconstant
    )

    npt.assert_allclose(cmp_radius, ref_radius, atol=1.5e-07)
    npt.assert_allclose(cmp_Ts_idx, ref_Ts_idx, atol=1.5e-07)
    npt.assert_allclose(cmp_subseq_idx, ref_subseq_idx, atol=1.5e-07)


@pytest.mark.filterwarnings("ignore", category=NumbaPerformanceWarning)
@pytest.mark.parametrize("seed", [79, 109, 112, 133, 151, 161, 251, 275, 309, 355])
@patch("stumpy.config.STUMPY_THREADS_PER_BLOCK", TEST_THREADS_PER_BLOCK)
def test_deterministic_gpu_ostinato_with_isconstant(seed):
    isconstant_custom_func = functools.partial(
        naive.isconstant_func_stddev_threshold, quantile_threshold=0.05
    )

    m = 50
    state = mersenne.seed_to_state(seed)
    with rng.fix_state(state):
        Ts = [rng.RNG.rand(n) for n in [64, 128, 256]]

        l = 64 - m + 1
        subseq_isconsant = np.full(l, 0, dtype=bool)
        subseq_isconsant[rng.RNG.randint(0, l)] = True
        Ts_subseq_isconstant = [
            subseq_isconsant,
            None,
            isconstant_custom_func,
        ]

        ref_radius, ref_Ts_idx, ref_subseq_idx = naive.ostinato(
            Ts, m, Ts_subseq_isconstant=Ts_subseq_isconstant
        )
        cmp_radius, cmp_Ts_idx, cmp_subseq_idx = gpu_ostinato(
            Ts, m, Ts_subseq_isconstant=Ts_subseq_isconstant
        )

        npt.assert_allclose(cmp_radius, ref_radius, atol=1.5e-07)
        npt.assert_allclose(cmp_Ts_idx, ref_Ts_idx, atol=1.5e-07)
        npt.assert_allclose(cmp_subseq_idx, ref_subseq_idx, atol=1.5e-07)


@pytest.mark.filterwarnings("ignore", category=NumbaPerformanceWarning)
@patch("stumpy.config.STUMPY_THREADS_PER_BLOCK", TEST_THREADS_PER_BLOCK)
def test_input_not_overwritten():
    # gpu_ostinato preprocesses its input, a list of time series,
    # by replacing nan value with 0 in each time series.
    # This test ensures that the original input is not overwritten
    m = 50
    Ts = [rng.RNG.rand(n) for n in [64, 128, 256]]
    for T in Ts:
        T[0] = np.nan

    # raise error if gpu_ostinato overwrite its input
    Ts_input = [T.copy() for T in Ts]
    gpu_ostinato(Ts_input, m)
    for i in range(len(Ts)):
        T_ref = Ts[i]
        T_cmp = Ts_input[i]
        npt.assert_allclose(
            T_cmp[np.isfinite(T_cmp)], T_ref[np.isfinite(T_ref)], atol=1.5e-07
        )


@pytest.mark.filterwarnings("ignore", category=NumbaPerformanceWarning)
@patch("stumpy.config.STUMPY_THREADS_PER_BLOCK", TEST_THREADS_PER_BLOCK)
def test_extract_several_consensus():
    # This test is to further ensure that the function `gpu_ostinato`
    # does not tamper with the original data.
    Ts = [rng.RNG.rand(n) for n in [64, 128]]
    Ts_ref = [T.copy() for T in Ts]
    Ts_cmp = [T.copy() for T in Ts]

    m = 20

    k = 2  # Get the first `k` consensus motifs
    for _ in range(k):
        # Find consensus motif and its NN in each time series in Ts_cmp
        # Remove them from Ts_cmp as well as Ts_ref, and assert that the
        # two time series are the same
        radius, Ts_idx, subseq_idx = gpu_ostinato(Ts_cmp, m)
        consensus_motif = Ts_cmp[Ts_idx][subseq_idx : subseq_idx + m].copy()
        for i in range(len(Ts_cmp)):
            if i == Ts_idx:
                query_idx = subseq_idx
            else:
                query_idx = None

            idx = np.argmin(core.mass(consensus_motif, Ts_cmp[i], query_idx=query_idx))
            Ts_cmp[i][idx : idx + m] = np.nan
            Ts_ref[i][idx : idx + m] = np.nan

            npt.assert_allclose(
                Ts_cmp[i][np.isfinite(Ts_cmp[i])],
                Ts_ref[i][np.isfinite(Ts_ref[i])],
                atol=1.5e-07,
            )
