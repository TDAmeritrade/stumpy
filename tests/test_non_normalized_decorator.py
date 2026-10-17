import naive
import numpy as np
import numpy.testing as npt
import tornado.ioloop
from dask.distributed import Client, LocalCluster

try:
    from numba_cuda_mlir import cuda
except ModuleNotFoundError:
    from numba import cuda

from stumpy import core, rng
from stumpy.aamp import aamp
from stumpy.aamp_mmotifs import aamp_mmotifs
from stumpy.aamp_motifs import aamp_match, aamp_motifs
from stumpy.aamp_ostinato import aamp_ostinato, aamp_ostinatoed
from stumpy.aamp_stimp import aamp_stimp, aamp_stimped
from stumpy.aampdist import aampdist, aampdisted
from stumpy.aampdist_snippets import aampdist_snippets
from stumpy.aamped import aamped
from stumpy.aampi import aampi

if cuda.is_available():
    from stumpy.gpu_aamp import gpu_aamp
    from stumpy.gpu_aamp_ostinato import gpu_aamp_ostinato
    from stumpy.gpu_aamp_stimp import gpu_aamp_stimp
    from stumpy.gpu_aampdist import gpu_aampdist
    from stumpy.gpu_mpdist import gpu_mpdist
    from stumpy.gpu_ostinato import gpu_ostinato
    from stumpy.gpu_stimp import gpu_stimp
    from stumpy.gpu_stump import gpu_stump
else:  # pragma: no cover
    from stumpy.core import _gpu_aamp_dnf as gpu_aamp  # noqa: F401
    from stumpy.core import (
        _gpu_aamp_ostinato_dnf as gpu_aamp_ostinato,
    )  # noqa: F401
    from stumpy.core import (
        _gpu_aamp_stimp_dnf as gpu_aamp_stimp,
    )  # noqa: F401
    from stumpy.core import _gpu_aampdist_dnf as gpu_aampdist  # noqa: F401
    from stumpy.core import _gpu_mpdist_dnf as gpu_mpdist  # noqa: F401
    from stumpy.core import _gpu_ostinato_dnf as gpu_ostinato  # noqa: F401
    from stumpy.core import _gpu_stimp_dnf as gpu_stimp  # noqa: F401
    from stumpy.core import _gpu_stump_dnf as gpu_stump  # noqa: F401

from stumpy.maamp import maamp, maamp_mdl, maamp_multi_distance_profile, maamp_subspace
from stumpy.maamped import maamped
from stumpy.mmotifs import mmotifs
from stumpy.motifs import match, motifs
from stumpy.mpdist import mpdist, mpdisted
from stumpy.mstump import mdl, mstump, multi_distance_profile, subspace
from stumpy.mstumped import mstumped
from stumpy.ostinato import ostinato, ostinatoed
from stumpy.scraamp import prescraamp, scraamp
from stumpy.scrump import prescrump, scrump
from stumpy.snippets import snippets
from stumpy.stimp import stimp, stimped
from stumpy.stump import stump
from stumpy.stumped import stumped
from stumpy.stumpi import stumpi

try:
    from numba.errors import NumbaPerformanceWarning
except ModuleNotFoundError:
    from numba.core.errors import NumbaPerformanceWarning

import pytest


@pytest.fixture(scope="module")
def dask_cluster():
    cluster = LocalCluster(
        n_workers=2,
        threads_per_worker=2,
        dashboard_address=None,
        worker_dashboard_address=None,
    )
    yield cluster.scheduler_address
    try:
        cluster.close(timeout=60)
    except tornado.ioloop.TimeoutError:  # pragma: no cover
        pass


test_data = [
    (np.array([[584, -11, 23, 79, 1001, 0, -19]], dtype=np.float64), 3),
    (rng.RNG.uniform(-1000, 1000, [5, 20]).astype(np.float64), 5),
]


def test_mass():
    Q = rng.RNG.random(10)
    T = rng.RNG.random(20)
    ref = core.mass_absolute(Q, T)
    comp = core.mass(Q, T, normalize=False)
    npt.assert_almost_equal(ref, comp)

    Q = rng.RNG.random(10)
    T = rng.RNG.random(20)
    T, T_subseq_isfinite = core.preprocess_non_normalized(T, 10)
    T_squared = np.sum(core.rolling_window(T * T, Q.shape[0]), axis=-1)
    ref = core.mass_absolute(Q, T)
    comp = core.mass(Q, T, M_T=T_subseq_isfinite, normalize=False)
    npt.assert_almost_equal(ref, comp)
    comp = core.mass(Q, T, Σ_T=T_squared, normalize=False)
    npt.assert_almost_equal(ref, comp)
    comp = core.mass(Q, T, M_T=T_subseq_isfinite, Σ_T=T_squared, normalize=False)
    npt.assert_almost_equal(ref, comp)


@pytest.mark.parametrize("T, m", test_data)
def test_stump(T, m):
    if T.ndim > 1:
        T = T.copy()
        T = T[0]

    ref = aamp(T, m)
    comp = stump(T, m, normalize=False)
    npt.assert_almost_equal(ref, comp)


@pytest.mark.parametrize("T, m", test_data)
def test_prescrump(T, m):
    if T.ndim > 1:
        T = T.copy()
        T = T[0]

    ref = prescraamp(T, m)
    comp = prescrump(T, m, normalize=False)
    npt.assert_almost_equal(ref, comp)


@pytest.mark.parametrize("T, m", test_data)
def test_scrump(T, m):
    if T.ndim > 1:
        T = T.copy()
        T = T[0]

    with rng.fix_state():
        ref = scraamp(T, m)
    with rng.fix_state():
        comp = scrump(T, m, normalize=False)
    npt.assert_almost_equal(ref.P_, comp.P_)

    for i in range(10):
        ref.update()
        comp.update()
        npt.assert_almost_equal(ref.P_, comp.P_)


@pytest.mark.parametrize("T, m", test_data)
def test_scrump_plus_plus(T, m):
    if T.ndim > 1:
        T = T.copy()
        T = T[0]
    with rng.fix_state():
        ref = scraamp(T, m, pre_scraamp=True)
    with rng.fix_state():
        comp = scrump(T, m, pre_scrump=True, normalize=False)
    npt.assert_almost_equal(ref.P_, comp.P_)

    for i in range(10):
        ref.update()
        comp.update()
        npt.assert_almost_equal(ref.P_, comp.P_)


@pytest.mark.parametrize("T, m", test_data)
def test_scrump_plus_plus_full(T, m):
    if T.ndim > 1:
        T = T.copy()
        T = T[0]

    with rng.fix_state():
        ref = scraamp(T, m, percentage=0.1, pre_scraamp=True)
    with rng.fix_state():
        comp = scrump(T, m, percentage=0.1, pre_scrump=True, normalize=False)
    npt.assert_almost_equal(ref.P_, comp.P_)

    for i in range(10):
        ref.update()
        comp.update()
        npt.assert_almost_equal(ref.P_, comp.P_)


@pytest.mark.filterwarnings("ignore:\\s+Port 8787 is already in use:UserWarning")
@pytest.mark.parametrize("T, m", test_data)
def test_stumped(T, m, dask_cluster):
    if T.ndim > 1:
        T = T.copy()
        T = T[0]

    with Client(dask_cluster) as dask_client:
        ref = aamped(dask_client, T, m)
        comp = stumped(dask_client, T, m, normalize=False)
        npt.assert_almost_equal(ref, comp)


@pytest.mark.filterwarnings("ignore", category=NumbaPerformanceWarning)
@pytest.mark.parametrize("T, m", test_data)
def test_gpu_stump(T, m):
    if not cuda.is_available():  # pragma: no cover
        pytest.skip("Skipping Tests No GPUs Available")

    if T.ndim > 1:
        T = T.copy()
        T = T[0]

    ref = gpu_aamp(T, m)
    comp = gpu_stump(T, m, normalize=False)
    npt.assert_almost_equal(ref, comp)


@pytest.mark.parametrize("T, m", test_data)
def test_stumpi(T, m):
    if T.ndim > 1:
        T = T.copy()
        T = T[0]

    ref_stream = aampi(T, m)
    comp_stream = stumpi(T, m, normalize=False)
    for i in range(10):
        t = rng.RNG.random()
        ref_stream.update(t)
        comp_stream.update(t)
        npt.assert_almost_equal(ref_stream.P_, comp_stream.P_)


def test_ostinato():
    m = 50
    Ts = [rng.RNG.random(n) for n in [64, 128, 256]]

    ref_radius, ref_Ts_idx, ref_subseq_idx = aamp_ostinato(Ts, m)
    comp_radius, comp_Ts_idx, comp_subseq_idx = ostinato(Ts, m, normalize=False)

    npt.assert_almost_equal(ref_radius, comp_radius)
    npt.assert_almost_equal(ref_Ts_idx, comp_Ts_idx)
    npt.assert_almost_equal(ref_subseq_idx, comp_subseq_idx)


@pytest.mark.filterwarnings("ignore:\\s+Port 8787 is already in use:UserWarning")
def test_ostinatoed(dask_cluster):
    m = 50
    Ts = [rng.RNG.random(n) for n in [64, 128, 256]]

    with Client(dask_cluster) as dask_client:
        ref_radius, ref_Ts_idx, ref_subseq_idx = aamp_ostinatoed(dask_client, Ts, m)
        comp_radius, comp_Ts_idx, comp_subseq_idx = ostinatoed(
            dask_client, Ts, m, normalize=False
        )

        npt.assert_almost_equal(ref_radius, comp_radius)
        npt.assert_almost_equal(ref_Ts_idx, comp_Ts_idx)
        npt.assert_almost_equal(ref_subseq_idx, comp_subseq_idx)


@pytest.mark.filterwarnings("ignore", category=NumbaPerformanceWarning)
def test_gpu_ostinato():
    if not cuda.is_available():  # pragma: no cover
        pytest.skip("Skipping Tests No GPUs Available")

    m = 50
    Ts = [rng.RNG.random(n) for n in [64, 128, 256]]

    ref_radius, ref_Ts_idx, ref_subseq_idx = gpu_aamp_ostinato(Ts, m)
    comp_radius, comp_Ts_idx, comp_subseq_idx = gpu_ostinato(Ts, m, normalize=False)

    npt.assert_almost_equal(ref_radius, comp_radius)
    npt.assert_almost_equal(ref_Ts_idx, comp_Ts_idx)
    npt.assert_almost_equal(ref_subseq_idx, comp_subseq_idx)


def test_mpdist():
    T_A = rng.RNG.uniform(-1000, 1000, size=8).astype(np.float64)
    T_B = rng.RNG.uniform(-1000, 1000, size=64).astype(np.float64)
    m = 5

    ref = aampdist(T_A, T_B, m)
    comp = mpdist(T_A, T_B, m, normalize=False)
    npt.assert_almost_equal(ref, comp)


@pytest.mark.filterwarnings("ignore:\\s+Port 8787 is already in use:UserWarning")
def test_mpdisted(dask_cluster):
    T_A = rng.RNG.uniform(-1000, 1000, size=8).astype(np.float64)
    T_B = rng.RNG.uniform(-1000, 1000, size=64).astype(np.float64)
    m = 5

    with Client(dask_cluster) as dask_client:
        ref = aampdisted(dask_client, T_A, T_B, m)
        comp = mpdisted(dask_client, T_A, T_B, m, normalize=False)
        npt.assert_almost_equal(ref, comp)


@pytest.mark.filterwarnings("ignore", category=NumbaPerformanceWarning)
def test_gpu_mpdist():
    if not cuda.is_available():  # pragma: no cover
        pytest.skip("Skipping Tests No GPUs Available")

    T_A = rng.RNG.uniform(-1000, 1000, size=8).astype(np.float64)
    T_B = rng.RNG.uniform(-1000, 1000, size=64).astype(np.float64)
    m = 5

    ref = gpu_aampdist(T_A, T_B, m)
    comp = gpu_mpdist(T_A, T_B, m, normalize=False)
    npt.assert_almost_equal(ref, comp)


@pytest.mark.parametrize("T, m", test_data)
def test_multi_distance_profile(T, m):
    for i in range(3):
        ref = maamp_multi_distance_profile(i, T, m)
        comp = multi_distance_profile(i, T, m, normalize=False)
        npt.assert_almost_equal(ref, comp)


@pytest.mark.parametrize("T, m", test_data)
def test_mstump(T, m):
    ref = maamp(T, m)
    comp = mstump(T, m, normalize=False)
    npt.assert_almost_equal(ref, comp)


@pytest.mark.filterwarnings("ignore:\\s+Port 8787 is already in use:UserWarning")
@pytest.mark.parametrize("T, m", test_data)
def test_mstumped(T, m, dask_cluster):
    with Client(dask_cluster) as dask_client:
        ref = maamped(dask_client, T, m)
        comp = mstumped(dask_client, T, m, normalize=False)
        npt.assert_almost_equal(ref, comp)


@pytest.mark.parametrize("T, m", test_data)
def test_subspace(T, m):
    subseq_idx = 1
    nn_idx = 4

    for k in range(T.shape[0]):
        ref_S = maamp_subspace(T, m, subseq_idx, nn_idx, k)
        comp_S = subspace(T, m, subseq_idx, nn_idx, k, normalize=False)
        npt.assert_almost_equal(ref_S, comp_S)


@pytest.mark.parametrize("T, m", test_data)
def test_mdl(T, m):
    subseq_idx = np.full(T.shape[0], 1)
    nn_idx = np.full(T.shape[0], 4)

    ref_MDL, ref_S = maamp_mdl(T, m, subseq_idx, nn_idx)
    comp_MDL, comp_S = mdl(T, m, subseq_idx, nn_idx, normalize=False)
    npt.assert_almost_equal(ref_MDL, comp_MDL)

    for ref, cmp in zip(ref_S, comp_S):
        npt.assert_almost_equal(ref, cmp)


@pytest.mark.parametrize("T, m", test_data)
def test_motifs(T, m):
    if T.ndim > 1:
        T = T.copy()
        T = T[0]

    mp = aamp(T, m)
    ref = aamp_motifs(T, mp[:, 0])
    comp = motifs(T, mp[:, 0], normalize=False)
    npt.assert_almost_equal(ref, comp)


@pytest.mark.parametrize("T, m", test_data)
def test_match(T, m):
    if T.ndim > 1:
        T = T.copy()
        T = T[0]

    Q = T[:m]
    ref = aamp_match(Q, T)
    comp = match(Q, T, normalize=False)
    npt.assert_almost_equal(ref, comp)


@pytest.mark.parametrize("T, m", test_data)
def test_mmotifs(T, m):
    mps, indices = maamp(T, m)
    ref_distances, ref_indices, ref_subspaces, ref_mdls = aamp_mmotifs(T, mps, indices)
    cmp_distances, cmp_indices, cmp_subspaces, cmp_mdls = mmotifs(
        T, mps, indices, normalize=False
    )
    npt.assert_almost_equal(ref_distances, cmp_distances)


@pytest.mark.filterwarnings("ignore:All-NaN slice encountered")
def test_snippets():
    T = rng.RNG.random(64)
    m = 10
    k = 2

    (
        ref_snippets,
        ref_indices,
        ref_profiles,
        ref_fractions,
        ref_areas,
        ref_regimes,
    ) = aampdist_snippets(T, m, k)
    (
        cmp_snippets,
        cmp_indices,
        cmp_profiles,
        cmp_fractions,
        cmp_areas,
        cmp_regimes,
    ) = snippets(T, m, k, normalize=False)
    npt.assert_almost_equal(ref_snippets, cmp_snippets)


@pytest.mark.parametrize("T, m", test_data)
def test_stimp(T, m):
    if T.ndim > 1:
        T = T.copy()
        T = T[0]
    n = 3
    with rng.fix_state():
        ref = aamp_stimp(T, m)
        for i in range(n):
            ref.update()

    with rng.fix_state():
        cmp = stimp(T, m, normalize=False)
        for i in range(n):
            cmp.update()

        # Compare raw pan
        ref_PAN = ref._PAN
        cmp_PAN = cmp._PAN

    naive.replace_inf(ref_PAN)
    naive.replace_inf(cmp_PAN)

    npt.assert_almost_equal(ref_PAN, cmp_PAN)

    # Compare transformed pan
    npt.assert_almost_equal(ref.PAN_, cmp.PAN_)


@pytest.mark.filterwarnings("ignore:\\s+Port 8787 is already in use:UserWarning")
@pytest.mark.parametrize("T, m", test_data)
def test_stimped(T, m, dask_cluster):
    if T.ndim > 1:
        T = T.copy()
        T = T[0]
    n = 3
    with Client(dask_cluster) as dask_client:
        with rng.fix_state():
            ref = aamp_stimped(dask_client, T, m)
            for i in range(n):
                ref.update()

        with rng.fix_state():
            cmp = stimped(dask_client, T, m, normalize=False)
            for i in range(n):
                cmp.update()

            # Compare raw pan
            ref_PAN = ref._PAN
            cmp_PAN = cmp._PAN

        naive.replace_inf(ref_PAN)
        naive.replace_inf(cmp_PAN)

        npt.assert_almost_equal(ref_PAN, cmp_PAN)

        # Compare transformed pan
        npt.assert_almost_equal(ref.PAN_, cmp.PAN_)


@pytest.mark.filterwarnings("ignore", category=NumbaPerformanceWarning)
@pytest.mark.parametrize("T, m", test_data)
def test_gpu_stimp(T, m):
    if not cuda.is_available():  # pragma: no cover
        pytest.skip("Skipping Tests No GPUs Available")

    if T.ndim > 1:
        T = T.copy()
        T = T[0]
    n = 3
    with rng.fix_state():
        ref = gpu_aamp_stimp(T, m)
        for i in range(n):
            ref.update()

    with rng.fix_state():
        cmp = gpu_stimp(T, m, normalize=False)
        for i in range(n):
            cmp.update()

        # Compare raw pan
        ref_PAN = ref._PAN
        cmp_PAN = cmp._PAN

    naive.replace_inf(ref_PAN)
    naive.replace_inf(cmp_PAN)

    npt.assert_almost_equal(ref_PAN, cmp_PAN)

    # Compare transformed pan
    npt.assert_almost_equal(ref.PAN_, cmp.PAN_)
