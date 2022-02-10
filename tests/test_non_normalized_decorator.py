import numpy as np
import numpy.testing as npt
import stumpy
from stumpy.mstump import multi_distance_profile
from stumpy.maamp import maamp_multi_distance_profile
from dask.distributed import Client, LocalCluster
from numba import cuda

try:
    from numba.errors import NumbaPerformanceWarning
except ModuleNotFoundError:
    from numba.core.errors import NumbaPerformanceWarning
import pytest


@pytest.fixture(scope="module")
def dask_cluster():
    cluster = LocalCluster(n_workers=2, threads_per_worker=2)
    yield cluster
    cluster.close()


test_data = [
    (np.array([[584, -11, 23, 79, 1001, 0, -19]], dtype=np.float64), 3),
    (np.random.uniform(-1000, 1000, [5, 20]).astype(np.float64), 5),
]


def test_mass():
    Q = np.random.rand(10)
    T = np.random.rand(20)
    ref = stumpy.core.mass_absolute(Q, T)
    comp = stumpy.core.mass(Q, T, normalize=False)
    npt.assert_almost_equal(ref, comp)

    Q = np.random.rand(10)
    T = np.random.rand(20)
    T, T_subseq_isfinite = stumpy.core.preprocess_non_normalized(T, 10)
    T_squared = np.sum(stumpy.core.rolling_window(T * T, Q.shape[0]), axis=-1)
    ref = stumpy.core.mass_absolute(Q, T)
    comp = stumpy.core.mass(Q, T, M_T=T_subseq_isfinite, normalize=False)
    npt.assert_almost_equal(ref, comp)
    comp = stumpy.core.mass(Q, T, Σ_T=T_squared, normalize=False)
    npt.assert_almost_equal(ref, comp)
    comp = stumpy.core.mass(Q, T, M_T=T_subseq_isfinite, Σ_T=T_squared, normalize=False)
    npt.assert_almost_equal(ref, comp)


@pytest.mark.parametrize("T, m", test_data)
def test_stump(T, m):
    if T.ndim > 1:
        T = T.copy()
        T = T[0]

    ref = stumpy.aamp(T, m)
    comp = stumpy.stump(T, m, normalize=False)
    npt.assert_almost_equal(ref, comp)


@pytest.mark.parametrize("T, m", test_data)
def test_prescrump(T, m):
    if T.ndim > 1:
        T = T.copy()
        T = T[0]

    ref = stumpy.prescraamp(T, m)
    comp = stumpy.prescrump(T, m, normalize=False)
    npt.assert_almost_equal(ref, comp)


@pytest.mark.parametrize("T, m", test_data)
def test_scrump(T, m):
    if T.ndim > 1:
        T = T.copy()
        T = T[0]

    seed = np.random.randint(100000)

    np.random.seed(seed)
    ref = stumpy.scraamp(T, m)
    np.random.seed(seed)
    comp = stumpy.scrump(T, m, normalize=False)
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
    seed = np.random.randint(100000)

    np.random.seed(seed)
    ref = stumpy.scraamp(T, m, pre_scraamp=True)
    np.random.seed(seed)
    comp = stumpy.scrump(T, m, pre_scrump=True, normalize=False)
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

    seed = np.random.randint(100000)

    np.random.seed(seed)
    ref = stumpy.scraamp(T, m, percentage=0.1, pre_scraamp=True)
    np.random.seed(seed)
    comp = stumpy.scrump(T, m, percentage=0.1, pre_scrump=True, normalize=False)
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
        ref = stumpy.aamped(dask_client, T, m)
        comp = stumpy.stumped(dask_client, T, m, normalize=False)
        npt.assert_almost_equal(ref, comp)


@pytest.mark.filterwarnings("ignore", category=NumbaPerformanceWarning)
@pytest.mark.parametrize("T, m", test_data)
def test_gpu_stump(T, m):
    if not cuda.is_available():  # pragma: no cover
        pytest.skip("Skipping Tests No GPUs Available")

    if T.ndim > 1:
        T = T.copy()
        T = T[0]

    ref = stumpy.gpu_aamp(T, m)
    comp = stumpy.gpu_stump(T, m, normalize=False)
    npt.assert_almost_equal(ref, comp)


@pytest.mark.parametrize("T, m", test_data)
def test_stumpi(T, m):
    if T.ndim > 1:
        T = T.copy()
        T = T[0]

    ref_stream = stumpy.aampi(T, m)
    comp_stream = stumpy.stumpi(T, m, normalize=False)
    for i in range(10):
        t = np.random.rand()
        ref_stream.update(t)
        comp_stream.update(t)
        npt.assert_almost_equal(ref_stream.P_, comp_stream.P_)


def test_ostinato():
    m = 50
    Ts = [np.random.rand(n) for n in [64, 128, 256]]

    ref_radius, ref_Ts_idx, ref_subseq_idx = stumpy.aamp_ostinato(Ts, m)
    comp_radius, comp_Ts_idx, comp_subseq_idx = stumpy.ostinato(Ts, m, normalize=False)

    npt.assert_almost_equal(ref_radius, comp_radius)
    npt.assert_almost_equal(ref_Ts_idx, comp_Ts_idx)
    npt.assert_almost_equal(ref_subseq_idx, comp_subseq_idx)


@pytest.mark.filterwarnings("ignore:\\s+Port 8787 is already in use:UserWarning")
def test_ostinatoed(dask_cluster):
    m = 50
    Ts = [np.random.rand(n) for n in [64, 128, 256]]

    with Client(dask_cluster) as dask_client:
        ref_radius, ref_Ts_idx, ref_subseq_idx = stumpy.aamp_ostinatoed(
            dask_client, Ts, m
        )
        comp_radius, comp_Ts_idx, comp_subseq_idx = stumpy.ostinatoed(
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
    Ts = [np.random.rand(n) for n in [64, 128, 256]]

    ref_radius, ref_Ts_idx, ref_subseq_idx = stumpy.gpu_aamp_ostinato(Ts, m)
    comp_radius, comp_Ts_idx, comp_subseq_idx = stumpy.gpu_ostinato(
        Ts, m, normalize=False
    )

    npt.assert_almost_equal(ref_radius, comp_radius)
    npt.assert_almost_equal(ref_Ts_idx, comp_Ts_idx)
    npt.assert_almost_equal(ref_subseq_idx, comp_subseq_idx)


def test_mpdist():
    T_A = np.random.uniform(-1000, 1000, [8]).astype(np.float64)
    T_B = np.random.uniform(-1000, 1000, [64]).astype(np.float64)
    m = 5

    ref = stumpy.aampdist(T_A, T_B, m)
    comp = stumpy.mpdist(T_A, T_B, m, normalize=False)
    npt.assert_almost_equal(ref, comp)


def test_mpdisted(dask_cluster):
    T_A = np.random.uniform(-1000, 1000, [8]).astype(np.float64)
    T_B = np.random.uniform(-1000, 1000, [64]).astype(np.float64)
    m = 5

    with Client(dask_cluster) as dask_client:
        ref = stumpy.aampdisted(dask_client, T_A, T_B, m)
        comp = stumpy.mpdisted(dask_client, T_A, T_B, m, normalize=False)
        npt.assert_almost_equal(ref, comp)


@pytest.mark.filterwarnings("ignore", category=NumbaPerformanceWarning)
def test_gpu_mpdist():
    if not cuda.is_available():  # pragma: no cover
        pytest.skip("Skipping Tests No GPUs Available")

    T_A = np.random.uniform(-1000, 1000, [8]).astype(np.float64)
    T_B = np.random.uniform(-1000, 1000, [64]).astype(np.float64)
    m = 5

    ref = stumpy.gpu_aampdist(T_A, T_B, m)
    comp = stumpy.gpu_mpdist(T_A, T_B, m, normalize=False)
    npt.assert_almost_equal(ref, comp)


@pytest.mark.parametrize("T, m", test_data)
def test_multi_distance_profile(T, m):
    for i in range(3):
        ref = maamp_multi_distance_profile(i, T, m)
        comp = multi_distance_profile(i, T, m, normalize=False)
        npt.assert_almost_equal(ref, comp)


@pytest.mark.parametrize("T, m", test_data)
def test_mstump(T, m):
    ref = stumpy.maamp(T, m)
    comp = stumpy.mstump(T, m, normalize=False)
    npt.assert_almost_equal(ref, comp)


@pytest.mark.filterwarnings("ignore:\\s+Port 8787 is already in use:UserWarning")
@pytest.mark.parametrize("T, m", test_data)
def test_mstumped(T, m, dask_cluster):
    with Client(dask_cluster) as dask_client:
        ref = stumpy.maamped(dask_client, T, m)
        comp = stumpy.mstumped(dask_client, T, m, normalize=False)
        npt.assert_almost_equal(ref, comp)


@pytest.mark.parametrize("T, m", test_data)
def test_subspace(T, m):
    subseq_idx = 1
    nn_idx = 4

    for k in range(T.shape[0]):
        ref_S = stumpy.maamp_subspace(T, m, subseq_idx, nn_idx, k)
        comp_S = stumpy.subspace(T, m, subseq_idx, nn_idx, k, normalize=False)
        npt.assert_almost_equal(ref_S, comp_S)


@pytest.mark.parametrize("T, m", test_data)
def test_mdl(T, m):
    subseq_idx = np.full(T.shape[0], 1)
    nn_idx = np.full(T.shape[0], 4)

    ref_MDL, ref_S = stumpy.maamp_mdl(T, m, subseq_idx, nn_idx)
    comp_MDL, comp_S = stumpy.mdl(T, m, subseq_idx, nn_idx, normalize=False)
    npt.assert_almost_equal(ref_MDL, comp_MDL)

    for ref, cmp in zip(ref_S, comp_S):
        npt.assert_almost_equal(ref, cmp)


@pytest.mark.parametrize("T, m", test_data)
def test_motifs(T, m):
    if T.ndim > 1:
        T = T.copy()
        T = T[0]

    mp = stumpy.aamp(T, m)
    ref = stumpy.aamp_motifs(T, mp[:, 0])
    comp = stumpy.motifs(T, mp[:, 0], normalize=False)
    npt.assert_almost_equal(ref, comp)


@pytest.mark.parametrize("T, m", test_data)
def test_match(T, m):
    if T.ndim > 1:
        T = T.copy()
        T = T[0]

    Q = T[:m]
    ref = stumpy.aamp_match(Q, T)
    comp = stumpy.match(Q, T, normalize=False)
    npt.assert_almost_equal(ref, comp)


def test_snippets():
    T = np.random.rand(64)
    m = 10
    k = 2

    (
        ref_snippets,
        ref_indices,
        ref_profiles,
        ref_fractions,
        ref_areas,
        ref_regimes,
    ) = stumpy.aampdist_snippets(T, m, k)
    (
        cmp_snippets,
        cmp_indices,
        cmp_profiles,
        cmp_fractions,
        cmp_areas,
        cmp_regimes,
    ) = stumpy.snippets(T, m, k, normalize=False)
    npt.assert_almost_equal(ref_snippets, cmp_snippets)
