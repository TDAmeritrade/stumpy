import numpy as np
import numpy.testing as npt
from dask.distributed import Client, LocalCluster
import stumpy
import naive
import pytest


@pytest.fixture(scope="module")
def dask_cluster():
    cluster = LocalCluster(n_workers=2, threads_per_worker=2)
    yield cluster
    cluster.close()


@pytest.mark.parametrize(
    "seed", np.random.choice(np.arange(10000), size=25, replace=False)
)
def test_random_ostinato(seed):
    m = 50
    np.random.seed(seed)
    Ts = [np.random.rand(n) for n in [64, 128, 256]]

    ref_radius, ref_Ts_idx, ref_subseq_idx = naive.aamp_ostinato(Ts, m)
    comp_radius, comp_Ts_idx, comp_subseq_idx = stumpy.aamp_ostinato(Ts, m)

    npt.assert_almost_equal(ref_radius, comp_radius)
    npt.assert_almost_equal(ref_Ts_idx, comp_Ts_idx)
    npt.assert_almost_equal(ref_subseq_idx, comp_subseq_idx)


@pytest.mark.parametrize("seed", [41, 88, 290, 292, 310, 328, 538, 556, 563, 570])
def test_deterministic_ostinato(seed):
    m = 50
    np.random.seed(seed)
    Ts = [np.random.rand(n) for n in [64, 128, 256]]

    for p in [1.0, 2.0, 3.0]:
        ref_radius, ref_Ts_idx, ref_subseq_idx = naive.aamp_ostinato(Ts, m, p=p)
        comp_radius, comp_Ts_idx, comp_subseq_idx = stumpy.aamp_ostinato(Ts, m, p=p)

        npt.assert_almost_equal(ref_radius, comp_radius)
        npt.assert_almost_equal(ref_Ts_idx, comp_Ts_idx)
        npt.assert_almost_equal(ref_subseq_idx, comp_subseq_idx)


@pytest.mark.parametrize(
    "seed", np.random.choice(np.arange(10000), size=25, replace=False)
)
def test_random_ostinatoed(seed, dask_cluster):
    with Client(dask_cluster) as dask_client:
        m = 50
        np.random.seed(seed)
        Ts = [np.random.rand(n) for n in [64, 128, 256]]

        ref_radius, ref_Ts_idx, ref_subseq_idx = naive.aamp_ostinato(Ts, m)
        comp_radius, comp_Ts_idx, comp_subseq_idx = stumpy.aamp_ostinatoed(
            dask_client, Ts, m
        )

        npt.assert_almost_equal(ref_radius, comp_radius)
        npt.assert_almost_equal(ref_Ts_idx, comp_Ts_idx)
        npt.assert_almost_equal(ref_subseq_idx, comp_subseq_idx)


@pytest.mark.parametrize("seed", [41, 88, 290, 292, 310, 328, 538, 556, 563, 570])
def test_deterministic_ostinatoed(seed, dask_cluster):
    with Client(dask_cluster) as dask_client:
        m = 50
        np.random.seed(seed)
        Ts = [np.random.rand(n) for n in [64, 128, 256]]

        for p in [1.0, 2.0, 3.0]:
            ref_radius, ref_Ts_idx, ref_subseq_idx = naive.aamp_ostinato(Ts, m, p=p)
            comp_radius, comp_Ts_idx, comp_subseq_idx = stumpy.aamp_ostinatoed(
                dask_client, Ts, m, p=p
            )

            npt.assert_almost_equal(ref_radius, comp_radius)
            npt.assert_almost_equal(ref_Ts_idx, comp_Ts_idx)
            npt.assert_almost_equal(ref_subseq_idx, comp_subseq_idx)
