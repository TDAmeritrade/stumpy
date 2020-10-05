import numpy as np
import numpy.testing as npt
import pandas as pd
from stumpy import stumped, core
from dask.distributed import Client, LocalCluster
import pytest
import warnings
import naive


@pytest.fixture(scope="module")
def dask_cluster():
    cluster = LocalCluster(n_workers=2, threads_per_worker=2)
    yield cluster
    cluster.close()


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

window_size = [8, 16, 32]


def test_stumped_int_input(dask_cluster):
    with pytest.raises(TypeError):
        with Client(dask_cluster) as dask_client:
            stumped(dask_client, np.arange(10), 5)


@pytest.mark.filterwarnings("ignore:numpy.dtype size changed")
@pytest.mark.filterwarnings("ignore:numpy.ufunc size changed")
@pytest.mark.filterwarnings("ignore:numpy.ndarray size changed")
@pytest.mark.filterwarnings("ignore:\\s+Port 8787 is already in use:UserWarning")
@pytest.mark.parametrize("T_A, T_B", test_data)
def test_stumped_self_join(T_A, T_B, dask_cluster):
    with Client(dask_cluster) as dask_client:
        m = 3
        zone = int(np.ceil(m / 4))
        ref_mp = naive.stump(T_B, m, exclusion_zone=zone)
        comp_mp = stumped(dask_client, T_B, m, ignore_trivial=True)
        naive.replace_inf(ref_mp)
        naive.replace_inf(comp_mp)
        npt.assert_almost_equal(ref_mp, comp_mp)


@pytest.mark.filterwarnings("ignore:numpy.dtype size changed")
@pytest.mark.filterwarnings("ignore:numpy.ufunc size changed")
@pytest.mark.filterwarnings("ignore:numpy.ndarray size changed")
@pytest.mark.filterwarnings("ignore:\\s+Port 8787 is already in use:UserWarning")
@pytest.mark.parametrize("T_A, T_B", test_data)
def test_stumped_self_join_df(T_A, T_B, dask_cluster):
    with Client(dask_cluster) as dask_client:
        m = 3
        zone = int(np.ceil(m / 4))
        ref_mp = naive.stump(T_B, m, exclusion_zone=zone)
        comp_mp = stumped(dask_client, pd.Series(T_B), m, ignore_trivial=True)
        naive.replace_inf(ref_mp)
        naive.replace_inf(comp_mp)
        npt.assert_almost_equal(ref_mp, comp_mp)


@pytest.mark.filterwarnings("ignore:numpy.dtype size changed")
@pytest.mark.filterwarnings("ignore:numpy.ufunc size changed")
@pytest.mark.filterwarnings("ignore:numpy.ndarray size changed")
@pytest.mark.filterwarnings("ignore:\\s+Port 8787 is already in use:UserWarning")
@pytest.mark.parametrize("T_A, T_B", test_data)
@pytest.mark.parametrize("m", window_size)
def test_stump_self_join_larger_window(T_A, T_B, m, dask_cluster):
    with Client(dask_cluster) as dask_client:
        for m in [8, 16, 32]:
            if len(T_B) > m:
                zone = int(np.ceil(m / 4))
                ref_mp = naive.stump(T_B, m, exclusion_zone=zone)
                comp_mp = stumped(dask_client, T_B, m, ignore_trivial=True)
                naive.replace_inf(ref_mp)
                naive.replace_inf(comp_mp)

                npt.assert_almost_equal(ref_mp, comp_mp)


@pytest.mark.filterwarnings("ignore:numpy.dtype size changed")
@pytest.mark.filterwarnings("ignore:numpy.ufunc size changed")
@pytest.mark.filterwarnings("ignore:numpy.ndarray size changed")
@pytest.mark.filterwarnings("ignore:\\s+Port 8787 is already in use:UserWarning")
@pytest.mark.parametrize("T_A, T_B", test_data)
@pytest.mark.parametrize("m", window_size)
def test_stump_self_join_larger_window_df(T_A, T_B, m, dask_cluster):
    with Client(dask_cluster) as dask_client:
        for m in [8, 16, 32]:
            if len(T_B) > m:
                zone = int(np.ceil(m / 4))
                ref_mp = naive.stump(T_B, m, exclusion_zone=zone)
                comp_mp = stumped(dask_client, pd.Series(T_B), m, ignore_trivial=True)
                naive.replace_inf(ref_mp)
                naive.replace_inf(comp_mp)

                npt.assert_almost_equal(ref_mp, comp_mp)


@pytest.mark.filterwarnings("ignore:numpy.dtype size changed")
@pytest.mark.filterwarnings("ignore:numpy.ufunc size changed")
@pytest.mark.filterwarnings("ignore:numpy.ndarray size changed")
@pytest.mark.filterwarnings("ignore:\\s+Port 8787 is already in use:UserWarning")
@pytest.mark.parametrize("T_A, T_B", test_data)
def test_stumped_A_B_join(T_A, T_B, dask_cluster):
    with Client(dask_cluster) as dask_client:
        m = 3
        ref_mp = naive.stump(T_A, m, T_B=T_B)
        comp_mp = stumped(dask_client, T_A, m, T_B, ignore_trivial=False)
        naive.replace_inf(ref_mp)
        naive.replace_inf(comp_mp)
        npt.assert_almost_equal(ref_mp, comp_mp)


@pytest.mark.filterwarnings("ignore:numpy.dtype size changed")
@pytest.mark.filterwarnings("ignore:numpy.ufunc size changed")
@pytest.mark.filterwarnings("ignore:numpy.ndarray size changed")
@pytest.mark.filterwarnings("ignore:\\s+Port 8787 is already in use:UserWarning")
@pytest.mark.parametrize("T_A, T_B", test_data)
def test_stumped_A_B_join_df(T_A, T_B, dask_cluster):
    with Client(dask_cluster) as dask_client:
        m = 3
        ref_mp = naive.stump(T_A, m, T_B=T_B)
        comp_mp = stumped(
            dask_client, pd.Series(T_A), m, pd.Series(T_B), ignore_trivial=False
        )
        naive.replace_inf(ref_mp)
        naive.replace_inf(comp_mp)
        npt.assert_almost_equal(ref_mp, comp_mp)
