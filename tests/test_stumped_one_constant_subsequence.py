import numpy as np
import numpy.testing as npt
import pandas as pd
from stumpy import stumped, core, config
from dask.distributed import Client, LocalCluster
import pytest
import warnings
import naive


@pytest.fixture(scope="module")
def dask_cluster():
    cluster = LocalCluster(n_workers=2, threads_per_worker=2)
    yield cluster
    cluster.close()


@pytest.mark.filterwarnings("ignore:A large number of values are smaller")
@pytest.mark.filterwarnings("ignore:For a self-join")
@pytest.mark.filterwarnings("ignore:numpy.dtype size changed")
@pytest.mark.filterwarnings("ignore:numpy.ufunc size changed")
@pytest.mark.filterwarnings("ignore:numpy.ndarray size changed")
@pytest.mark.filterwarnings("ignore:\\s+Port 8787 is already in use:UserWarning")
def test_stumped_one_constant_subsequence_self_join(dask_cluster):
    with Client(dask_cluster) as dask_client:
        T_A = np.concatenate(
            (np.zeros(20, dtype=np.float64), np.ones(5, dtype=np.float64))
        )
        m = 3
        zone = int(np.ceil(m / 4))
        left = naive.stamp(T_A, m, exclusion_zone=zone)
        right = stumped(dask_client, T_A, m, ignore_trivial=True)
        naive.replace_inf(left)
        naive.replace_inf(right)
        npt.assert_almost_equal(left[:, 0], right[:, 0])  # ignore indices


@pytest.mark.filterwarnings("ignore:A large number of values are smaller")
@pytest.mark.filterwarnings("ignore:For a self-join")
@pytest.mark.filterwarnings("ignore:numpy.dtype size changed")
@pytest.mark.filterwarnings("ignore:numpy.ufunc size changed")
@pytest.mark.filterwarnings("ignore:numpy.ndarray size changed")
@pytest.mark.filterwarnings("ignore:\\s+Port 8787 is already in use:UserWarning")
def test_stumped_one_constant_subsequence_self_join_df(dask_cluster):
    with Client(dask_cluster) as dask_client:
        T_A = np.concatenate(
            (np.zeros(20, dtype=np.float64), np.ones(5, dtype=np.float64))
        )
        m = 3
        zone = int(np.ceil(m / 4))
        left = naive.stamp(T_A, m, exclusion_zone=zone)
        right = stumped(dask_client, pd.Series(T_A), m, ignore_trivial=True)
        naive.replace_inf(left)
        naive.replace_inf(right)
        npt.assert_almost_equal(left[:, 0], right[:, 0])  # ignore indices


@pytest.mark.filterwarnings("ignore:\\s+A large number of values are smaller")
@pytest.mark.filterwarnings("ignore:\\s+For a self-join")
@pytest.mark.filterwarnings("ignore:numpy.dtype size changed")
@pytest.mark.filterwarnings("ignore:numpy.ufunc size changed")
@pytest.mark.filterwarnings("ignore:numpy.ndarray size changed")
@pytest.mark.filterwarnings("ignore:\\s+Port 8787 is already in use:UserWarning")
def test_stumped_one_constant_subsequence_A_B_join(dask_cluster):
    with Client(dask_cluster) as dask_client:
        T_A = np.random.rand(20)
        T_B = np.concatenate(
            (np.zeros(20, dtype=np.float64), np.ones(5, dtype=np.float64))
        )
        m = 3
        left = naive.stamp(T_A, m, T_B=T_B)
        right = stumped(dask_client, T_A, m, T_B, ignore_trivial=False)
        naive.replace_inf(left)
        naive.replace_inf(right)
        npt.assert_almost_equal(left[:, 0], right[:, 0])  # ignore indices


@pytest.mark.filterwarnings("ignore:\\s+A large number of values are smaller")
@pytest.mark.filterwarnings("ignore:\\s+For a self-join")
@pytest.mark.filterwarnings("ignore:numpy.dtype size changed")
@pytest.mark.filterwarnings("ignore:numpy.ufunc size changed")
@pytest.mark.filterwarnings("ignore:numpy.ndarray size changed")
@pytest.mark.filterwarnings("ignore:\\s+Port 8787 is already in use:UserWarning")
def test_stumped_one_constant_subsequence_A_B_join_df(dask_cluster):
    with Client(dask_cluster) as dask_client:
        T_A = np.random.rand(20)
        T_B = np.concatenate(
            (np.zeros(20, dtype=np.float64), np.ones(5, dtype=np.float64))
        )
        m = 3
        left = naive.stamp(T_A, m, T_B=T_B)
        right = stumped(
            dask_client, pd.Series(T_A), m, pd.Series(T_B), ignore_trivial=False
        )
        naive.replace_inf(left)
        naive.replace_inf(right)
        npt.assert_almost_equal(left[:, 0], right[:, 0])  # ignore indices


@pytest.mark.filterwarnings("ignore:\\s+A large number of values are smaller")
@pytest.mark.filterwarnings("ignore:\\s+For a self-join")
@pytest.mark.filterwarnings("ignore:numpy.dtype size changed")
@pytest.mark.filterwarnings("ignore:numpy.ufunc size changed")
@pytest.mark.filterwarnings("ignore:numpy.ndarray size changed")
@pytest.mark.filterwarnings("ignore:\\s+Port 8787 is already in use:UserWarning")
def test_stumped_one_constant_subsequence_A_B_join_swap(dask_cluster):
    with Client(dask_cluster) as dask_client:
        T_A = np.random.rand(20)
        T_B = np.concatenate(
            (np.zeros(20, dtype=np.float64), np.ones(5, dtype=np.float64))
        )
        m = 3
        left = naive.stamp(T_A, m, T_B=T_B)
        right = stumped(dask_client, T_A, m, T_B, ignore_trivial=False)
        naive.replace_inf(left)
        naive.replace_inf(right)
        npt.assert_almost_equal(left[:, 0], right[:, 0])  # ignore indices


@pytest.mark.filterwarnings("ignore:\\s+A large number of values are smaller")
@pytest.mark.filterwarnings("ignore:\\s+For a self-join")
@pytest.mark.filterwarnings("ignore:numpy.dtype size changed")
@pytest.mark.filterwarnings("ignore:numpy.ufunc size changed")
@pytest.mark.filterwarnings("ignore:numpy.ndarray size changed")
@pytest.mark.filterwarnings("ignore:\\s+Port 8787 is already in use:UserWarning")
def test_stumped_one_constant_subsequence_A_B_join_df_swap(dask_cluster):
    with Client(dask_cluster) as dask_client:
        T_A = np.random.rand(20)
        T_B = np.concatenate(
            (np.zeros(20, dtype=np.float64), np.ones(5, dtype=np.float64))
        )
        m = 3
        left = naive.stamp(T_A, m, T_B=T_B)
        right = stumped(
            dask_client, pd.Series(T_A), m, pd.Series(T_B), ignore_trivial=False
        )
        naive.replace_inf(left)
        naive.replace_inf(right)
        npt.assert_almost_equal(left[:, 0], right[:, 0])  # ignore indices


@pytest.mark.filterwarnings("ignore:A large number of values are smaller")
@pytest.mark.filterwarnings("ignore:For a self-join")
@pytest.mark.filterwarnings("ignore:numpy.dtype size changed")
@pytest.mark.filterwarnings("ignore:numpy.ufunc size changed")
@pytest.mark.filterwarnings("ignore:numpy.ndarray size changed")
@pytest.mark.filterwarnings("ignore:\\s+Port 8787 is already in use:UserWarning")
def test_stumped_identical_subsequence_self_join(dask_cluster):
    with Client(dask_cluster) as dask_client:
        identical = np.random.rand(8)
        T_A = np.random.rand(20)
        T_A[1 : 1 + identical.shape[0]] = identical
        T_A[11 : 11 + identical.shape[0]] = identical
        m = 3
        zone = int(np.ceil(m / 4))
        left = naive.stamp(T_A, m, exclusion_zone=zone)
        right = stumped(dask_client, T_A, m, ignore_trivial=True)
        naive.replace_inf(left)
        naive.replace_inf(right)
        npt.assert_almost_equal(
            left[:, 0], right[:, 0], decimal=config.STUMPY_TEST_PRECISION
        )  # ignore indices


@pytest.mark.filterwarnings("ignore:\\s+A large number of values are smaller")
@pytest.mark.filterwarnings("ignore:\\s+For a self-join")
@pytest.mark.filterwarnings("ignore:numpy.dtype size changed")
@pytest.mark.filterwarnings("ignore:numpy.ufunc size changed")
@pytest.mark.filterwarnings("ignore:numpy.ndarray size changed")
@pytest.mark.filterwarnings("ignore:\\s+Port 8787 is already in use:UserWarning")
def test_stumped_one_constant_subsequence_A_B_join(dask_cluster):
    with Client(dask_cluster) as dask_client:
        identical = np.random.rand(8)
        T_A = np.random.rand(20)
        T_B = np.random.rand(20)
        T_A[1 : 1 + identical.shape[0]] = identical
        T_B[11 : 11 + identical.shape[0]] = identical
        m = 3
        left = naive.stamp(T_A, m, T_B=T_B)
        right = stumped(dask_client, T_A, m, T_B, ignore_trivial=False)
        naive.replace_inf(left)
        naive.replace_inf(right)
        npt.assert_almost_equal(left[:, 0], right[:, 0], decimal=config.STUMPY_TEST_PRECISION)  # ignore indices
