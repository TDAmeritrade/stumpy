import numpy as np
import numpy.testing as npt
import pandas as pd
from stumpy import stumped, core
from dask.distributed import Client, LocalCluster
import pytest
import warnings
import utils


@pytest.fixture(scope="module")
def dask_client():
    cluster = LocalCluster(n_workers=2, threads_per_worker=2)
    client = Client(cluster)
    yield client
    # teardown
    client.close()
    cluster.close()


@pytest.mark.filterwarnings("ignore:\\s+A large number of values are smaller")
@pytest.mark.filterwarnings("ignore:\\s+For a self-join")
@pytest.mark.filterwarnings("ignore:numpy.dtype size changed")
@pytest.mark.filterwarnings("ignore:numpy.ufunc size changed")
@pytest.mark.filterwarnings("ignore:numpy.ndarray size changed")
@pytest.mark.filterwarnings("ignore:\\s+Port 8787 is already in use:UserWarning")
def test_two_constant_subsequences_A_B_join_swap(dask_client):
    T_A = np.concatenate(
        (np.zeros(10, dtype=np.float64), np.ones(10, dtype=np.float64))
    )
    T_B = np.concatenate((np.zeros(20, dtype=np.float64), np.ones(5, dtype=np.float64)))
    m = 3
    left = np.array(
        [utils.naive_mass(Q, T_B, m) for Q in core.rolling_window(T_A, m)], dtype=object
    )
    right = stumped(dask_client, T_B, m, T_A, ignore_trivial=False)
    utils.replace_inf(left)
    utils.replace_inf(right)
    npt.assert_almost_equal(left[:, 0], right[:, 0])  # ignore indices


@pytest.mark.filterwarnings("ignore:\\s+A large number of values are smaller")
@pytest.mark.filterwarnings("ignore:\\s+For a self-join")
@pytest.mark.filterwarnings("ignore:numpy.dtype size changed")
@pytest.mark.filterwarnings("ignore:numpy.ufunc size changed")
@pytest.mark.filterwarnings("ignore:numpy.ndarray size changed")
@pytest.mark.filterwarnings("ignore:\\s+Port 8787 is already in use:UserWarning")
def test_constant_subsequence_A_B_join_df_swap(dask_client):
    T_A = np.concatenate(
        (np.zeros(10, dtype=np.float64), np.ones(10, dtype=np.float64))
    )
    T_B = np.concatenate((np.zeros(20, dtype=np.float64), np.ones(5, dtype=np.float64)))
    m = 3
    left = np.array(
        [utils.naive_mass(Q, T_B, m) for Q in core.rolling_window(T_A, m)], dtype=object
    )
    right = stumped(
        dask_client, pd.Series(T_B), m, pd.Series(T_A), ignore_trivial=False
    )
    utils.replace_inf(left)
    utils.replace_inf(right)
    npt.assert_almost_equal(left[:, 0], right[:, 0])  # ignore indices
