import numpy as np
import numpy.testing as npt
import pandas as pd
from stumpy import core, mstumped
import pytest
from dask.distributed import Client, LocalCluster
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


test_data = [
    (np.array([[584, -11, 23, 79, 1001, 0, -19]], dtype=np.float64), 3),
    (np.random.uniform(-1000, 1000, [3, 10]).astype(np.float64), 5),
]


@pytest.mark.filterwarnings("ignore:\\s+Port 8787 is already in use:UserWarning")
@pytest.mark.parametrize("T, m", test_data)
def test_mstumped(T, m, dask_client):
    excl_zone = int(np.ceil(m / 4))

    left_P, left_I = utils.naive_mstump(T, m, excl_zone)
    right_P, right_I = mstumped(dask_client, T, m)

    npt.assert_almost_equal(left_P, right_P)
    npt.assert_almost_equal(left_I, right_I)


@pytest.mark.filterwarnings("ignore:\\s+Port 8787 is already in use:UserWarning")
@pytest.mark.parametrize("T, m", test_data)
def test_mstumped_df(T, m, dask_client):
    excl_zone = int(np.ceil(m / 4))

    left_P, left_I = utils.naive_mstump(T, m, excl_zone)
    df = pd.DataFrame(T.T)
    right_P, right_I = mstumped(dask_client, df, m)

    npt.assert_almost_equal(left_P, right_P)
    npt.assert_almost_equal(left_I, right_I)


@pytest.mark.filterwarnings("ignore:\\s+Port 8787 is already in use:UserWarning")
def test_constant_subsequence_self_join(dask_client):
    T_A = np.concatenate((np.zeros(20, dtype=np.float64), np.ones(5, dtype=np.float64)))
    T = np.array([T_A, T_A, np.random.rand(T_A.shape[0])])
    m = 3

    excl_zone = int(np.ceil(m / 4))

    left_P, left_I = utils.naive_mstump(T, m, excl_zone)
    right_P, right_I = mstumped(dask_client, T, m)

    npt.assert_almost_equal(left_P, right_P)  # ignore indices
