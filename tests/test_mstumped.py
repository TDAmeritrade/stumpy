import numpy as np
import numpy.testing as npt
import pandas as pd
from stumpy import core, mstumped
import pytest
from dask.distributed import Client, LocalCluster
import warnings
import naive


@pytest.fixture(scope="module")
def dask_cluster():
    cluster = LocalCluster(n_workers=2, threads_per_worker=2)
    yield cluster
    cluster.close()


test_data = [
    (np.array([[584, -11, 23, 79, 1001, 0, -19]], dtype=np.float64), 3),
    (np.random.uniform(-1000, 1000, [5, 20]).astype(np.float64), 5),
]


@pytest.mark.filterwarnings("ignore:\\s+Port 8787 is already in use:UserWarning")
@pytest.mark.parametrize("T, m", test_data)
def test_mstumped(T, m, dask_cluster):
    with Client(dask_cluster) as dask_client:
        excl_zone = int(np.ceil(m / 4))

        left_P, left_I = naive.mstump(T, m, excl_zone)
        right_P, right_I = mstumped(dask_client, T, m)

        npt.assert_almost_equal(left_P, right_P)
        npt.assert_almost_equal(left_I, right_I)


@pytest.mark.filterwarnings("ignore:\\s+Port 8787 is already in use:UserWarning")
@pytest.mark.parametrize("T, m", test_data)
def test_mstumped_include(T, m, dask_cluster):
    with Client(dask_cluster) as dask_client:
        for width in range(T.shape[0]):
            for i in range(T.shape[0] - width):
                include = np.asarray(range(i, i + width + 1))

                excl_zone = int(np.ceil(m / 4))

                left_P, left_I = naive.mstump(T, m, excl_zone, include)
                right_P, right_I = mstumped(dask_client, T, m, include)

                npt.assert_almost_equal(left_P, right_P)
                npt.assert_almost_equal(left_I, right_I)


@pytest.mark.filterwarnings("ignore:\\s+Port 8787 is already in use:UserWarning")
@pytest.mark.parametrize("T, m", test_data)
def test_mstumped_discords(T, m, dask_cluster):
    with Client(dask_cluster) as dask_client:
        excl_zone = int(np.ceil(m / 4))

        left_P, left_I = naive.mstump(T, m, excl_zone, discords=True)
        right_P, right_I = mstumped(dask_client, T, m, discords=True)

        npt.assert_almost_equal(left_P, right_P)
        npt.assert_almost_equal(left_I, right_I)


@pytest.mark.filterwarnings("ignore:\\s+Port 8787 is already in use:UserWarning")
@pytest.mark.parametrize("T, m", test_data)
def test_mstumped_include_discords(T, m, dask_cluster):
    with Client(dask_cluster) as dask_client:
        for width in range(T.shape[0]):
            for i in range(T.shape[0] - width):
                include = np.asarray(range(i, i + width + 1))

                excl_zone = int(np.ceil(m / 4))

                left_P, left_I = naive.mstump(T, m, excl_zone, include, discords=True)
                right_P, right_I = mstumped(dask_client, T, m, include, discords=True)

                npt.assert_almost_equal(left_P, right_P)
                npt.assert_almost_equal(left_I, right_I)


@pytest.mark.filterwarnings("ignore:\\s+Port 8787 is already in use:UserWarning")
@pytest.mark.parametrize("T, m", test_data)
def test_mstumped_df(T, m, dask_cluster):
    with Client(dask_cluster) as dask_client:
        excl_zone = int(np.ceil(m / 4))

        left_P, left_I = naive.mstump(T, m, excl_zone)
        df = pd.DataFrame(T.T)
        right_P, right_I = mstumped(dask_client, df, m)

        npt.assert_almost_equal(left_P, right_P)
        npt.assert_almost_equal(left_I, right_I)


@pytest.mark.filterwarnings("ignore:\\s+Port 8787 is already in use:UserWarning")
def test_constant_subsequence_self_join(dask_cluster):
    with Client(dask_cluster) as dask_client:
        T_A = np.concatenate(
            (np.zeros(20, dtype=np.float64), np.ones(5, dtype=np.float64))
        )
        T = np.array([T_A, T_A, np.random.rand(T_A.shape[0])])
        m = 3

        excl_zone = int(np.ceil(m / 4))

        left_P, left_I = naive.mstump(T, m, excl_zone)
        right_P, right_I = mstumped(dask_client, T, m)

        npt.assert_almost_equal(left_P, right_P)  # ignore indices
