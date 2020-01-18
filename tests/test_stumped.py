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
    cluster = LocalCluster(n_workers=None, threads_per_worker=2)
    client = Client(cluster)
    yield client
    # teardown
    client.close()
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


@pytest.mark.filterwarnings("ignore:numpy.dtype size changed")
@pytest.mark.filterwarnings("ignore:numpy.ufunc size changed")
@pytest.mark.filterwarnings("ignore:numpy.ndarray size changed")
@pytest.mark.filterwarnings("ignore:\\s+Port 8787 is already in use:UserWarning")
@pytest.mark.parametrize("T_A, T_B", test_data)
def test_stumped_self_join(T_A, T_B, dask_client):
    m = 3
    zone = int(np.ceil(m / 4))
    left = np.array(
        [
            utils.naive_mass(Q, T_B, m, i, zone, True)
            for i, Q in enumerate(core.rolling_window(T_B, m))
        ],
        dtype=object,
    )
    right = stumped(dask_client, T_B, m, ignore_trivial=True)
    utils.replace_inf(left)
    utils.replace_inf(right)
    npt.assert_almost_equal(left, right)


@pytest.mark.filterwarnings("ignore:numpy.dtype size changed")
@pytest.mark.filterwarnings("ignore:numpy.ufunc size changed")
@pytest.mark.filterwarnings("ignore:numpy.ndarray size changed")
@pytest.mark.filterwarnings("ignore:\\s+Port 8787 is already in use:UserWarning")
@pytest.mark.parametrize("T_A, T_B", test_data)
def test_stumped_self_join_df(T_A, T_B, dask_client):
    m = 3
    zone = int(np.ceil(m / 4))
    left = np.array(
        [
            utils.naive_mass(Q, T_B, m, i, zone, True)
            for i, Q in enumerate(core.rolling_window(T_B, m))
        ],
        dtype=object,
    )
    right = stumped(dask_client, pd.Series(T_B), m, ignore_trivial=True)
    utils.replace_inf(left)
    utils.replace_inf(right)
    npt.assert_almost_equal(left, right)


@pytest.mark.filterwarnings("ignore:numpy.dtype size changed")
@pytest.mark.filterwarnings("ignore:numpy.ufunc size changed")
@pytest.mark.filterwarnings("ignore:numpy.ndarray size changed")
@pytest.mark.filterwarnings("ignore:\\s+Port 8787 is already in use:UserWarning")
@pytest.mark.parametrize("T_A, T_B", test_data)
def test_stump_self_join_larger_window(T_A, T_B, dask_client):
    for m in [8, 16, 32]:
        if len(T_B) > m:
            zone = int(np.ceil(m / 4))
            left = np.array(
                [
                    utils.naive_mass(Q, T_B, m, i, zone, True)
                    for i, Q in enumerate(core.rolling_window(T_B, m))
                ],
                dtype=object,
            )
            right = stumped(dask_client, T_B, m, ignore_trivial=True)
            utils.replace_inf(left)
            utils.replace_inf(right)

            npt.assert_almost_equal(left, right)


@pytest.mark.filterwarnings("ignore:numpy.dtype size changed")
@pytest.mark.filterwarnings("ignore:numpy.ufunc size changed")
@pytest.mark.filterwarnings("ignore:numpy.ndarray size changed")
@pytest.mark.filterwarnings("ignore:\\s+Port 8787 is already in use:UserWarning")
@pytest.mark.parametrize("T_A, T_B", test_data)
def test_stump_self_join_larger_window_df(T_A, T_B, dask_client):
    for m in [8, 16, 32]:
        if len(T_B) > m:
            zone = int(np.ceil(m / 4))
            left = np.array(
                [
                    utils.naive_mass(Q, T_B, m, i, zone, True)
                    for i, Q in enumerate(core.rolling_window(T_B, m))
                ],
                dtype=object,
            )
            right = stumped(dask_client, pd.Series(T_B), m, ignore_trivial=True)
            utils.replace_inf(left)
            utils.replace_inf(right)

            npt.assert_almost_equal(left, right)


@pytest.mark.filterwarnings("ignore:numpy.dtype size changed")
@pytest.mark.filterwarnings("ignore:numpy.ufunc size changed")
@pytest.mark.filterwarnings("ignore:numpy.ndarray size changed")
@pytest.mark.filterwarnings("ignore:\\s+Port 8787 is already in use:UserWarning")
@pytest.mark.parametrize("T_A, T_B", test_data)
def test_stumped_A_B_join(T_A, T_B, dask_client):
    m = 3
    left = np.array(
        [utils.naive_mass(Q, T_A, m) for Q in core.rolling_window(T_B, m)], dtype=object
    )
    right = stumped(dask_client, T_A, m, T_B, ignore_trivial=False)
    utils.replace_inf(left)
    utils.replace_inf(right)
    npt.assert_almost_equal(left, right)


@pytest.mark.filterwarnings("ignore:numpy.dtype size changed")
@pytest.mark.filterwarnings("ignore:numpy.ufunc size changed")
@pytest.mark.filterwarnings("ignore:numpy.ndarray size changed")
@pytest.mark.filterwarnings("ignore:\\s+Port 8787 is already in use:UserWarning")
@pytest.mark.parametrize("T_A, T_B", test_data)
def test_stumped_A_B_join_df(T_A, T_B, dask_client):
    m = 3
    left = np.array(
        [utils.naive_mass(Q, T_A, m) for Q in core.rolling_window(T_B, m)], dtype=object
    )
    right = stumped(
        dask_client, pd.Series(T_A), m, pd.Series(T_B), ignore_trivial=False
    )
    utils.replace_inf(left)
    utils.replace_inf(right)
    npt.assert_almost_equal(left, right)
