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
    (np.random.uniform(-1000, 1000, [3, 10]).astype(np.float64), 5),
]

substitution_locations = [slice(0, 0), 0, -1, slice(1, 3), [0, 3]]


@pytest.mark.filterwarnings("ignore:\\s+Port 8787 is already in use:UserWarning")
@pytest.mark.parametrize("T, m", test_data)
@pytest.mark.parametrize("substitution_location", substitution_locations)
def test_mstumped_one_subsequence_inf_self_join_first_dimension(
    T, m, substitution_location, dask_cluster
):
    with Client(dask_cluster) as dask_client:
        excl_zone = int(np.ceil(m / 4))

        T_sub = T.copy()
        T_sub[0, substitution_location] = np.inf

        ref_P, ref_I, _ = naive.mstump(T_sub, m, excl_zone)
        comp_P, comp_I = mstumped(dask_client, T_sub, m)

        npt.assert_almost_equal(ref_P, comp_P)
        npt.assert_almost_equal(ref_I, comp_I)


@pytest.mark.filterwarnings("ignore:\\s+Port 8787 is already in use:UserWarning")
@pytest.mark.parametrize("T, m", test_data)
@pytest.mark.parametrize("substitution_location", substitution_locations)
def test_mstumped_one_subsequence_inf_self_join_all_dimensions(
    T, m, substitution_location, dask_cluster
):
    with Client(dask_cluster) as dask_client:
        excl_zone = int(np.ceil(m / 4))

        T_sub = T.copy()
        T_sub[:, substitution_location] = np.inf

        ref_P, ref_I, _ = naive.mstump(T_sub, m, excl_zone)
        comp_P, comp_I = mstumped(dask_client, T_sub, m)

        npt.assert_almost_equal(ref_P, comp_P)
        npt.assert_almost_equal(ref_I, comp_I)
