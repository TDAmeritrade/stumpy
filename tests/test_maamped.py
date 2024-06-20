import naive
import numpy as np
import numpy.testing as npt
import pandas as pd
import pytest
from dask.distributed import Client, LocalCluster

from stumpy import config, maamped


@pytest.fixture(scope="module")
def dask_cluster():
    cluster = LocalCluster(
        n_workers=2,
        threads_per_worker=2,
        dashboard_address=None,
        worker_dashboard_address=None,
    )
    yield cluster
    cluster.close()


test_data = [
    (np.array([[584, -11, 23, 79, 1001, 0, -19]], dtype=np.float64), 3),
    (np.random.uniform(-1000, 1000, [5, 20]).astype(np.float64), 5),
]

substitution_locations = [slice(0, 0), 0, -1, slice(1, 3), [0, 3]]


def test_maamped_int_input(dask_cluster):
    with pytest.raises(TypeError):
        with Client(dask_cluster) as dask_client:
            maamped(dask_client, np.arange(20).reshape(2, 10), 5)


@pytest.mark.filterwarnings("ignore:\\s+Port 8787 is already in use:UserWarning")
@pytest.mark.parametrize("T, m", test_data)
def test_maamped(T, m, dask_cluster):
    with Client(dask_cluster) as dask_client:
        excl_zone = int(np.ceil(m / 4))

        ref_P, ref_I = naive.maamp(T, m, excl_zone)
        comp_P, comp_I = maamped(dask_client, T, m)

        npt.assert_almost_equal(ref_P, comp_P)
        npt.assert_almost_equal(ref_I, comp_I)


@pytest.mark.filterwarnings("ignore:\\s+Port 8787 is already in use:UserWarning")
@pytest.mark.parametrize("T, m", test_data)
def test_maamped_include(T, m, dask_cluster):
    with Client(dask_cluster) as dask_client:
        for width in range(T.shape[0]):
            for i in range(T.shape[0] - width):
                include = np.asarray(range(i, i + width + 1))

                excl_zone = int(np.ceil(m / 4))

                ref_P, ref_I = naive.maamp(T, m, excl_zone, include)
                comp_P, comp_I = maamped(dask_client, T, m, include)

                npt.assert_almost_equal(ref_P, comp_P)
                npt.assert_almost_equal(ref_I, comp_I)


@pytest.mark.filterwarnings("ignore:\\s+Port 8787 is already in use:UserWarning")
@pytest.mark.parametrize("T, m", test_data)
def test_maamped_discords(T, m, dask_cluster):
    with Client(dask_cluster) as dask_client:
        excl_zone = int(np.ceil(m / 4))

        ref_P, ref_I = naive.maamp(T, m, excl_zone, discords=True)
        comp_P, comp_I = maamped(dask_client, T, m, discords=True)

        npt.assert_almost_equal(ref_P, comp_P)
        npt.assert_almost_equal(ref_I, comp_I)


@pytest.mark.filterwarnings("ignore:\\s+Port 8787 is already in use:UserWarning")
@pytest.mark.parametrize("T, m", test_data)
def test_maamped_include_discords(T, m, dask_cluster):
    with Client(dask_cluster) as dask_client:
        for width in range(T.shape[0]):
            for i in range(T.shape[0] - width):
                include = np.asarray(range(i, i + width + 1))

                excl_zone = int(np.ceil(m / 4))

                ref_P, ref_I = naive.maamp(T, m, excl_zone, include, discords=True)
                comp_P, comp_I = maamped(dask_client, T, m, include, discords=True)

                npt.assert_almost_equal(ref_P, comp_P)
                npt.assert_almost_equal(ref_I, comp_I)


@pytest.mark.filterwarnings("ignore:\\s+Port 8787 is already in use:UserWarning")
@pytest.mark.parametrize("T, m", test_data)
def test_maamped_df(T, m, dask_cluster):
    with Client(dask_cluster) as dask_client:
        excl_zone = int(np.ceil(m / 4))

        ref_P, ref_I = naive.maamp(T, m, excl_zone)
        df = pd.DataFrame(T.T)
        comp_P, comp_I = maamped(dask_client, df, m)

        npt.assert_almost_equal(ref_P, comp_P)
        npt.assert_almost_equal(ref_I, comp_I)


@pytest.mark.filterwarnings("ignore:\\s+Port 8787 is already in use:UserWarning")
def test_maamped_constant_subsequence_self_join(dask_cluster):
    with Client(dask_cluster) as dask_client:
        T_A = np.concatenate(
            (np.zeros(20, dtype=np.float64), np.ones(5, dtype=np.float64))
        )
        T = np.array([T_A, T_A, np.random.rand(T_A.shape[0])])
        m = 3

        excl_zone = int(np.ceil(m / 4))

        ref_P, ref_I = naive.maamp(T, m, excl_zone)
        comp_P, comp_I = maamped(dask_client, T, m)

        npt.assert_almost_equal(ref_P, comp_P)  # ignore indices


@pytest.mark.filterwarnings("ignore:\\s+Port 8787 is already in use:UserWarning")
def test_maamped_identical_subsequence_self_join(dask_cluster):
    with Client(dask_cluster) as dask_client:
        identical = np.random.rand(8)
        T_A = np.random.rand(20)
        T_A[1 : 1 + identical.shape[0]] = identical
        T_A[11 : 11 + identical.shape[0]] = identical
        T = np.array([T_A, T_A, np.random.rand(T_A.shape[0])])
        m = 3

        excl_zone = int(np.ceil(m / 4))

        ref_P, ref_I = naive.maamp(T, m, excl_zone)
        comp_P, comp_I = maamped(dask_client, T, m)

        npt.assert_almost_equal(
            ref_P, comp_P, decimal=config.STUMPY_TEST_PRECISION
        )  # ignore indices


@pytest.mark.filterwarnings("ignore:\\s+Port 8787 is already in use:UserWarning")
@pytest.mark.parametrize("T, m", test_data)
@pytest.mark.parametrize("substitution_location", substitution_locations)
def test_maamped_one_subsequence_inf_self_join_first_dimension(
    T, m, substitution_location, dask_cluster
):
    with Client(dask_cluster) as dask_client:
        excl_zone = int(np.ceil(m / 4))

        T_sub = T.copy()
        T_sub[0, substitution_location] = np.inf

        ref_P, ref_I = naive.maamp(T_sub, m, excl_zone)
        comp_P, comp_I = maamped(dask_client, T_sub, m)

        npt.assert_almost_equal(ref_P, comp_P)
        npt.assert_almost_equal(ref_I, comp_I)


@pytest.mark.filterwarnings("ignore:\\s+Port 8787 is already in use:UserWarning")
@pytest.mark.parametrize("T, m", test_data)
@pytest.mark.parametrize("substitution_location", substitution_locations)
def test_maamped_one_subsequence_inf_self_join_all_dimensions(
    T, m, substitution_location, dask_cluster
):
    with Client(dask_cluster) as dask_client:
        excl_zone = int(np.ceil(m / 4))

        T_sub = T.copy()
        T_sub[:, substitution_location] = np.inf

        ref_P, ref_I = naive.maamp(T_sub, m, excl_zone)
        comp_P, comp_I = maamped(dask_client, T_sub, m)

        npt.assert_almost_equal(ref_P, comp_P)
        npt.assert_almost_equal(ref_I, comp_I)


@pytest.mark.filterwarnings("ignore:\\s+Port 8787 is already in use:UserWarning")
@pytest.mark.parametrize("T, m", test_data)
@pytest.mark.parametrize("substitution_location", substitution_locations)
def test_maamped_one_subsequence_nan_self_join_first_dimension(
    T, m, substitution_location, dask_cluster
):
    with Client(dask_cluster) as dask_client:
        excl_zone = int(np.ceil(m / 4))

        T_sub = T.copy()
        T_sub[0, substitution_location] = np.nan

        ref_P, ref_I = naive.maamp(T_sub, m, excl_zone)
        comp_P, comp_I = maamped(dask_client, T_sub, m)

        npt.assert_almost_equal(ref_P, comp_P)
        npt.assert_almost_equal(ref_I, comp_I)


@pytest.mark.filterwarnings("ignore:\\s+Port 8787 is already in use:UserWarning")
@pytest.mark.parametrize("T, m", test_data)
@pytest.mark.parametrize("substitution_location", substitution_locations)
def test_maamped_one_subsequence_nan_self_join_all_dimensions(
    T, m, substitution_location, dask_cluster
):
    with Client(dask_cluster) as dask_client:
        excl_zone = int(np.ceil(m / 4))

        T_sub = T.copy()
        T_sub[:, substitution_location] = np.nan

        ref_P, ref_I = naive.maamp(T_sub, m, excl_zone)
        comp_P, comp_I = maamped(dask_client, T_sub, m)

        npt.assert_almost_equal(ref_P, comp_P)
        npt.assert_almost_equal(ref_I, comp_I)
