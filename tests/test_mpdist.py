import functools

import naive
import numpy as np
import numpy.testing as npt
import pytest
from dask.distributed import Client, LocalCluster

from stumpy import mpdist, mpdisted
from stumpy.mpdist import _mpdist_vect


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
    (
        np.array([9, 8100, -60, 7], dtype=np.float64),
        np.array([584, -11, 23, 79, 1001, 0, -19], dtype=np.float64),
    ),
    (
        np.random.uniform(-1000, 1000, [8]).astype(np.float64),
        np.random.uniform(-1000, 1000, [64]).astype(np.float64),
    ),
]

percentage = [0.25, 0.5, 0.75]
k = [0, 1, 2, 3, 4]


@pytest.mark.parametrize("T_A, T_B", test_data)
def test_mpdist_vect(T_A, T_B):
    m = 3
    ref_mpdist_vect = naive.mpdist_vect(T_A, T_B, m)

    Q_subseq_isconstant = naive.rolling_isconstant(T_A, m)
    T_subseq_isconstant = naive.rolling_isconstant(T_B, m)
    μ_Q, σ_Q = naive.compute_mean_std(T_A, m)
    M_T, Σ_T = naive.compute_mean_std(T_B, m)
    comp_mpdist_vect = _mpdist_vect(
        T_A, T_B, m, μ_Q, σ_Q, M_T, Σ_T, Q_subseq_isconstant, T_subseq_isconstant
    )

    npt.assert_almost_equal(ref_mpdist_vect, comp_mpdist_vect)


@pytest.mark.parametrize("T_A, T_B", test_data)
@pytest.mark.parametrize("percentage", percentage)
def test_mpdist_vect_percentage(T_A, T_B, percentage):
    m = 3
    ref_mpdist_vect = naive.mpdist_vect(T_A, T_B, m, percentage=percentage)

    Q_subseq_isconstant = naive.rolling_isconstant(T_A, m)
    T_subseq_isconstant = naive.rolling_isconstant(T_B, m)
    μ_Q, σ_Q = naive.compute_mean_std(T_A, m)
    M_T, Σ_T = naive.compute_mean_std(T_B, m)
    comp_mpdist_vect = _mpdist_vect(
        T_A,
        T_B,
        m,
        μ_Q,
        σ_Q,
        M_T,
        Σ_T,
        Q_subseq_isconstant,
        T_subseq_isconstant,
        percentage=percentage,
    )

    npt.assert_almost_equal(ref_mpdist_vect, comp_mpdist_vect)


@pytest.mark.parametrize("T_A, T_B", test_data)
@pytest.mark.parametrize("k", k)
def test_mpdist_vect_k(T_A, T_B, k):
    m = 3
    ref_mpdist_vect = naive.mpdist_vect(T_A, T_B, m, k=k)

    Q_subseq_isconstant = naive.rolling_isconstant(T_A, m)
    T_subseq_isconstant = naive.rolling_isconstant(T_B, m)
    μ_Q, σ_Q = naive.compute_mean_std(T_A, m)
    M_T, Σ_T = naive.compute_mean_std(T_B, m)
    comp_mpdist_vect = _mpdist_vect(
        T_A, T_B, m, μ_Q, σ_Q, M_T, Σ_T, Q_subseq_isconstant, T_subseq_isconstant, k=k
    )

    npt.assert_almost_equal(ref_mpdist_vect, comp_mpdist_vect)


@pytest.mark.parametrize("T_A, T_B", test_data)
def test_mpdist(T_A, T_B):
    m = 3
    ref_mpdist = naive.mpdist(T_A, T_B, m)
    comp_mpdist = mpdist(T_A, T_B, m)

    npt.assert_almost_equal(ref_mpdist, comp_mpdist)


@pytest.mark.parametrize("T_A, T_B", test_data)
def test_mpdist_with_isconstant(T_A, T_B):
    m = 3
    T_A_subseq_isconstant = np.random.choice(
        [True, False], size=len(T_A) - m + 1, replace=True
    )
    T_B_subseq_isconstant = np.random.choice(
        [True, False], size=len(T_B) - m + 1, replace=True
    )
    ref_mpdist = naive.mpdist(
        T_A,
        T_B,
        m,
        T_A_subseq_isconstant=T_A_subseq_isconstant,
        T_B_subseq_isconstant=T_B_subseq_isconstant,
    )
    comp_mpdist = mpdist(
        T_A,
        T_B,
        m,
        T_A_subseq_isconstant=T_A_subseq_isconstant,
        T_B_subseq_isconstant=T_B_subseq_isconstant,
    )

    npt.assert_almost_equal(ref_mpdist, comp_mpdist)


@pytest.mark.parametrize("T_A, T_B", test_data)
@pytest.mark.parametrize("percentage", percentage)
def test_mpdist_percentage(T_A, T_B, percentage):
    m = 3
    ref_mpdist = naive.mpdist(T_A, T_B, m, percentage=percentage)
    comp_mpdist = mpdist(T_A, T_B, m, percentage=percentage)

    npt.assert_almost_equal(ref_mpdist, comp_mpdist)


@pytest.mark.parametrize("T_A, T_B", test_data)
@pytest.mark.parametrize("k", k)
def test_mpdist_k(T_A, T_B, k):
    m = 3
    ref_mpdist = naive.mpdist(T_A, T_B, m, k=k)
    comp_mpdist = mpdist(T_A, T_B, m, k=k)

    npt.assert_almost_equal(ref_mpdist, comp_mpdist)


@pytest.mark.filterwarnings("ignore:numpy.dtype size changed")
@pytest.mark.filterwarnings("ignore:numpy.ufunc size changed")
@pytest.mark.filterwarnings("ignore:numpy.ndarray size changed")
@pytest.mark.filterwarnings("ignore:\\s+Port 8787 is already in use:UserWarning")
@pytest.mark.parametrize("T_A, T_B", test_data)
def test_mpdisted(T_A, T_B, dask_cluster):
    with Client(dask_cluster) as dask_client:
        m = 3
        ref_mpdist = naive.mpdist(T_A, T_B, m)
        comp_mpdist = mpdisted(dask_client, T_A, T_B, m)

        npt.assert_almost_equal(ref_mpdist, comp_mpdist)


@pytest.mark.filterwarnings("ignore:numpy.dtype size changed")
@pytest.mark.filterwarnings("ignore:numpy.ufunc size changed")
@pytest.mark.filterwarnings("ignore:numpy.ndarray size changed")
@pytest.mark.filterwarnings("ignore:\\s+Port 8787 is already in use:UserWarning")
@pytest.mark.parametrize("T_A, T_B", test_data)
def test_mpdisted_with_isconstant(T_A, T_B, dask_cluster):
    with Client(dask_cluster) as dask_client:
        m = 3
        T_A_subseq_isconstant = np.random.choice(
            [True, False], size=len(T_A) - m + 1, replace=True
        )
        T_B_subseq_isconstant = np.random.choice(
            [True, False], size=len(T_B) - m + 1, replace=True
        )
        ref_mpdist = naive.mpdist(
            T_A,
            T_B,
            m,
            T_A_subseq_isconstant=T_A_subseq_isconstant,
            T_B_subseq_isconstant=T_B_subseq_isconstant,
        )
        comp_mpdist = mpdisted(
            dask_client,
            T_A,
            T_B,
            m,
            T_A_subseq_isconstant=T_A_subseq_isconstant,
            T_B_subseq_isconstant=T_B_subseq_isconstant,
        )

        npt.assert_almost_equal(ref_mpdist, comp_mpdist)


@pytest.mark.parametrize("T_A, T_B", test_data)
def test_mpdist_vect_with_isconstant(T_A, T_B):
    m = 3

    isconstant_custom_func = functools.partial(
        naive.isconstant_func_stddev_threshold, quantile_threshold=0.05
    )
    T_A_subseq_isconstant = naive.rolling_isconstant(T_A, m, isconstant_custom_func)
    T_B_subseq_isconstant = naive.rolling_isconstant(T_B, m, isconstant_custom_func)

    ref_mpdist_vect = naive.mpdist_vect(
        T_A,
        T_B,
        m,
        T_A_subseq_isconstant=T_A_subseq_isconstant,
        T_B_subseq_isconstant=T_B_subseq_isconstant,
    )

    Q_subseq_isconstant = T_A_subseq_isconstant
    T_subseq_isconstant = T_B_subseq_isconstant
    μ_Q, σ_Q = naive.compute_mean_std(T_A, m)
    M_T, Σ_T = naive.compute_mean_std(T_B, m)
    comp_mpdist_vect = _mpdist_vect(
        T_A, T_B, m, μ_Q, σ_Q, M_T, Σ_T, Q_subseq_isconstant, T_subseq_isconstant
    )

    npt.assert_almost_equal(ref_mpdist_vect, comp_mpdist_vect)
