import math
import numpy as np
import numpy.testing as npt
from stumpy import aampdist, aampdisted
from stumpy.aampdist import _aampdist_vect
from dask.distributed import Client, LocalCluster
import pytest
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

percentage = [0.25, 0.5, 0.75]
k = [0, 1, 2, 3, 4]


@pytest.mark.parametrize("T_A, T_B", test_data)
def test_aampdist_vect(T_A, T_B):
    m = 3
    n_A = T_A.shape[0]
    n_B = T_B.shape[0]
    j = n_A - m + 1  # `k` is reserved for `P_ABBA` selection
    P_ABBA = np.empty(2 * j, dtype=np.float64)
    ref_mpdist_vect = np.empty(n_B - n_A + 1)

    percentage = 0.05
    k = min(math.ceil(percentage * (2 * n_A)), 2 * j - 1)
    k = min(int(k), P_ABBA.shape[0] - 1)

    for i in range(n_B - n_A + 1):
        P_ABBA[:j] = naive.aamp(T_A, m, T_B[i : i + n_A])[:, 0]
        P_ABBA[j:] = naive.aamp(T_B[i : i + n_A], m, T_A)[:, 0]
        P_ABBA.sort()
        ref_mpdist_vect[i] = P_ABBA[k]

    comp_mpdist_vect = _aampdist_vect(T_A, T_B, m)

    npt.assert_almost_equal(ref_mpdist_vect, comp_mpdist_vect)


@pytest.mark.parametrize("T_A, T_B", test_data)
@pytest.mark.parametrize("percentage", percentage)
def test_aampdist_vect_percentage(T_A, T_B, percentage):
    m = 3
    n_A = T_A.shape[0]
    n_B = T_B.shape[0]
    j = n_A - m + 1  # `k` is reserved for `P_ABBA` selection
    P_ABBA = np.empty(2 * j, dtype=np.float64)
    ref_mpdist_vect = np.empty(n_B - n_A + 1)

    k = min(math.ceil(percentage * (2 * n_A)), 2 * j - 1)
    k = min(int(k), P_ABBA.shape[0] - 1)

    for i in range(n_B - n_A + 1):
        P_ABBA[:j] = naive.aamp(T_A, m, T_B[i : i + n_A])[:, 0]
        P_ABBA[j:] = naive.aamp(T_B[i : i + n_A], m, T_A)[:, 0]
        P_ABBA.sort()
        ref_mpdist_vect[i] = P_ABBA[min(k, P_ABBA.shape[0] - 1)]

    comp_mpdist_vect = _aampdist_vect(T_A, T_B, m, percentage=percentage)

    npt.assert_almost_equal(ref_mpdist_vect, comp_mpdist_vect)


@pytest.mark.parametrize("T_A, T_B", test_data)
@pytest.mark.parametrize("k", k)
def test_aampdist_vect_k(T_A, T_B, k):
    m = 3
    n_A = T_A.shape[0]
    n_B = T_B.shape[0]
    j = n_A - m + 1  # `k` is reserved for `P_ABBA` selection
    P_ABBA = np.empty(2 * j, dtype=np.float64)
    ref_mpdist_vect = np.empty(n_B - n_A + 1)

    k = min(int(k), P_ABBA.shape[0] - 1)

    for i in range(n_B - n_A + 1):
        P_ABBA[:j] = naive.aamp(T_A, m, T_B[i : i + n_A])[:, 0]
        P_ABBA[j:] = naive.aamp(T_B[i : i + n_A], m, T_A)[:, 0]
        P_ABBA.sort()
        ref_mpdist_vect[i] = P_ABBA[min(k, P_ABBA.shape[0] - 1)]

    comp_mpdist_vect = _aampdist_vect(T_A, T_B, m, k=k)

    npt.assert_almost_equal(ref_mpdist_vect, comp_mpdist_vect)


@pytest.mark.parametrize("T_A, T_B", test_data)
def test_aampdist(T_A, T_B):
    m = 3
    ref_mpdist = naive.aampdist(T_A, T_B, m)
    comp_mpdist = aampdist(T_A, T_B, m)

    npt.assert_almost_equal(ref_mpdist, comp_mpdist)


@pytest.mark.parametrize("T_A, T_B", test_data)
@pytest.mark.parametrize("percentage", percentage)
def test_aampdist_percentage(T_A, T_B, percentage):
    m = 3
    ref_mpdist = naive.aampdist(T_A, T_B, m, percentage=percentage)
    comp_mpdist = aampdist(T_A, T_B, m, percentage=percentage)

    npt.assert_almost_equal(ref_mpdist, comp_mpdist)


@pytest.mark.parametrize("T_A, T_B", test_data)
@pytest.mark.parametrize("k", k)
def test_aampdist_k(T_A, T_B, k):
    m = 3
    ref_mpdist = naive.aampdist(T_A, T_B, m, k=k)
    comp_mpdist = aampdist(T_A, T_B, m, k=k)

    npt.assert_almost_equal(ref_mpdist, comp_mpdist)


@pytest.mark.filterwarnings("ignore:numpy.dtype size changed")
@pytest.mark.filterwarnings("ignore:numpy.ufunc size changed")
@pytest.mark.filterwarnings("ignore:numpy.ndarray size changed")
@pytest.mark.filterwarnings("ignore:\\s+Port 8787 is already in use:UserWarning")
@pytest.mark.parametrize("T_A, T_B", test_data)
def test_aampdisted(T_A, T_B, dask_cluster):
    with Client(dask_cluster) as dask_client:
        m = 3
        ref_mpdist = naive.aampdist(T_A, T_B, m)
        comp_mpdist = aampdisted(dask_client, T_A, T_B, m)

        npt.assert_almost_equal(ref_mpdist, comp_mpdist)
