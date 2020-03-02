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


def naive_mass(Q, T, m, trivial_idx, excl_zone):
    D = np.linalg.norm(
        utils.z_norm(core.rolling_window(T, m), 1) - utils.z_norm(Q), axis=1
    )
    start = max(0, trivial_idx - excl_zone)
    stop = min(T.shape[0] - Q.shape[0] + 1, trivial_idx + excl_zone)
    D[start : stop + 1] = np.inf

    return D


def naive_PI(D, trivial_idx):
    P = np.full((D.shape[0], D.shape[1]), np.inf)
    I = np.ones((D.shape[0], D.shape[1]), dtype="int64") * -1

    D = np.sort(D, axis=0)

    D_prime = np.zeros(D.shape[1])
    for i in range(D.shape[0]):
        D_prime = D_prime + D[i]
        D_prime_prime = D_prime / (i + 1)
        # Element-wise Min
        # col_idx = np.argmin([left_P[i, :], D_prime_prime], axis=0)
        # col_mask = col_idx > 0
        col_mask = P[i] > D_prime_prime
        P[i, col_mask] = D_prime_prime[col_mask]
        I[i, col_mask] = trivial_idx

    return P, I


def naive_mstump(T, m):
    zone = int(np.ceil(m / 4))
    Q = core.rolling_window(T, m)
    D = np.empty((Q.shape[0], Q.shape[1]))
    P = np.full((Q.shape[0], Q.shape[1]), np.inf)
    I = np.ones((Q.shape[0], Q.shape[1]), dtype="int64") * -1

    # Left
    for i in range(Q.shape[1]):
        D[:] = 0.0
        for dim in range(T.shape[0]):
            D[dim] = naive_mass(Q[dim, i], T[dim], m, i, zone)

        P_i, I_i = naive_PI(D, i)

        for dim in range(T.shape[0]):
            col_mask = P[dim] > P_i[dim]
            P[dim, col_mask] = P_i[dim, col_mask]
            I[dim, col_mask] = I_i[dim, col_mask]

    return P, I


test_data = [
    (np.array([[584, -11, 23, 79, 1001, 0, -19]], dtype=np.float64), 3),
    (np.random.uniform(-1000, 1000, [3, 10]).astype(np.float64), 5),
]


@pytest.mark.filterwarnings("ignore:\\s+Port 8787 is already in use:UserWarning")
@pytest.mark.parametrize("T, m", test_data)
def test_mstumped(T, m, dask_client):
    left_P, left_I = naive_mstump(T, m)
    right_P, right_I = mstumped(dask_client, T, m)

    npt.assert_almost_equal(left_P.T, right_P)
    npt.assert_almost_equal(left_I.T, right_I)


@pytest.mark.filterwarnings("ignore:\\s+Port 8787 is already in use:UserWarning")
@pytest.mark.parametrize("T, m", test_data)
def test_mstumped_df(T, m, dask_client):
    left_P, left_I = naive_mstump(T, m)
    df = pd.DataFrame(T.T)
    right_P, right_I = mstumped(dask_client, df, m)

    npt.assert_almost_equal(left_P.T, right_P)
    npt.assert_almost_equal(left_I.T, right_I)


@pytest.mark.filterwarnings("ignore:\\s+Port 8787 is already in use:UserWarning")
def test_constant_subsequence_self_join(dask_client):
    T_A = np.concatenate((np.zeros(20, dtype=np.float64), np.ones(5, dtype=np.float64)))
    T = np.array([T_A, T_A, np.random.rand(T_A.shape[0])])
    m = 3

    left_P, left_I = naive_mstump(T, m)
    right_P, right_I = mstumped(dask_client, T, m)

    npt.assert_almost_equal(left_P.T, right_P)  # ignore indices
