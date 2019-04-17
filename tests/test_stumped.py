import numpy as np
import numpy.testing as npt
from stumpy import stumped, core
from dask.distributed import Client, LocalCluster
import pytest
import warnings

@pytest.fixture(scope='module')
def dask_client():
    cluster = LocalCluster(n_workers=None, threads_per_worker=2)
    client = Client(cluster)
    yield client
    # teardown
    client.close()
    cluster.close()

def naive_mass(Q, T, m, trivial_idx=None, excl_zone=0, ignore_trivial=False):
    D = np.linalg.norm(core.z_norm(core.rolling_window(T, m), 1) - core.z_norm(Q), axis=1)
    if ignore_trivial:
            start = max(0, trivial_idx - excl_zone)
            stop = min(T.shape[0]-Q.shape[0]+1, trivial_idx + excl_zone)
            D[start:stop] = np.inf
    I = np.argmin(D)
    P = D[I]

    # Get left and right matrix profiles for self-joins
    if ignore_trivial and trivial_idx > 0:
        PL = np.inf
        IL = -1
        for i in range(trivial_idx):
            if D[i] < PL:
                IL = i
                PL = D[i]
        if start <= IL <= stop:
            IL = -1
    else:
        IL = -1

    if ignore_trivial and trivial_idx+1 < D.shape[0]:
        PR = np.inf
        IR = -1
        for i in range(trivial_idx+1, D.shape[0]):
            if D[i] < PR:
                IR = i
                PR = D[i]
        if start <= IR <= stop:
            IR = -1
    else:
        IR = -1

    return P, I, IL, IR

def replace_inf(x, value=0):
    x[x == np.inf] = value
    x[x == -np.inf] = value
    return

test_data = [
    (np.array([9,8100,-60,7], dtype=np.float64), np.array([584,-11,23,79,1001,0,-19], dtype=np.float64)),
    (np.random.uniform(-1000, 1000, [8]).astype(np.float64), np.random.uniform(-1000, 1000, [64]).astype(np.float64))
]

@pytest.mark.filterwarnings("ignore:numpy.dtype size changed")
@pytest.mark.filterwarnings("ignore:numpy.ufunc size changed")
@pytest.mark.filterwarnings("ignore:numpy.ndarray size changed")
@pytest.mark.filterwarnings("ignore:\\s+Port 8787 is already in use:UserWarning")
@pytest.mark.parametrize("T_A, T_B", test_data)
def test_stumped_self_join(T_A, T_B, dask_client):
    m = 3
    zone = int(np.ceil(m/4))
    left = np.array([naive_mass(Q, T_B, m, i, zone, True) for i, Q in enumerate(core.rolling_window(T_B, m))], dtype=object)
    right = stumped(dask_client, T_B, m, ignore_trivial=True)
    replace_inf(left)
    replace_inf(right)
    npt.assert_almost_equal(left, right)

@pytest.mark.filterwarnings("ignore:numpy.dtype size changed")
@pytest.mark.filterwarnings("ignore:numpy.ufunc size changed")
@pytest.mark.filterwarnings("ignore:numpy.ndarray size changed")
@pytest.mark.filterwarnings("ignore:\\s+Port 8787 is already in use:UserWarning")
@pytest.mark.parametrize("T_A, T_B", test_data)
def test_stumped_A_B_join(T_A, T_B, dask_client):
    m = 3
    left = np.array([naive_mass(Q, T_A, m) for Q in core.rolling_window(T_B, m)], dtype=object)
    right = stumped(dask_client, T_A, m, T_B, ignore_trivial=False)
    replace_inf(left)
    replace_inf(right)
    npt.assert_almost_equal(left, right)
