import numpy as np
import numpy.testing as npt
from stumpy import aamp_stimp, aamp_stimped

from dask.distributed import Client, LocalCluster
import pytest
import naive


T = [
    np.array([584, -11, 23, 79, 1001, 0, -19], dtype=np.float64),
    np.random.uniform(-1000, 1000, [64]).astype(np.float64),
]

n = [9, 10, 16]


@pytest.fixture(scope="module")
def dask_cluster():
    cluster = LocalCluster(n_workers=2, threads_per_worker=2)
    yield cluster
    cluster.close()


@pytest.mark.parametrize("T", T)
def test_aamp_stimp_1_percent(T):
    threshold = 0.2
    percentage = 0.01
    min_m = 3
    n = T.shape[0] - min_m + 1

    seed = np.random.randint(100000)

    np.random.seed(seed)
    pan = aamp_stimp(
        T,
        min_m=min_m,
        max_m=None,
        step=1,
        percentage=percentage,
        pre_scraamp=True,
    )

    for i in range(n):
        pan.update()

    ref_PAN = np.full((pan.M_.shape[0], T.shape[0]), fill_value=np.inf)

    np.random.seed(seed)
    for idx, m in enumerate(pan.M_[:n]):
        zone = int(np.ceil(m / 4))
        s = zone
        tmp_P, tmp_I = naive.prescraamp(T, m, T, s=s, exclusion_zone=zone)
        ref_mp = naive.scraamp(T, m, T, percentage, zone, True, s)
        for i in range(ref_mp.shape[0]):
            if tmp_P[i] < ref_mp[i, 0]:
                ref_mp[i, 0] = tmp_P[i]
                ref_mp[i, 1] = tmp_I[i]
        ref_PAN[pan._bfs_indices[idx], : ref_mp.shape[0]] = ref_mp[:, 0]

    # Compare raw pan
    cmp_PAN = pan._PAN

    naive.replace_inf(ref_PAN)
    naive.replace_inf(cmp_PAN)

    npt.assert_almost_equal(ref_PAN, cmp_PAN)

    # Compare transformed pan
    cmp_pan = pan.PAN_
    ref_pan = naive.transform_pan(
        pan._PAN,
        pan._M,
        threshold,
        pan._bfs_indices,
        pan._n_processed,
        np.min(T),
        np.max(T),
    )

    naive.replace_inf(ref_pan)
    naive.replace_inf(cmp_pan)

    npt.assert_almost_equal(ref_pan, cmp_pan)


@pytest.mark.parametrize("T", T)
def test_aamp_stimp_max_m(T):
    threshold = 0.2
    percentage = 0.01
    min_m = 3
    max_m = 5
    n = T.shape[0] - min_m + 1

    seed = np.random.randint(100000)

    np.random.seed(seed)
    pan = aamp_stimp(
        T,
        min_m=min_m,
        max_m=max_m,
        step=1,
        percentage=percentage,
        pre_scraamp=True,
    )

    for i in range(n):
        pan.update()

    ref_PAN = np.full((pan.M_.shape[0], T.shape[0]), fill_value=np.inf)

    np.random.seed(seed)
    for idx, m in enumerate(pan.M_[:n]):
        zone = int(np.ceil(m / 4))
        s = zone
        tmp_P, tmp_I = naive.prescraamp(T, m, T, s=s, exclusion_zone=zone)
        ref_mp = naive.scraamp(T, m, T, percentage, zone, True, s)
        for i in range(ref_mp.shape[0]):
            if tmp_P[i] < ref_mp[i, 0]:
                ref_mp[i, 0] = tmp_P[i]
                ref_mp[i, 1] = tmp_I[i]
        ref_PAN[pan._bfs_indices[idx], : ref_mp.shape[0]] = ref_mp[:, 0]

    # Compare raw pan
    cmp_PAN = pan._PAN

    naive.replace_inf(ref_PAN)
    naive.replace_inf(cmp_PAN)

    npt.assert_almost_equal(ref_PAN, cmp_PAN)

    # Compare transformed pan
    cmp_pan = pan.PAN_
    ref_pan = naive.transform_pan(
        pan._PAN,
        pan._M,
        threshold,
        pan._bfs_indices,
        pan._n_processed,
        np.min(T),
        np.max(T),
    )

    naive.replace_inf(ref_pan)
    naive.replace_inf(cmp_pan)

    npt.assert_almost_equal(ref_pan, cmp_pan)


@pytest.mark.parametrize("T", T)
def test_aamp_stimp_100_percent(T):
    threshold = 0.2
    percentage = 1.0
    min_m = 3
    n = T.shape[0] - min_m + 1

    pan = aamp_stimp(
        T,
        min_m=min_m,
        max_m=None,
        step=1,
        percentage=percentage,
        pre_scraamp=True,
    )

    for i in range(n):
        pan.update()

    ref_PAN = np.full((pan.M_.shape[0], T.shape[0]), fill_value=np.inf)

    for idx, m in enumerate(pan.M_[:n]):
        zone = int(np.ceil(m / 4))
        ref_mp = naive.aamp(T, m, T_B=None, exclusion_zone=zone)
        ref_PAN[pan._bfs_indices[idx], : ref_mp.shape[0]] = ref_mp[:, 0]

    # Compare raw pan
    cmp_PAN = pan._PAN

    naive.replace_inf(ref_PAN)
    naive.replace_inf(cmp_PAN)

    npt.assert_almost_equal(ref_PAN, cmp_PAN)

    # Compare transformed pan
    cmp_pan = pan.PAN_
    ref_pan = naive.transform_pan(
        pan._PAN,
        pan._M,
        threshold,
        pan._bfs_indices,
        pan._n_processed,
        np.min(T),
        np.max(T),
    )

    naive.replace_inf(ref_pan)
    naive.replace_inf(cmp_pan)

    npt.assert_almost_equal(ref_pan, cmp_pan)


@pytest.mark.filterwarnings("ignore:numpy.dtype size changed")
@pytest.mark.filterwarnings("ignore:numpy.ufunc size changed")
@pytest.mark.filterwarnings("ignore:numpy.ndarray size changed")
@pytest.mark.filterwarnings("ignore:\\s+Port 8787 is already in use:UserWarning")
@pytest.mark.parametrize("T", T)
def test_aamp_stimped(T, dask_cluster):
    with Client(dask_cluster) as dask_client:
        threshold = 0.2
        min_m = 3
        n = T.shape[0] - min_m + 1

        pan = aamp_stimped(
            dask_client,
            T,
            min_m=min_m,
            max_m=None,
            step=1,
        )

        for i in range(n):
            pan.update()

        ref_PAN = np.full((pan.M_.shape[0], T.shape[0]), fill_value=np.inf)

        for idx, m in enumerate(pan.M_[:n]):
            zone = int(np.ceil(m / 4))
            ref_mp = naive.aamp(T, m, T_B=None, exclusion_zone=zone)
            ref_PAN[pan._bfs_indices[idx], : ref_mp.shape[0]] = ref_mp[:, 0]

        # Compare raw pan
        cmp_PAN = pan._PAN

        naive.replace_inf(ref_PAN)
        naive.replace_inf(cmp_PAN)

        npt.assert_almost_equal(ref_PAN, cmp_PAN)

        # Compare transformed pan
        cmp_pan = pan.PAN_
        ref_pan = naive.transform_pan(
            pan._PAN,
            pan._M,
            threshold,
            pan._bfs_indices,
            pan._n_processed,
            np.min(T),
            np.max(T),
        )

        naive.replace_inf(ref_pan)
        naive.replace_inf(cmp_pan)

        npt.assert_almost_equal(ref_pan, cmp_pan)
