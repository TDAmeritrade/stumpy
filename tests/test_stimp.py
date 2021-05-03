import numpy as np
import numpy.testing as npt
from stumpy import stimp
from stumpy.stimp import _bfs_indices
import pytest
import naive

T = [
    np.array([584, -11, 23, 79, 1001, 0, -19], dtype=np.float64),
    np.random.uniform(-1000, 1000, [64]).astype(np.float64),
]

n = [9, 10, 16]


def split(node, out):
    mid = len(node) // 2
    out.append(node[mid])
    return node[:mid], node[mid + 1 :]


def naive_bsf_indices(n):
    a = np.arange(n)
    nodes = [a.tolist()]
    out = []

    while nodes:
        tmp = []
        for node in nodes:
            for n in split(node, out):
                if n:
                    tmp.append(n)
        nodes = tmp

    return np.array(out)


@pytest.mark.parametrize("n", n)
def test_bsf_indices(n):
    ref_bsf_indices = naive_bsf_indices(n)
    cmp_bsf_indices = np.array(list(_bfs_indices(n)))

    npt.assert_almost_equal(ref_bsf_indices, cmp_bsf_indices)


@pytest.mark.parametrize("T", T)
def test_stimp(T):
    percentage = 0.01
    min_m = 3
    n = T.shape[0] - min_m + 1

    seed = np.random.randint(100000)

    np.random.seed(seed)
    pmp = stimp(
        T,
        min_m=min_m,
        max_m=None,
        step=1,
        percentage=percentage,
        pre_scrump=True,
        normalize=True,
    )

    for i in range(n):
        pmp.update()

    ref_P = np.full((pmp.M_.shape[0], T.shape[0]), fill_value=np.inf)
    ref_I = np.ones((pmp.M_.shape[0], T.shape[0]), dtype=np.int64) * -1

    np.random.seed(seed)
    for idx, m in enumerate(pmp.M_[:n]):
        zone = int(np.ceil(m / 4))
        s = zone
        tmp_P, tmp_I = naive.prescrump(T, m, T, s=s, exclusion_zone=zone)
        ref_mp = naive.scrump(T, m, T, percentage, zone, True, s)
        for i in range(ref_mp.shape[0]):
            if tmp_P[i] < ref_mp[i, 0]:
                ref_mp[i, 0] = tmp_P[i]
                ref_mp[i, 1] = tmp_I[i]
        ref_P[pmp.bfs_indices_[idx], : ref_mp.shape[0]] = ref_mp[:, 0]
        ref_I[pmp.bfs_indices_[idx], : ref_mp.shape[0]] = ref_mp[:, 1]

    comp_P = pmp.P_
    comp_I = pmp.I_

    naive.replace_inf(ref_P)
    naive.replace_inf(ref_I)
    naive.replace_inf(comp_P)
    naive.replace_inf(comp_I)

    npt.assert_almost_equal(ref_P, comp_P)
    npt.assert_almost_equal(ref_I, comp_I)


@pytest.mark.parametrize("T", T)
def test_stimp_max_m(T):
    percentage = 0.01
    min_m = 3
    max_m = 5
    n = T.shape[0] - min_m + 1

    seed = np.random.randint(100000)

    np.random.seed(seed)
    pmp = stimp(
        T,
        min_m=min_m,
        max_m=max_m,
        step=1,
        percentage=percentage,
        pre_scrump=True,
        normalize=True,
    )

    for i in range(n):
        pmp.update()

    ref_P = np.full((pmp.M_.shape[0], T.shape[0]), fill_value=np.inf)
    ref_I = np.ones((pmp.M_.shape[0], T.shape[0]), dtype=np.int64) * -1

    np.random.seed(seed)
    for idx, m in enumerate(pmp.M_[:n]):
        zone = int(np.ceil(m / 4))
        s = zone
        tmp_P, tmp_I = naive.prescrump(T, m, T, s=s, exclusion_zone=zone)
        ref_mp = naive.scrump(T, m, T, percentage, zone, True, s)
        for i in range(ref_mp.shape[0]):
            if tmp_P[i] < ref_mp[i, 0]:
                ref_mp[i, 0] = tmp_P[i]
                ref_mp[i, 1] = tmp_I[i]
        ref_P[pmp.bfs_indices_[idx], : ref_mp.shape[0]] = ref_mp[:, 0]
        ref_I[pmp.bfs_indices_[idx], : ref_mp.shape[0]] = ref_mp[:, 1]

    comp_P = pmp.P_
    comp_I = pmp.I_

    naive.replace_inf(ref_P)
    naive.replace_inf(ref_I)
    naive.replace_inf(comp_P)
    naive.replace_inf(comp_I)

    npt.assert_almost_equal(ref_P, comp_P)
    npt.assert_almost_equal(ref_I, comp_I)


@pytest.mark.parametrize("T", T)
def test_stimp_100_percent(T):
    percentage = 1.0
    min_m = 3
    n = T.shape[0] - min_m + 1

    seed = np.random.randint(100000)

    np.random.seed(seed)
    pmp = stimp(
        T,
        min_m=min_m,
        max_m=None,
        step=1,
        percentage=percentage,
        pre_scrump=True,
        normalize=True,
    )

    for i in range(n):
        pmp.update()

    ref_P = np.full((pmp.M_.shape[0], T.shape[0]), fill_value=np.inf)
    ref_I = np.ones((pmp.M_.shape[0], T.shape[0]), dtype=np.int64) * -1

    np.random.seed(seed)
    for idx, m in enumerate(pmp.M_[:n]):
        zone = int(np.ceil(m / 4))
        ref_mp = naive.stump(T, m, T_B=None, exclusion_zone=zone)
        ref_P[pmp.bfs_indices_[idx], : ref_mp.shape[0]] = ref_mp[:, 0]
        ref_I[pmp.bfs_indices_[idx], : ref_mp.shape[0]] = ref_mp[:, 1]

    comp_P = pmp.P_
    comp_I = pmp.I_

    naive.replace_inf(ref_P)
    naive.replace_inf(ref_I)
    naive.replace_inf(comp_P)
    naive.replace_inf(comp_I)

    npt.assert_almost_equal(ref_P, comp_P)
    npt.assert_almost_equal(ref_I, comp_I)
