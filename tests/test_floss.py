import numpy as np
import numpy.testing as npt
from stumpy import _nnmark, _iac, _cac, _rea, fluss, stump, core, floss
import copy
import pytest


def naive_nnmark(I):
    nnmark = np.zeros(I.shape[0], dtype=np.int)
    for i in range(nnmark.shape[0]):
        j = I[i]
        nnmark[min(i, j)] = nnmark[min(i, j)] + 1
        nnmark[max(i, j)] = nnmark[max(i, j)] - 1

    return np.cumsum(nnmark)


def naive_iac(width):
    height = width / 2
    a = height / ((width / 2) * (width / 2))
    b = height
    c = width / 2
    x = np.arange(width)
    y = -(a * (x - c) * (x - c)) + b

    return y


def naive_cac(I, L, excl_factor, custom_iac=None):
    n = I.shape[0]
    AC = np.zeros(n)
    CAC = np.zeros(n)
    AC = naive_nnmark(I)
    if custom_iac is not None:
        IAC = custom_iac
    else:
        IAC = naive_iac(n)
    CAC = np.minimum(AC / IAC, 1.0)
    CAC[: L * excl_factor] = 1.0
    CAC[-L * excl_factor :] = 1.0

    return CAC


def naive_right_mp(data, m):
    mp = stump(data, m)
    k = mp.shape[0]
    right_nn = np.zeros((k, m))
    right_indices = [np.arange(IR, IR + m) for IR in mp[:, 3].tolist()]
    right_nn[:] = data[np.array(right_indices)]
    mp[:, 0] = np.linalg.norm(
        core.z_norm(core.rolling_window(data, m), 1) - core.z_norm(right_nn, 1), axis=1
    )
    inf_indices = np.argwhere(mp[:, 3] < 0).flatten()
    mp[inf_indices, 0] = np.inf
    mp[inf_indices, 3] = inf_indices

    return mp


def naive_distance_profile(Q, T, m):
    D = np.linalg.norm(
        core.z_norm(core.rolling_window(T, m), 1) - core.z_norm(Q), axis=1
    )
    return D


def naive_rea(cac, n_regimes, L, excl_factor):
    cac_list = cac.tolist()
    loc_regimes = [None] * (n_regimes - 1)
    for i in range(n_regimes - 1):
        loc_regimes[i] = cac_list.index(min(cac_list))
        excl_start = max(loc_regimes[i] - L * excl_factor, 0)
        excl_stop = min(loc_regimes[i] + L * excl_factor, len(cac_list))
        for excl in range(excl_start, excl_stop):
            cac_list[excl] = 1.0

    return np.array(loc_regimes, dtype=np.int)


test_data = [(np.random.randint(0, 50, size=50, dtype=np.int))]


@pytest.mark.parametrize("I", test_data)
def test_nnmark(I):
    left = naive_nnmark(I)
    right = _nnmark(I)
    npt.assert_almost_equal(left, right)


@pytest.mark.parametrize("I", test_data)
def test_cac(I):
    L = 5
    excl_factor = 1
    custom_iac = _iac(I.shape[0])
    left = naive_cac(I, L, excl_factor, custom_iac)
    bidirectional = True
    right = _cac(I, L, bidirectional, excl_factor)
    npt.assert_almost_equal(left, right)


@pytest.mark.parametrize("I", test_data)
def test_cac_custom_iac(I):
    L = 5
    excl_factor = 1
    left = naive_cac(I, L, excl_factor)
    custom_iac = naive_iac(I.shape[0])
    bidirectional = True
    right = _cac(I, L, bidirectional, excl_factor, custom_iac)
    npt.assert_almost_equal(left, right)


@pytest.mark.parametrize("I", test_data)
def test_rea(I):
    L = 5
    excl_factor = 1
    cac = naive_cac(I, L, excl_factor)
    n_regimes = 3
    left = naive_rea(cac, n_regimes, L, excl_factor)
    right = _rea(cac, n_regimes, L, excl_factor)
    npt.assert_almost_equal(left, right)


@pytest.mark.parametrize("I", test_data)
def test_fluss(I):
    L = 5
    excl_factor = 1
    custom_iac = naive_iac(I.shape[0])
    left_cac = naive_cac(I, L, excl_factor)
    n_regimes = 3
    left_rea = naive_rea(left_cac, n_regimes, L, excl_factor)
    right_cac, right_rea = fluss(I, L, n_regimes, excl_factor, custom_iac)
    npt.assert_almost_equal(left_cac, right_cac)
    npt.assert_almost_equal(left_rea, right_rea)


def test_floss():
    data = np.random.uniform(-1000, 1000, [64])
    m = 5
    old_data = data[:30]
    n = old_data.shape[0]
    add_data = data[30:]

    left_mp = naive_right_mp(old_data, m)
    right_mp = stump(old_data, m)
    k = left_mp.shape[0]

    rolling_Ts = core.rolling_window(data[1:], n)
    L = 5
    excl_factor = 1
    custom_iac = _iac(k, bidirectional=False)
    right_gen = floss(
        right_mp, old_data, add_data, m, L, excl_factor, custom_iac=custom_iac
    )
    last_idx = n - m + 1
    excl_zone = int(np.ceil(m / 4))
    zone_start = max(0, k - excl_zone)
    for i, T in enumerate(rolling_Ts):
        left_mp[:] = np.roll(left_mp, -1, axis=0)
        left_mp[-1, 0] = np.inf
        left_mp[-1, 3] = last_idx + i

        D = naive_distance_profile(T[-m:], T, m)
        D[zone_start:] = np.inf

        update_idx = np.argwhere(D < left_mp[:, 0]).flatten()
        left_mp[update_idx, 0] = D[update_idx]
        left_mp[update_idx, 3] = last_idx + i

        left_cac = _cac(
            left_mp[:, 3] - i - 1,
            L,
            bidirectional=False,
            excl_factor=excl_factor,
            custom_iac=custom_iac,
        )
        right_cac, right_mp, tmp_T = next(right_gen)
        npt.assert_almost_equal(left_cac, right_cac)
