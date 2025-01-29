import functools

import naive
import numpy as np
import numpy.testing as npt
import pytest

from stumpy import aamp, core, floss, fluss, stump
from stumpy.floss import _cac, _iac, _nnmark, _rea


def naive_nnmark(I):
    nnmark = np.zeros(I.shape[0], dtype=np.int64)
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


def naive_right_mp(T, m, normalize=True, p=2.0, T_subseq_isconstant=None):
    # computing `T_subseq_isconstant` boolean array
    T_subseq_isconstant = naive.rolling_isconstant(T, m, T_subseq_isconstant)

    if normalize:
        mp = stump(T_A=T, m=m, T_A_subseq_isconstant=T_subseq_isconstant)
    else:
        mp = aamp(T, m, p=p)
    k = mp.shape[0]
    right_nn = np.zeros((k, m))
    right_indices = [np.arange(IR, IR + m) for IR in mp[:, 3].tolist()]
    right_nn[:] = T[np.array(right_indices)]
    if normalize:
        mp[:, 0] = np.linalg.norm(
            core.z_norm(core.rolling_window(T, m), 1) - core.z_norm(right_nn, 1),
            axis=1,
        )
        for i, nn_i in enumerate(mp[:, 3]):
            if T_subseq_isconstant[i] and T_subseq_isconstant[nn_i]:
                mp[i, 0] = 0
            elif T_subseq_isconstant[i] or T_subseq_isconstant[nn_i]:
                mp[i, 0] = np.sqrt(m)
            else:  # pragma: no cover
                pass
    else:
        mp[:, 0] = np.linalg.norm(core.rolling_window(T, m) - right_nn, axis=1, ord=p)
    inf_indices = np.argwhere(mp[:, 3] < 0).flatten()
    mp[inf_indices, 0] = np.inf
    mp[inf_indices, 3] = inf_indices

    return mp


def naive_rea(cac, n_regimes, L, excl_factor):
    cac_list = cac.tolist()
    loc_regimes = [None] * (n_regimes - 1)
    for i in range(n_regimes - 1):
        loc_regimes[i] = cac_list.index(min(cac_list))
        excl_start = max(loc_regimes[i] - L * excl_factor, 0)
        excl_stop = min(loc_regimes[i] + L * excl_factor, len(cac_list))
        for excl in range(excl_start, excl_stop):
            cac_list[excl] = 1.0

    return np.array(loc_regimes, dtype=np.int64)


test_data = [np.random.randint(0, 50, size=50, dtype=np.int64)]

substitution_locations = [(slice(0, 0), 0, -1, slice(1, 3), [0, 3])]
substitution_values = [np.nan, np.inf]


@pytest.mark.parametrize("I", test_data)
def test_nnmark(I):
    ref = naive_nnmark(I)
    comp = _nnmark(I)
    npt.assert_almost_equal(ref, comp)


@pytest.mark.parametrize("I", test_data)
def test_cac(I):
    L = 5
    excl_factor = 1
    custom_iac = _iac(I.shape[0])
    ref = naive_cac(I, L, excl_factor, custom_iac)
    bidirectional = True
    comp = _cac(I, L, bidirectional, excl_factor)
    npt.assert_almost_equal(ref, comp)


@pytest.mark.parametrize("I", test_data)
def test_cac_custom_iac(I):
    L = 5
    excl_factor = 1
    ref = naive_cac(I, L, excl_factor)
    custom_iac = naive_iac(I.shape[0])
    bidirectional = True
    comp = _cac(I, L, bidirectional, excl_factor, custom_iac)
    npt.assert_almost_equal(ref, comp)


@pytest.mark.parametrize("I", test_data)
def test_rea(I):
    L = 5
    excl_factor = 1
    cac = naive_cac(I, L, excl_factor)
    n_regimes = 3
    ref = naive_rea(cac, n_regimes, L, excl_factor)
    comp = _rea(cac, n_regimes, L, excl_factor)
    npt.assert_almost_equal(ref, comp)


@pytest.mark.parametrize("I", test_data)
def test_fluss(I):
    L = 5
    excl_factor = 1
    custom_iac = naive_iac(I.shape[0])
    ref_cac = naive_cac(I, L, excl_factor)
    n_regimes = 3
    ref_rea = naive_rea(ref_cac, n_regimes, L, excl_factor)
    comp_cac, comp_rea = fluss(I, L, n_regimes, excl_factor, custom_iac)
    npt.assert_almost_equal(ref_cac, comp_cac)
    npt.assert_almost_equal(ref_rea, comp_rea)


def test_floss():
    data = np.random.uniform(-1000, 1000, [64])
    m = 5
    n = 30
    old_data = data[:n]

    mp = naive_right_mp(old_data, m)
    comp_mp = stump(old_data, m)
    k = mp.shape[0]

    rolling_Ts = core.rolling_window(data[1:], n)
    L = 5
    excl_factor = 1
    custom_iac = _iac(k, bidirectional=False)
    stream = floss(comp_mp, old_data, m, L, excl_factor, custom_iac=custom_iac)
    last_idx = n - m + 1
    excl_zone = int(np.ceil(m / 4))
    zone_start = max(0, k - excl_zone)
    for i, ref_T in enumerate(rolling_Ts):
        mp[:, 1] = -1
        mp[:, 2] = -1
        mp[:] = np.roll(mp, -1, axis=0)
        mp[-1, 0] = np.inf
        mp[-1, 3] = last_idx + i

        D = naive.distance_profile(ref_T[-m:], ref_T, m)
        D[zone_start:] = np.inf

        update_idx = np.argwhere(D < mp[:, 0]).flatten()
        mp[update_idx, 0] = D[update_idx]
        mp[update_idx, 3] = last_idx + i

        ref_cac_1d = _cac(
            mp[:, 3] - i - 1,
            L,
            bidirectional=False,
            excl_factor=excl_factor,
            custom_iac=custom_iac,
        )

        ref_mp = mp.copy()
        ref_P = ref_mp[:, 0]
        ref_I = ref_mp[:, 3]
        ref_I[ref_mp[:, 0] == np.inf] = -1

        stream.update(ref_T[-1])
        comp_cac_1d = stream.cac_1d_
        comp_P = stream.P_

        comp_I = stream.I_
        comp_T = stream.T_

        naive.replace_inf(ref_P)
        naive.replace_inf(comp_P)

        npt.assert_almost_equal(ref_cac_1d, comp_cac_1d)
        npt.assert_almost_equal(ref_P, comp_P)
        npt.assert_almost_equal(ref_I, comp_I)
        npt.assert_almost_equal(ref_T, comp_T)


def test_aamp_floss():
    data = np.random.uniform(-1000, 1000, [64])
    m = 5
    n = 30
    old_data = data[:n]

    for p in range(1, 4):
        mp = naive_right_mp(old_data, m, normalize=False, p=p)
        comp_mp = aamp(old_data, m, p=p)
        k = mp.shape[0]

        rolling_Ts = core.rolling_window(data[1:], n)
        L = 5
        excl_factor = 1
        custom_iac = _iac(k, bidirectional=False)
        stream = floss(
            comp_mp,
            old_data,
            m,
            L,
            excl_factor,
            custom_iac=custom_iac,
            normalize=False,
            p=p,
        )
        last_idx = n - m + 1
        excl_zone = int(np.ceil(m / 4))
        zone_start = max(0, k - excl_zone)
        for i, ref_T in enumerate(rolling_Ts):
            mp[:, 1] = -1
            mp[:, 2] = -1
            mp[:] = np.roll(mp, -1, axis=0)
            mp[-1, 0] = np.inf
            mp[-1, 3] = last_idx + i

            D = naive.aamp_distance_profile(ref_T[-m:], ref_T, m, p=p)
            D[zone_start:] = np.inf

            update_idx = np.argwhere(D < mp[:, 0]).flatten()
            mp[update_idx, 0] = D[update_idx]
            mp[update_idx, 3] = last_idx + i

            ref_cac_1d = _cac(
                mp[:, 3] - i - 1,
                L,
                bidirectional=False,
                excl_factor=excl_factor,
                custom_iac=custom_iac,
            )

            ref_mp = mp.copy()
            ref_P = ref_mp[:, 0]
            ref_I = ref_mp[:, 3]
            ref_I[ref_mp[:, 0] == np.inf] = -1

            stream.update(ref_T[-1])
            comp_cac_1d = stream.cac_1d_
            comp_P = stream.P_
            comp_I = stream.I_
            comp_T = stream.T_

            naive.replace_inf(ref_P)
            naive.replace_inf(comp_P)

            npt.assert_almost_equal(ref_cac_1d, comp_cac_1d)
            npt.assert_almost_equal(ref_P, comp_P)
            npt.assert_almost_equal(ref_I, comp_I)
            npt.assert_almost_equal(ref_T, comp_T)


@pytest.mark.parametrize("substitute", substitution_values)
@pytest.mark.parametrize("substitution_locations", substitution_locations)
def test_floss_inf_nan(substitute, substitution_locations):
    T = np.random.uniform(-1000, 1000, [64])
    m = 5
    n = 30
    data = T.copy()
    for substitution_location in substitution_locations:
        data[:] = T[:]
        data[substitution_location] = substitute
        old_data = data[:n]

        mp = naive_right_mp(old_data, m)
        comp_mp = stump(old_data, m)
        k = mp.shape[0]

        rolling_Ts = core.rolling_window(data[1:], n)
        L = 5
        excl_factor = 1
        custom_iac = _iac(k, bidirectional=False)
        stream = floss(comp_mp, old_data, m, L, excl_factor, custom_iac=custom_iac)
        last_idx = n - m + 1
        excl_zone = int(np.ceil(m / 4))
        zone_start = max(0, k - excl_zone)
        for i, ref_T in enumerate(rolling_Ts):
            mp[:, 1] = -1
            mp[:, 2] = -1
            mp[:] = np.roll(mp, -1, axis=0)
            mp[-1, 0] = np.inf
            mp[-1, 3] = last_idx + i

            D = naive.distance_profile(ref_T[-m:], ref_T, m)
            D[zone_start:] = np.inf

            ref_T_isfinite = np.isfinite(ref_T)
            ref_T_subseq_isfinite = np.all(
                core.rolling_window(ref_T_isfinite, m), axis=1
            )

            D[~ref_T_subseq_isfinite] = np.inf
            update_idx = np.argwhere(D < mp[:, 0]).flatten()
            mp[update_idx, 0] = D[update_idx]
            mp[update_idx, 3] = last_idx + i

            ref_cac_1d = _cac(
                mp[:, 3] - i - 1,
                L,
                bidirectional=False,
                excl_factor=excl_factor,
                custom_iac=custom_iac,
            )

            ref_mp = mp.copy()
            ref_P = ref_mp[:, 0]
            ref_I = ref_mp[:, 3]
            ref_I[ref_mp[:, 0] == np.inf] = -1

            stream.update(ref_T[-1])
            comp_cac_1d = stream.cac_1d_
            comp_P = stream.P_
            comp_I = stream.I_
            comp_T = stream.T_

            naive.replace_inf(ref_P)
            naive.replace_inf(comp_P)

            npt.assert_almost_equal(ref_cac_1d, comp_cac_1d)
            npt.assert_almost_equal(ref_P, comp_P)
            npt.assert_almost_equal(ref_I, comp_I)
            npt.assert_almost_equal(ref_T, comp_T)


@pytest.mark.parametrize("substitute", substitution_values)
@pytest.mark.parametrize("substitution_locations", substitution_locations)
def test_aamp_floss_inf_nan(substitute, substitution_locations):
    T = np.random.uniform(-1000, 1000, [64])
    m = 5
    n = 30
    data = T.copy()
    for substitution_location in substitution_locations:
        data[:] = T[:]
        data[substitution_location] = substitute
        old_data = data[:n]

        mp = naive_right_mp(old_data, m, normalize=False)
        comp_mp = aamp(old_data, m)
        k = mp.shape[0]

        rolling_Ts = core.rolling_window(data[1:], n)
        L = 5
        excl_factor = 1
        custom_iac = _iac(k, bidirectional=False)
        stream = floss(
            comp_mp, old_data, m, L, excl_factor, custom_iac=custom_iac, normalize=False
        )
        last_idx = n - m + 1
        excl_zone = int(np.ceil(m / 4))
        zone_start = max(0, k - excl_zone)
        for i, ref_T in enumerate(rolling_Ts):
            mp[:, 1] = -1
            mp[:, 2] = -1
            mp[:] = np.roll(mp, -1, axis=0)
            mp[-1, 0] = np.inf
            mp[-1, 3] = last_idx + i

            D = naive.aamp_distance_profile(ref_T[-m:], ref_T, m)
            D[zone_start:] = np.inf

            ref_T_isfinite = np.isfinite(ref_T)
            ref_T_subseq_isfinite = np.all(
                core.rolling_window(ref_T_isfinite, m), axis=1
            )

            D[~ref_T_subseq_isfinite] = np.inf
            update_idx = np.argwhere(D < mp[:, 0]).flatten()
            mp[update_idx, 0] = D[update_idx]
            mp[update_idx, 3] = last_idx + i

            ref_cac_1d = _cac(
                mp[:, 3] - i - 1,
                L,
                bidirectional=False,
                excl_factor=excl_factor,
                custom_iac=custom_iac,
            )

            ref_mp = mp.copy()
            ref_P = ref_mp[:, 0]
            ref_I = ref_mp[:, 3]
            ref_I[ref_mp[:, 0] == np.inf] = -1

            stream.update(ref_T[-1])
            comp_cac_1d = stream.cac_1d_
            comp_P = stream.P_
            comp_I = stream.I_
            comp_T = stream.T_

            naive.replace_inf(ref_P)
            naive.replace_inf(comp_P)

            npt.assert_almost_equal(ref_cac_1d, comp_cac_1d)
            npt.assert_almost_equal(ref_P, comp_P)
            npt.assert_almost_equal(ref_I, comp_I)
            npt.assert_almost_equal(ref_T, comp_T)


def test_floss_with_isconstant():
    data = np.random.uniform(-1, 1, [64])
    m = 5
    n = 30
    old_data = data[:n]

    quantile_threshold = 0.5
    sliding_stddev = naive.rolling_nanstd(old_data, m)
    stddev_threshold = np.quantile(sliding_stddev, quantile_threshold)
    isconstant_custom_func = functools.partial(
        naive.isconstant_func_stddev_threshold,
        stddev_threshold=stddev_threshold,
    )

    mp = naive_right_mp(T=old_data, m=m, T_subseq_isconstant=isconstant_custom_func)
    comp_mp = stump(T_A=old_data, m=m, T_A_subseq_isconstant=isconstant_custom_func)
    k = mp.shape[0]

    rolling_Ts = core.rolling_window(data[1:], n)
    L = 5
    excl_factor = 1
    custom_iac = _iac(k, bidirectional=False)
    stream = floss(
        comp_mp,
        old_data,
        m,
        L,
        excl_factor,
        custom_iac=custom_iac,
        T_subseq_isconstant_func=isconstant_custom_func,
    )
    last_idx = n - m + 1
    excl_zone = int(np.ceil(m / 4))
    zone_start = max(0, k - excl_zone)
    for i, ref_T in enumerate(rolling_Ts):
        mp[:, 1] = -1
        mp[:, 2] = -1
        mp[:] = np.roll(mp, -1, axis=0)
        mp[-1, 0] = np.inf
        mp[-1, 3] = last_idx + i

        ref_Q = ref_T[-m:]
        ref_Q_isconstant = isconstant_custom_func(ref_Q, m)[0]
        ref_T_subseq_isconstant = isconstant_custom_func(ref_T, m)
        D = naive.distance_profile(ref_Q, ref_T, m)
        for j in range(len(D)):
            if ref_Q_isconstant and ref_T_subseq_isconstant[j]:
                D[j] = 0
            elif ref_Q_isconstant or ref_T_subseq_isconstant[j]:
                D[j] = np.sqrt(m)
            else:
                pass
        D[zone_start:] = np.inf

        update_idx = np.argwhere(D < mp[:, 0]).flatten()
        mp[update_idx, 0] = D[update_idx]
        mp[update_idx, 3] = last_idx + i

        ref_cac_1d = _cac(
            mp[:, 3] - i - 1,
            L,
            bidirectional=False,
            excl_factor=excl_factor,
            custom_iac=custom_iac,
        )

        ref_mp = mp.copy()
        ref_P = ref_mp[:, 0]
        ref_I = ref_mp[:, 3]
        ref_I[ref_mp[:, 0] == np.inf] = -1

        stream.update(ref_T[-1])
        comp_cac_1d = stream.cac_1d_
        comp_P = stream.P_
        comp_I = stream.I_
        comp_T = stream.T_

        naive.replace_inf(ref_P)
        naive.replace_inf(comp_P)

        npt.assert_almost_equal(ref_cac_1d, comp_cac_1d)
        npt.assert_almost_equal(ref_P, comp_P)
        npt.assert_almost_equal(ref_I, comp_I)
        npt.assert_almost_equal(ref_T, comp_T)
