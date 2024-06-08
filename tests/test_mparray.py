import naive
import numpy as np
import numpy.testing as npt
import pandas as pd
import pytest

from stumpy import aamp, config, stump
from stumpy.mparray import mparray

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

kNN = [1, 2, 3, 4]


@pytest.mark.parametrize("T_A, T_B", test_data)
def test_mparray_init(T_A, T_B):
    # Test different `mparray` initialization approaches
    m = 3
    k = 2
    arr = stump(T_B, m, ignore_trivial=True, k=k)
    mp = mparray(arr, m, k, config.STUMPY_EXCL_ZONE_DENOM)
    assert mp._m == m
    assert mp._k == k
    assert mp._excl_zone_denom == config.STUMPY_EXCL_ZONE_DENOM

    slice_mp = mp[1:, :]  # Initialize "new-from-template"
    assert slice_mp._m == m
    assert slice_mp._k == k
    assert mp._excl_zone_denom == config.STUMPY_EXCL_ZONE_DENOM


@pytest.mark.parametrize("T_A, T_B", test_data)
@pytest.mark.parametrize("k", kNN)
def test_mparray_self_join(T_A, T_B, k):
    m = 3
    zone = int(np.ceil(m / 4))

    ref_mp = naive.stump(T_B, m, exclusion_zone=zone, k=k)
    comp_mp = stump(T_B, m, ignore_trivial=True, k=k)
    naive.replace_inf(ref_mp)
    naive.replace_inf(comp_mp)
    npt.assert_almost_equal(np.squeeze(ref_mp[:, :k]), comp_mp.P_)
    npt.assert_almost_equal(np.squeeze(ref_mp[:, k : 2 * k]), comp_mp.I_)
    npt.assert_almost_equal(np.squeeze(ref_mp[:, 2 * k]), comp_mp.left_I_)
    npt.assert_almost_equal(np.squeeze(ref_mp[:, 2 * k + 1]), comp_mp.right_I_)

    comp_mp = stump(pd.Series(T_B), m, ignore_trivial=True, k=k)
    naive.replace_inf(comp_mp)
    npt.assert_almost_equal(np.squeeze(ref_mp[:, :k]), comp_mp.P_)
    npt.assert_almost_equal(np.squeeze(ref_mp[:, k : 2 * k]), comp_mp.I_)
    npt.assert_almost_equal(np.squeeze(ref_mp[:, 2 * k]), comp_mp.left_I_)
    npt.assert_almost_equal(np.squeeze(ref_mp[:, 2 * k + 1]), comp_mp.right_I_)

    ref_mp = naive.aamp(T_B, m, exclusion_zone=zone, k=k)
    comp_mp = aamp(T_B, m, ignore_trivial=True, k=k)
    naive.replace_inf(ref_mp)
    naive.replace_inf(comp_mp)
    npt.assert_almost_equal(np.squeeze(ref_mp[:, :k]), comp_mp.P_)
    npt.assert_almost_equal(np.squeeze(ref_mp[:, k : 2 * k]), comp_mp.I_)
    npt.assert_almost_equal(np.squeeze(ref_mp[:, 2 * k]), comp_mp.left_I_)
    npt.assert_almost_equal(np.squeeze(ref_mp[:, 2 * k + 1]), comp_mp.right_I_)

    comp_mp = aamp(pd.Series(T_B), m, ignore_trivial=True, k=k)
    naive.replace_inf(comp_mp)
    npt.assert_almost_equal(np.squeeze(ref_mp[:, :k]), comp_mp.P_)
    npt.assert_almost_equal(np.squeeze(ref_mp[:, k : 2 * k]), comp_mp.I_)
    npt.assert_almost_equal(np.squeeze(ref_mp[:, 2 * k]), comp_mp.left_I_)
    npt.assert_almost_equal(np.squeeze(ref_mp[:, 2 * k + 1]), comp_mp.right_I_)


@pytest.mark.parametrize("T_A, T_B", test_data)
@pytest.mark.parametrize("k", kNN)
def test_mparray_A_B_join(T_A, T_B, k):
    m = 3
    ref_mp = naive.stump(T_A, m, T_B=T_B, k=k)
    comp_mp = stump(T_A, m, T_B, ignore_trivial=False, k=k)
    naive.replace_inf(ref_mp)
    naive.replace_inf(comp_mp)
    npt.assert_almost_equal(np.squeeze(ref_mp[:, :k]), comp_mp.P_)
    npt.assert_almost_equal(np.squeeze(ref_mp[:, k : 2 * k]), comp_mp.I_)
    npt.assert_almost_equal(np.squeeze(ref_mp[:, 2 * k]), comp_mp.left_I_)
    npt.assert_almost_equal(np.squeeze(ref_mp[:, 2 * k + 1]), comp_mp.right_I_)

    comp_mp = stump(pd.Series(T_A), m, pd.Series(T_B), ignore_trivial=False, k=k)
    naive.replace_inf(comp_mp)
    npt.assert_almost_equal(np.squeeze(ref_mp[:, :k]), comp_mp.P_)
    npt.assert_almost_equal(np.squeeze(ref_mp[:, k : 2 * k]), comp_mp.I_)
    npt.assert_almost_equal(np.squeeze(ref_mp[:, 2 * k]), comp_mp.left_I_)
    npt.assert_almost_equal(np.squeeze(ref_mp[:, 2 * k + 1]), comp_mp.right_I_)

    ref_mp = naive.aamp(T_A, m, T_B=T_B, k=k)
    comp_mp = aamp(T_A, m, T_B, ignore_trivial=False, k=k)
    naive.replace_inf(ref_mp)
    naive.replace_inf(comp_mp)
    npt.assert_almost_equal(np.squeeze(ref_mp[:, :k]), comp_mp.P_)
    npt.assert_almost_equal(np.squeeze(ref_mp[:, k : 2 * k]), comp_mp.I_)
    npt.assert_almost_equal(np.squeeze(ref_mp[:, 2 * k]), comp_mp.left_I_)
    npt.assert_almost_equal(np.squeeze(ref_mp[:, 2 * k + 1]), comp_mp.right_I_)

    comp_mp = aamp(pd.Series(T_A), m, pd.Series(T_B), ignore_trivial=False, k=k)
    naive.replace_inf(comp_mp)
    npt.assert_almost_equal(np.squeeze(ref_mp[:, :k]), comp_mp.P_)
    npt.assert_almost_equal(np.squeeze(ref_mp[:, k : 2 * k]), comp_mp.I_)
    npt.assert_almost_equal(np.squeeze(ref_mp[:, 2 * k]), comp_mp.left_I_)
    npt.assert_almost_equal(np.squeeze(ref_mp[:, 2 * k + 1]), comp_mp.right_I_)
