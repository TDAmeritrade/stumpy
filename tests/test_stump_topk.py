import numpy as np
import numpy.testing as npt
import pandas as pd
from stumpy import stump, config
import pytest
import naive


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

substitution_locations = [(slice(0, 0), 0, -1, slice(1, 3), [0, 3])]
substitution_values = [np.nan, np.inf]


def test_stump_int_input():
    with pytest.raises(TypeError):
        stump(np.arange(10), 5)


@pytest.mark.parametrize("T_A, T_B", test_data)
def test_stump_self_join_1NN(T_A, T_B):
    k = 1
    m = 3
    zone = int(np.ceil(m / 4))
    ref_mp = naive.stump_topk(T_B, m, exclusion_zone=zone, k=k)
    comp_mp = stump(T_B, m, ignore_trivial=True)
    naive.replace_inf(ref_mp)
    naive.replace_inf(comp_mp)
    npt.assert_almost_equal(ref_mp, comp_mp)

    comp_mp = stump(pd.Series(T_B), m, ignore_trivial=True)
    naive.replace_inf(comp_mp)
    npt.assert_almost_equal(ref_mp, comp_mp)


def test_stump_self_join_KNN(T_A, T_B):
    k = 3
    m = 3
    zone = int(np.ceil(m / 4))
    ref_mp = naive.stump_topk(T_B, m, exclusion_zone=zone, k=k)
    comp_mp = stump(T_B, m, ignore_trivial=True)
    naive.replace_inf(ref_mp)
    naive.replace_inf(comp_mp)
    npt.assert_almost_equal(ref_mp, comp_mp)

    comp_mp = stump(pd.Series(T_B), m, ignore_trivial=True)
    naive.replace_inf(comp_mp)
    npt.assert_almost_equal(ref_mp, comp_mp)
