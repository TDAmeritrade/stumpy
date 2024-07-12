import naive
import numpy as np
import numpy.testing as npt
import pytest

from stumpy import maamp, mstump

test_data = [
    (np.array([[584, -11, 23, 79, 1001, 0, -19]], dtype=np.float64), 3),
    (np.random.uniform(-1000, 1000, [5, 20]).astype(np.float64), 5),
]


@pytest.mark.parametrize("T, m", test_data)
def test_mmparray_mstump(T, m):
    excl_zone = int(np.ceil(m / 4))

    ref_P, ref_I = naive.mstump(T, m, excl_zone)
    comp = mstump(T, m)

    npt.assert_almost_equal(ref_P, comp.P_)
    npt.assert_almost_equal(ref_I, comp.I_)


@pytest.mark.parametrize("T, m", test_data)
def test_mmparray_maamp(T, m):
    excl_zone = int(np.ceil(m / 4))

    for p in [1.0, 2.0, 3.0]:
        ref_P, ref_I = naive.maamp(T, m, excl_zone, p=p)
        comp = maamp(T, m, p=p)

        npt.assert_almost_equal(ref_P, comp.P_)
        npt.assert_almost_equal(ref_I, comp.I_)
