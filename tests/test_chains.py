import numpy as np
import numpy.testing as npt
import pytest

from stumpy import allc, atsc

test_data = [
    (
        np.array([47, 32, 1, 22, 2, 58, 3, 36, 4, -5, 5, 40], dtype=np.int64),
        np.array([11, 7, 4, 7, 6, 11, 8, 11, 10, 10, 11, -1], dtype=np.int64),
        np.array([-1, 0, 1, 1, 2, 0, 4, 1, 6, 2, 8, 7], dtype=np.int64),
    )
]


@pytest.mark.parametrize("Value, IR, IL", test_data)
def test_atsc(Value, IR, IL):
    j = 2
    ref = np.array([2, 4, 6, 8, 10], np.int64)
    comp = atsc(IL, IR, j)
    npt.assert_equal(ref, comp)


@pytest.mark.parametrize("Value, IR, IL", test_data)
def test_allc(Value, IR, IL):
    S_ref = [
        np.array([1, 7, 11], dtype=np.int64),
        np.array([0], dtype=np.int64),
        np.array([3], dtype=np.int64),
        np.array([9], dtype=np.int64),
        np.array([2, 4, 6, 8, 10], dtype=np.int64),
        np.array([5], dtype=np.int64),
    ]
    C_ref = np.array([2, 4, 6, 8, 10], dtype=np.int64)
    S_comp, C_comp = allc(IL, IR)

    S_ref = sorted(S_ref, key=lambda x: (len(x), list(x)))
    S_comp = sorted(S_comp, key=lambda x: (len(x), list(x)))

    npt.assert_equal(S_ref, S_comp)
    npt.assert_equal(C_ref, C_comp)
