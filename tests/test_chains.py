import numpy as np
import numpy.testing as npt
from stumpy import atsc, allc
import pytest

test_data = [
    (np.array([47, 32, 1, 22, 2, 58, 3, 36, 4, -5, 5, 40], dtype=np.int64),
     np.array([11, 7, 4, 7, 6, 11, 8, 11, 10, 10, 11, -1], dtype=np.int64),
     np.array([-1, 0, 1, 1, 2, 0, 4, 1, 6, 2, 8, 7], dtype=np.int64),)
]

@pytest.mark.parametrize("Value, IR, IL", test_data)
def test_atsc(Value, IR, IL):
    j = 2
    left = np.array([2, 4, 6, 8, 10], np.int64)
    right = atsc(IL, IR, j)
    npt.assert_equal(left, right)

@pytest.mark.parametrize("Value, IR, IL", test_data)
def test_allc(Value, IR, IL):
    j = 2
    S_left = [np.array([ 1,  7, 11], dtype=np.int64), 
              np.array([0], dtype=np.int64), 
              np.array([3], dtype=np.int64),
              np.array([9], dtype=np.int64),
              np.array([2,  4,  6,  8, 10], dtype=np.int64), 
              np.array([5], dtype=np.int64)
             ]
    C_left = np.array([2, 4, 6, 8, 10], dtype=np.int64)
    S_right, C_right = allc(IL, IR)
    npt.assert_equal(S_left, S_right)
    npt.assert_equal(C_left, C_right)