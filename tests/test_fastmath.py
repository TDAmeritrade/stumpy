import numpy as np
from numba import njit

from stumpy import fastmath


def test_set():
    # Test the _set and _reset function in fastmath.py
    # The test is done by changing the value of fastmath flag for
    # the fastmath._add_assoc function, taken from the following link:
    # https://numba.pydata.org/numba-doc/dev/user/performance-tips.html#fastmath
    py_func = fastmath._add_assoc.py_func

    x = 0.0
    y = np.inf
    fastmath_flags = [False, {"reassoc", "nsz"}, {"reassoc"}, {"nsz"}]
    for flag in fastmath_flags:
        ref = njit(fastmath=flag)(py_func)(x, y)

        fastmath._set("fastmath", "_add_assoc", flag)
        comp = fastmath._add_assoc(x, y)

        if np.isnan(ref) and np.isnan(comp):
            assert True
        else:
            assert ref == comp


def test_reset():
    # Test the _set and _reset function in fastmath.py
    # The test is done by changing the value of fastmath flag for
    # the fastmath._add_assoc function, taken from the following link:
    # https://numba.pydata.org/numba-doc/dev/user/performance-tips.html#fastmath
    fastmath._set("fastmath", "_add_assoc", False)
    fastmath._reset("fastmath", "_add_assoc")
    assert fastmath._add_assoc(0.0, np.inf) == 0.0
