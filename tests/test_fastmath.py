import numba
import numpy as np
import pytest

from stumpy import fastmath

if numba.config.DISABLE_JIT:
    pytest.skip("Skipping Tests JIT is disabled", allow_module_level=True)


def test_set():
    # The test is done by changing the value of fastmath flag for
    # the fastmath._add_assoc function, taken from the following link:
    # https://numba.pydata.org/numba-doc/dev/user/performance-tips.html#fastmath

    # case1: flag=False
    fastmath._set("fastmath", "_add_assoc", flag=False)
    out = fastmath._add_assoc(0, np.inf)
    assert np.isnan(out)

    # case2: flag={'reassoc', 'nsz'}
    fastmath._set("fastmath", "_add_assoc", flag={"reassoc", "nsz"})
    out = fastmath._add_assoc(0, np.inf)
    assert out == 0.0

    # case3: flag={'reassoc'}
    fastmath._set("fastmath", "_add_assoc", flag={"reassoc"})
    out = fastmath._add_assoc(0, np.inf)
    assert np.isnan(out)

    # case4: flag={'nsz'}
    fastmath._set("fastmath", "_add_assoc", flag={"nsz"})
    out = fastmath._add_assoc(0, np.inf)
    assert np.isnan(out)


def test_reset():
    # The test is done by changing the value of fastmath flag for
    # the fastmath._add_assoc function, taken from the following link:
    # https://numba.pydata.org/numba-doc/dev/user/performance-tips.html#fastmath
    # and then reset it to the default value, i.e. `True`
    fastmath._set("fastmath", "_add_assoc", False)
    fastmath._reset("fastmath", "_add_assoc")
    assert fastmath._add_assoc(0.0, np.inf) == 0.0
