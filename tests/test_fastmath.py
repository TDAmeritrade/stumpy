import numpy as np

from stumpy import fastmath


def test_fastmath():
    # Test the _set and _reset function in fastmath.py
    # The test is done by changing the value of fastmath flag for
    # the fastmath._add_assoc function
    # See: https://numba.pydata.org/numba-doc/dev/user/performance-tips.html#fastmath

    x, y = 0.0, np.inf

    # fastmath=False
    fastmath._set("fastmath", "_add_assoc", False)
    out = fastmath._add_assoc(x, y)
    assert np.isnan(out)

    # fastmath={'reassoc', 'nsz'}
    fastmath._set("fastmath", "_add_assoc", {"reassoc", "nsz"})
    out = fastmath._add_assoc(x, y)
    assert out == 0.0

    # fastmath={'reassoc'}
    fastmath._set("fastmath", "_add_assoc", {"reassoc"})
    out = fastmath._add_assoc(x, y)
    assert np.isnan(out)

    # fastmath={'nsz'}
    fastmath._set("fastmath", "_add_assoc", {"nsz"})
    out = fastmath._add_assoc(x, y)
    assert np.isnan(out)

    # reset value of fastmath (default is True)
    fastmath._reset("fastmath", "_add_assoc")
    out = fastmath._add_assoc(x, y)
    assert out == 0.0
