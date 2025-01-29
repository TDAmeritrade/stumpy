import numba
import numpy as np

from stumpy import cache, stump


def test_cache_get_njit_funcs():
    njit_funcs = cache.get_njit_funcs()
    assert len(njit_funcs) > 0


def test_cache_save_after_clear():
    T = np.random.rand(10)
    m = 3

    cache.save()
    stump(T, m)
    ref_cache = cache._get_cache()

    if numba.config.DISABLE_JIT:
        assert len(ref_cache) == 0
    else:  # pragma: no cover
        assert len(ref_cache) > 0

    cache.clear()
    assert len(cache._get_cache()) == 0

    cache.save()
    stump(T, m)
    comp_cache = cache._get_cache()

    assert sorted(ref_cache) == sorted(comp_cache)

    cache.clear()
