import numpy as np

from stumpy import cache, stump


def test_cache_get_njit_funcs():
    njit_funcs = cache.get_njit_funcs()
    assert len(njit_funcs) > 0


def test_cache_save_after_clear():
    cache.save()

    T = np.random.rand(10)
    m = 3
    stump(T, m)

    ref_cache = cache._get_cache()

    cache.clear()
    # testing cache._clear()
    assert len(cache._get_cache()) == 0

    cache.save()
    stump(T, m)
    comp_cache = cache._get_cache()

    # testing cache._save() after cache._clear()
    assert sorted(ref_cache) == sorted(comp_cache)

    cache.clear()
