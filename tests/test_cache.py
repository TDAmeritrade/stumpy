import numpy as np

from stumpy import cache, stump


def test_cache_get_njit_funcs():
    njit_funcs = cache.get_njit_funcs()
    assert len(njit_funcs) > 0


def test_cache_save_after_clear():
    T = np.random.rand(10)
    m = 3

    cache_dir = "stumpy/__pycache__"

    cache.save(cache_dir)
    stump(T, m)
    ref_cache = cache._get_cache(cache_dir)

    cache.clear(cache_dir)
    assert len(cache._get_cache(cache_dir)) == 0

    cache.save(cache_dir)
    stump(T, m)
    comp_cache = cache._get_cache(cache_dir)

    assert sorted(ref_cache) == sorted(comp_cache)

    cache.clear(cache_dir)
