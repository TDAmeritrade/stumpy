import numba
import numpy as np
import pytest

from stumpy import cache, stump

if numba.config.DISABLE_JIT:
    pytest.skip("Skipping Tests JIT is disabled", allow_module_level=True)


def test_cache_save_after_clear():
    T = np.random.rand(10)
    m = 3
    stump(T, m)

    cache.save()
    ref_cache = cache._get_cache()

    cache.clear()
    # testing cache._clear()
    assert len(cache._get_cache()) == 0

    cache.save()
    comp_cache = cache._get_cache()

    # testing cache._save() after cache._clear()
    assert sorted(ref_cache) == sorted(comp_cache)
