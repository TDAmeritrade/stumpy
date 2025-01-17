import numba
import numpy as np
import pytest

from stumpy import cache, stump

if numba.config.DISABLE_JIT:
    pytest.skip("Skipping Tests JIT is disabled", allow_module_level=True)


def test_cache_save():
    def get_cache_fnames_ref():
        cache._clear()
        cache._enable()
        stump(np.random.rand(10), 3)
        cache_data_fnames = [
            fname for fname in cache._get_cache() if fname.endswith(".nbc")
        ]
        cache_index_fnames = [
            fname for fname in cache._get_cache() if fname.endswith(".nbi")
        ]
        cache._clear()
        return cache_data_fnames, cache_index_fnames

    def get_cache_fnames_comp():
        cache._clear()
        cache._save()
        stump(np.random.rand(10), 3)
        cache_data_fnames = [
            fname for fname in cache._get_cache() if fname.endswith(".nbc")
        ]
        cache_index_fnames = [
            fname for fname in cache._get_cache() if fname.endswith(".nbi")
        ]
        cache._clear()
        return cache_data_fnames, cache_index_fnames

    ref_data, ref_index = get_cache_fnames_ref()
    comp_data, comp_index = get_cache_fnames_comp()

    assert sorted(ref_data) == sorted(comp_data)
    assert set(ref_index).issubset(comp_index)


def test_cache_save_after_clear():
    T = np.random.rand(10)
    m = 3
    stump(T, m)

    cache._save()
    ref_cache = cache._get_cache()

    cache._clear()
    # testing cache._clear()
    assert len(cache._get_cache()) == 0

    cache._save()
    comp_cache = cache._get_cache()

    # testing cache._save() after cache._clear()
    assert sorted(ref_cache) == sorted(comp_cache)
