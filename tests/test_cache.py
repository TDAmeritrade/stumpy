import numba
import numpy as np
import pytest

from stumpy import cache, stump

if numba.config.DISABLE_JIT:
    pytest.skip("Skipping Tests JIT is disabled", allow_module_level=True)


def test_cache_save():
    def get_cache_fnames_ref():
        cache.clear()
        cache._enable()
        stump(np.random.rand(10), 3)
        cache_files = cache._get_cache()
        cache.clear()
        return cache_files

    def get_cache_fnames_comp():
        cache.clear()
        cache.save()
        stump(np.random.rand(10), 3)
        cache_files = cache._get_cache()
        cache.clear()
        return cache_files

    ref_cache_files = get_cache_fnames_ref()
    comp_cache_files = get_cache_fnames_comp()

    # check nbc files
    ref_nbc = [fname for fname in ref_cache_files if fname.endswith(".nbc")]
    comp_nbc = [fname for fname in comp_cache_files if fname.endswith(".nbc")]
    assert sorted(ref_nbc) == sorted(comp_nbc)

    # check nbi files
    ref_nbi = [fname for fname in ref_cache_files if fname.endswith(".nbi")]
    comp_nbi = [fname for fname in comp_cache_files if fname.endswith(".nbi")]
    assert set(ref_nbi).issubset(comp_nbi)


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
