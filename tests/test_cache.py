import numba
import numpy as np
import pytest

from stumpy import cache, stump

if numba.config.DISABLE_JIT:
    pytest.skip("Skipping Tests JIT is disabled", allow_module_level=True)


def test_cache():
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
