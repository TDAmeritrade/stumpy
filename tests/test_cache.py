import numpy as np

from stumpy import cache, stump


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


if __name__ == "__main__":
    test_cache_save()
