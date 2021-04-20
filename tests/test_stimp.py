import numpy as np
import numpy.testing as npt
import pytest
from stumpy.stimp import _bfs_indices

n = [9, 10, 16]


def split(node, out):
    mid = len(node) // 2
    out.append(node[mid])
    return node[:mid], node[mid + 1 :]


def naive_bsf_indices(n):
    a = np.arange(n)
    nodes = [a.tolist()]
    out = []

    while nodes:
        tmp = []
        for node in nodes:
            for n in split(node, out):
                if n:
                    tmp.append(n)
        nodes = tmp

    return np.array(out)


@pytest.mark.parametrize("n", n)
def test_bsf_indices(n):
    ref_bsf_indices = naive_bsf_indices(n)
    cmp_bsf_indices = np.array(list(_bfs_indices(n)))

    npt.assert_almost_equal(ref_bsf_indices, cmp_bsf_indices)
