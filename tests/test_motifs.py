import numpy as np
import numpy.testing as npt
import pytest

from stumpy import motifs, stump


def test_k_motifs_static():
    T = np.array([0.0, 1.0, 0.0, -1.0, -1.0, 0.0, 1.0, 0.0, -0.5])
    m = 3
    k = 1

    left_indices = [[0, 5]]
    left_profile_values = [0]

    P = stump(T, m, ignore_trivial=True)
    right_indices, right_profile_values = motifs.k_motifs(T, P, k=k, atol=0.001)

    npt.assert_array_equal(left_indices, right_indices)
    npt.assert_almost_equal(left_profile_values, right_profile_values)


def test_find_occurences_static():
    T = np.array([0.0, 1.0, 0.0, -1.0, -1.0, 0.0, 1.0, 0.0, -0.5])
    Q = np.array([0.0, 1.0, 0.0])

    left = [0, 5]
    right = motifs.find_occurences(Q, T, atol=0.001)

    npt.assert_array_equal(left, right)
