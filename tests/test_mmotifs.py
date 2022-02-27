import numpy as np

# import numpy.testing as npt
# import pytest

from stumpy.mmotifs import mmotifs
from stumpy.mstump import mstump


# These are tests for multidimensional motif discovery


def test_motifs_multidimensional_one_motif_all_dimensions():
    T = np.array(
        [
            [5.0, 0.0, 3.0, 3.0, 7.0, 9.0, 3.0, 5.0, 2.0, 4.0, 7.0, 6.0, 8.0, 8.0, 1.0],
            [
                7.0,
                3.0,
                5.0,
                9.0,
                8.0,
                7.0,
                4.0,
                5.0,
                10.0,
                8.0,
                4.0,
                3.0,
                2.0,
                0.0,
                1.0,
            ],
            [6.0, 7.0, 7.0, 8.0, 1.0, 5.0, 9.0, 8.0, 9.0, 4.0, 3.0, 0.0, 3.0, 5.0, 0.0],
            [
                0.0,
                1.0,
                3.0,
                2.0,
                6.0,
                1.0,
                9.0,
                10.0,
                1.0,
                2.0,
                2.0,
                5.0,
                1.0,
                0.0,
                4.0,
            ],
        ]
    )
    m = 3

    P, I = mstump(T, m)
    motif_distances, motif_indices, motif_subspaces, motif_mdls = mmotifs(
        T, P, I, max_motifs=1
    )


# def test_motifs_multidimensional_two_motifs_all_dimensions():
