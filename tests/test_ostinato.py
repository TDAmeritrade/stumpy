import numpy as np
import numpy.testing as npt
import stumpy
from stumpy.ostinato import _get_central_motif
import pytest


def naive_consensus_search(tss, m):
    """
    Brute force consensus motif from
    <https://www.cs.ucr.edu/~eamonn/consensus_Motif_ICDM_Long_version.pdf>

    See Table 1

    Note that there is a bug in the pseudocode at line 8 where `i` should be `j`.
    This implementation fixes it.
    """
    k = len(tss)

    rad = np.inf
    ts_ind = 0
    ss_ind = 0

    for j in range(k):
        radii = np.zeros(len(tss[j]) - m + 1)
        for i in range(k):
            if i != j:
                mp = stumpy.stump(tss[j], m, tss[i], ignore_trivial=False)
                radii = np.max((radii, mp[:, 0]), axis=0)
        min_rad_index = np.argmin(radii)
        min_rad = radii[min_rad_index]
        if min_rad < rad:
            rad = min_rad
            ts_ind = j
            ss_ind = min_rad_index

    return _get_central_motif(tss, rad, ts_ind, ss_ind, m)


@pytest.mark.parametrize(
    "seed", np.random.choice(np.arange(10000), size=100, replace=False)
)
def test_ostinato(seed):
    m = 50
    np.random.seed(seed)
    tss = [np.random.rand(n) for n in [64, 128, 256]]

    rad_naive, ts_ind_naive, ss_ind_naive = naive_consensus_search(tss, m)
    rad_ostinato, ts_ind_ostinato, ss_ind_ostinato = stumpy.ostinato(tss, m)

    npt.assert_almost_equal(rad_naive, rad_ostinato)
    npt.assert_almost_equal(ts_ind_naive, ts_ind_ostinato)
    npt.assert_almost_equal(ss_ind_naive, ss_ind_ostinato)
