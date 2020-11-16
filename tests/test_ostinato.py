import numpy as np
import numpy.testing as npt
import stumpy
import pytest


def naive_consensus_search(ts, m):
    """
    Brute force consensus motif from
    <https://www.cs.ucr.edu/~eamonn/consensus_Motif_ICDM_Long_version.pdf>

    See Table 1

    Note that there is a bug in the pseudocode at line 8 where `i` should be `j`.
    This implementation fixes it.
    """
    k = len(ts)

    rad = np.inf
    tsind = 0
    ssind = 0

    for j in range(k):
        radii = np.zeros(len(ts[j]) - m + 1)
        for i in range(k):
            if i != j:
                mp = stumpy.stump(ts[j], m, ts[i], ignore_trivial=False)
                radii = np.max((radii, mp[:, 0]), axis=0)
        min_rad_index = np.argmin(radii)
        min_rad = radii[min_rad_index]
        if min_rad < rad:
            rad = min_rad
            tsind = j
            ssind = min_rad_index

    return rad, tsind, ssind


@pytest.mark.parametrize(
    "seed", np.random.choice(np.arange(10000), size=100, replace=False)
)
def test_ostinato(seed):
    m = 50
    np.random.seed(seed)
    ts = [np.random.rand(n) for n in [64, 128, 256]]

    rad_naive, tsind_naive, ssind_naive = naive_consensus_search(ts, m)
    rad_ostinato, tsind_ostinato, ssind_ostinato = stumpy.ostinato(ts, m)

    npt.assert_almost_equal(rad_naive, rad_ostinato)
