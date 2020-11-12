import numpy as np
import numpy.testing as npt
import stumpy

def test_ostinato():
    m = 50
    ts = [np.random.rand(n) for n in [64, 128, 256]]

    rad_naive, tsind_naive, ssind_naive = naive_consensus_search(ts, m)
    rad_ostinato, tsind_ostinato, ssind_ostinato = stumpy.ostinato(ts, m)

    npt.assert_almost_equal(rad_naive, rad_ostinato)
    npt.assert_almost_equal(tsind_naive, tsind_ostinato)
    npt.assert_almost_equal(ssind_naive, ssind_ostinato)
