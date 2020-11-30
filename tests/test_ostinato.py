import numpy as np
import numpy.testing as npt
import stumpy
import naive
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
                mp = naive.stump(tss[j], m, tss[i])
                radii = np.sort((radii, mp[:, 0]), axis=0)[-1, :]
        min_rad_index = naive_argmin(radii)
        min_rad = radii[min_rad_index]
        if min_rad < rad:
            rad = min_rad
            ts_ind = j
            ss_ind = min_rad_index

    return naive_get_central_motif(tss, rad, ts_ind, ss_ind, m)


def naive_get_central_motif(tss, rad, ts_ind, ss_ind, m):
    """
    Compare subsequences with the same radius and return the most central motif

    Parameters
    ----------
    tss : list
        List of time series for which to find the most central motif

    rad : float
        Best radius found by a consensus search algorithm

    ts_ind : int
        Index of time series in which `rad` was found first

    ss_ind : int
        Start index of subsequence in `ts_ind` that has radius `rad`

    m : int
        Window size

    Returns
    -------
    rad : float
        Radius of the most central consensus motif

    ts_ind : int
        Index of time series which contains the most central consensus motif

    ss_ind : int
        Start index of most central consensus motif within the time series `ts_ind`
        that contains it

    Notes
    -----
    <https://www.cs.ucr.edu/~eamonn/consensus_Motif_ICDM_Long_version.pdf>

    See Table 2

    The ostinato algorithm proposed in the paper finds the best radius
    in `tss`. Intuitively, the radius is the minimum distance of a
    subsequence to encompass at least one nearest neighbor subsequence
    from all other time series. The best radius in `tss` is the minimum
    radius amongst all radii. Some data sets might contain multiple
    subsequences which have the same optimal radius.
    The greedy Ostinato algorithm only finds one of them, which might
    not be the most central motif. The most central motif amongst the
    subsequences with the best radius is the one with the smallest mean
    distance to nearest neighbors in all other time series. To find this
    central motif it is necessary to search the subsequences with the
    best radius via `stumpy.ostinato._get_central_motif`
    """
    k = len(tss)

    # Ostinato hit: get nearest neighbors' distances and indices
    q_ost = tss[ts_ind][ss_ind : ss_ind + m]
    ss_ind_nn_ost, d_ost = naive_across_series_nearest_neighbors(q_ost, tss)

    # Alternative candidates: Distance to ostinato hit equals best radius
    ts_ind_alt = np.flatnonzero(naive_isclose(d_ost, rad))
    ss_ind_alt = ss_ind_nn_ost[ts_ind_alt]
    d_alt = np.zeros((len(ts_ind_alt), k), dtype=float)
    for i, (tsi, ssi) in enumerate(zip(ts_ind_alt, ss_ind_alt)):
        q = tss[tsi][ssi : ssi + m]
        _, d_alt[i] = naive_across_series_nearest_neighbors(q, tss)
    rad_alt = np.sort(d_alt, axis=1)[:, -1]
    d_mean_alt = np.mean(d_alt, axis=1)

    # Alternatives with same radius and lower mean distance
    alt_better = (
        naive_isclose(rad_alt, rad).astype(int)
        + (d_mean_alt < d_ost.mean()).astype(int)
    ) == 2
    # Alternatives with same radius and same mean distance
    alt_same = (
        naive_isclose(rad_alt, rad).astype(int)
        + naive_isclose(d_mean_alt, d_ost.mean()).astype(int)
    ) == 2
    if np.any(alt_better):
        ts_ind_alt = ts_ind_alt[alt_better]
        ss_ind_alt = ss_ind_alt[alt_better]
        d_mean_alt = d_mean_alt[alt_better]
        i_alt_best = naive_argmin(d_mean_alt)
        return rad, ts_ind_alt[i_alt_best], ss_ind_alt[i_alt_best]
    elif np.any(alt_same):
        # Default to the first match in the list of time series
        ts_ind_alt = ts_ind_alt[alt_same]
        ss_ind_alt = ss_ind_alt[alt_same]
        i_alt_first = naive_argmin(ts_ind_alt)
        if ts_ind_alt[i_alt_first] < ts_ind:
            return rad, ts_ind_alt[i_alt_first], ss_ind_alt[i_alt_first]
        else:
            return rad, ts_ind, ss_ind
    else:
        return rad, ts_ind, ss_ind


def naive_across_series_nearest_neighbors(q, tss):
    """
    For multiple time series find, per individual time series, the subsequences closest
    to a query.

    Parameters
    ----------
    q : ndarray
        Query array or subsequence

    tss : list
        List of time series for which to the nearest neighbors to `q`

    Returns
    -------
    ss_ind_nn : ndarray
        Indices to subsequences in `tss` that are closest to `q`

    d : ndarray
        Distances to subsequences in `tss` that are closest to `q`
    """
    k = len(tss)
    d = np.zeros(k, dtype=float)
    ss_ind_nn = np.zeros(k, dtype=int)
    for i in range(k):
        dp = naive.distance_profile(q, tss[i], len(q))
        ss_ind_nn[i] = naive_argmin(dp)
        d[i] = dp[ss_ind_nn[i]]
    return ss_ind_nn, d


def naive_argmin(a):
    return np.flatnonzero(a == np.sort(a)[0])[0]


def naive_isclose(a, b, rtol=1e-05, atol=1e-08):
    return np.abs(a - b) <= (atol + rtol * np.abs(b))


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
