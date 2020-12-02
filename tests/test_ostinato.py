import numpy as np
import numpy.testing as npt
import stumpy
import naive
from stumpy.ostinato import _get_central_motif
import pytest


def naive_consensus_search(Ts, m):
    """
    Brute force consensus motif from
    <https://www.cs.ucr.edu/~eamonn/consensus_Motif_ICDM_Long_version.pdf>

    See Table 1

    Note that there is a bug in the pseudocode at line 8 where `i` should be `j`.
    This implementation fixes it.
    """
    k = len(Ts)

    bsf_radius = np.inf
    Ts_idx = 0
    subseq_idx = 0

    for j in range(k):
        radii = np.zeros(len(Ts[j]) - m + 1)
        for i in range(k):
            if i != j:
                mp = naive.stump(Ts[j], m, Ts[i])
                radii = np.maximum(radii, mp[:, 0])
        min_radius_idx = np.argmin(radii)
        min_radius = radii[min_radius_idx]
        if min_radius < bsf_radius:
            bsf_radius = min_radius
            Ts_idx = j
            subseq_idx = min_radius_idx

    return naive_get_central_motif(Ts, bsf_radius, Ts_idx, subseq_idx, m)


def naive_get_central_motif(Ts, radius, Ts_idx, subseq_idx, m):
    """
    Compare subsequences with the same radius and return the most central motif

    Parameters
    ----------
    Ts : list
        List of time series for which to find the most central motif

    radius : float
        Best radius found by a consensus search algorithm

    Ts_idx : int
        Index of time series in which `radius` was first found

    subseq_idx : int
        Start index of the subsequence in `Ts[Ts_idx]` that has radius `radius`

    m : int
        Window size

    Returns
    -------
    radius : float
        Radius of the most central consensus motif

    Ts_idx : int
        Index of time series which contains the most central consensus motif

    subseq_idx : int
        Start index of most central consensus motif within the time series `Ts_idx`
        that contains it

    Notes
    -----
    <https://www.cs.ucr.edu/~eamonn/consensus_Motif_ICDM_Long_version.pdf>

    See Table 2

    The ostinato algorithm proposed in the paper finds the best radius
    in `Ts`. Intuitively, the radius is the minimum distance of a
    subsequence to encompass at least one nearest neighbor subsequence
    from all other time series. The best radius in `Ts` is the minimum
    radius amongst all radii. Some data sets might contain multiple
    subsequences which have the same optimal radius.
    The greedy Ostinato algorithm only finds one of them, which might
    not be the most central motif. The most central motif amongst the
    subsequences with the best radius is the one with the smallest mean
    distance to nearest neighbors in all other time series. To find this
    central motif it is necessary to search the subsequences with the
    best radius via `stumpy.ostinato._get_central_motif`
    """
    k = len(Ts)

    # Ostinato hit: get nearest neighbors' distances and indices
    q_ost = Ts[Ts_idx][subseq_idx : subseq_idx + m]
    subseq_idx_nn_ost, d_ost = naive_across_series_nearest_neighbors(q_ost, Ts)

    # Alternative candidates: Distance to ostinato hit equals best radius
    Ts_idx_alt = np.flatnonzero(np.isclose(d_ost, radius))
    subseq_idx_alt = subseq_idx_nn_ost[Ts_idx_alt]
    d_alt = np.zeros((len(Ts_idx_alt), k), dtype=float)
    for i, (Tsi, subseqi) in enumerate(zip(Ts_idx_alt, subseq_idx_alt)):
        q = Ts[Tsi][subseqi : subseqi + m]
        _, d_alt[i] = naive_across_series_nearest_neighbors(q, Ts)
    rad_alt = np.sort(d_alt, axis=1)[:, -1]
    d_mean_alt = np.mean(d_alt, axis=1)

    # Alternatives with same radius and lower mean distance
    alt_better = (
        np.isclose(rad_alt, radius).astype(int)
        + (d_mean_alt < d_ost.mean()).astype(int)
    ) == 2
    # Alternatives with same radius and same mean distance
    alt_same = (
        np.isclose(rad_alt, radius).astype(int)
        + np.isclose(d_mean_alt, d_ost.mean()).astype(int)
    ) == 2
    if np.any(alt_better):
        Ts_idx_alt = Ts_idx_alt[alt_better]
        subseq_idx_alt = subseq_idx_alt[alt_better]
        d_mean_alt = d_mean_alt[alt_better]
        i_alt_best = np.argmin(d_mean_alt)
        return radius, Ts_idx_alt[i_alt_best], subseq_idx_alt[i_alt_best]
    elif np.any(alt_same):
        # Default to the first match in the list of time series
        Ts_idx_alt = Ts_idx_alt[alt_same]
        subseq_idx_alt = subseq_idx_alt[alt_same]
        i_alt_first = np.argmin(Ts_idx_alt)
        if Ts_idx_alt[i_alt_first] < Ts_idx:
            return radius, Ts_idx_alt[i_alt_first], subseq_idx_alt[i_alt_first]
        else:
            return radius, Ts_idx, subseq_idx
    else:
        return radius, Ts_idx, subseq_idx


def naive_across_series_nearest_neighbors(q, Ts):
    """
    For multiple time series find, per individual time series, the subsequences closest
    to a query.

    Parameters
    ----------
    q : ndarray
        Query array or subsequence

    Ts : list
        List of time series for which to the nearest neighbors to `q`

    Returns
    -------
    subseq_idx_nn : ndarray
        Indices to subsequences in `Ts` that are closest to `q`

    d : ndarray
        Distances to subsequences in `Ts` that are closest to `q`
    """
    k = len(Ts)
    d = np.zeros(k, dtype=float)
    subseq_idx_nn = np.zeros(k, dtype=int)
    for i in range(k):
        dp = naive.distance_profile(q, Ts[i], len(q))
        subseq_idx_nn[i] = np.argmin(dp)
        d[i] = dp[subseq_idx_nn[i]]
    return subseq_idx_nn, d


@pytest.mark.parametrize(
    "seed", np.random.choice(np.arange(10000), size=100, replace=False)
)
def test_ostinato(seed):
    m = 50
    np.random.seed(seed)
    Ts = [np.random.rand(n) for n in [64, 128, 256]]

    ref_radius, ref_Ts_idx, ref_subseq_idx = naive_consensus_search(Ts, m)
    comp_radius, comp_Ts_idx, comp_subseq_idx = stumpy.ostinato(Ts, m)

    npt.assert_almost_equal(ref_radius, comp_radius)
    npt.assert_almost_equal(ref_Ts_idx, comp_Ts_idx)
    npt.assert_almost_equal(ref_subseq_idx, comp_subseq_idx)
