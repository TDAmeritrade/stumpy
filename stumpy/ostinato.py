import numpy as np
from . import core, stump


def ostinato(tss, m):
    """
    Find the consensus motif of multiple time series

    This is a wrapper around the vanilla version of the ostinato algorithm
    which finds the best radius and a helper function that finds the most
    central conserved motif.

    Parameters
    ----------
    tss : list
        List of time series for which to find the consensus motif

    m : int
        Window size

    Returns
    -------
    rad : float
        Radius of the most central consensus motif

    ts_ind : int
        Index of time series which contains the most central consensus motif

    ss_ind : int
        Start index of the most central consensus motif within the time series
        `ts_ind` that contains it

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
    rad, ts_ind, ss_ind = _ostinato(tss, m)
    return _get_central_motif(tss, rad, ts_ind, ss_ind, m)


def _ostinato(tss, m):
    """
    Find the consensus motif of multiple time series

    Parameters
    ----------
    tss : list
        List of time series for which to find the consensus motif

    m : int
        Window size

    Returns
    -------
    bsf_rad : float
        Radius of the consensus motif

    ts_ind : int
        Index of time series which contains the consensus motif

    ss_ind : int
        Start index of consensus motif within the time series ts_ind
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
    # Preprocess means and stddevs and handle np.nan/np.inf
    Ts = [None] * len(tss)
    M_Ts = [None] * len(tss)
    Σ_Ts = [None] * len(tss)
    for i, T in enumerate(tss):
        Ts[i], M_Ts[i], Σ_Ts[i] = core.preprocess(T, m)

    bsf_rad, ts_ind, ss_ind = np.inf, 0, 0
    k = len(Ts)
    for j in range(k):
        if j < (k - 1):
            h = j + 1
        else:
            h = 0

        mp = stump(Ts[j], m, Ts[h], ignore_trivial=False)
        si = np.argsort(mp[:, 0])
        for q in si:
            rad = mp[q, 0]
            if rad >= bsf_rad:
                break
            for i in range(k):
                if ~np.isin(i, [j, h]):
                    QT = core.sliding_dot_product(Ts[j][q : q + m], Ts[i])
                    rad = np.max(
                        (
                            rad,
                            np.min(
                                core._mass(
                                    Ts[j][q : q + m],
                                    Ts[i],
                                    QT,
                                    M_Ts[j][q],
                                    Σ_Ts[j][q],
                                    M_Ts[i],
                                    Σ_Ts[i],
                                )
                            ),
                        )
                    )
                    if rad >= bsf_rad:
                        break
            if rad < bsf_rad:
                bsf_rad, ts_ind, ss_ind = rad, j, q

    return bsf_rad, ts_ind, ss_ind


def _get_central_motif(tss, rad, ts_ind, ss_ind, m):
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
    ss_ind_nn_ost, d_ost = _across_series_nearest_neighbors(q_ost, tss)

    # Alternative candidates: Distance to ostinato hit equals best radius
    ts_ind_alt = np.flatnonzero(np.isclose(d_ost, rad))
    ss_ind_alt = ss_ind_nn_ost[ts_ind_alt]
    d_alt = np.zeros((len(ts_ind_alt), k), dtype=float)
    for i, (tsi, ssi) in enumerate(zip(ts_ind_alt, ss_ind_alt)):
        q = tss[tsi][ssi : ssi + m]
        _, d_alt[i] = _across_series_nearest_neighbors(q, tss)
    rad_alt = np.max(d_alt, axis=1)
    d_mean_alt = np.mean(d_alt, axis=1)

    # Alternatives with same radius and lower mean distance
    alt_better = np.logical_and(
        np.isclose(rad_alt, rad),
        d_mean_alt < d_ost.mean(),
    )
    # Alternatives with same radius and same mean distance
    alt_same = np.logical_and(
        np.isclose(rad_alt, rad),
        np.isclose(d_mean_alt, d_ost.mean()),
    )
    if np.any(alt_better):
        ts_ind_alt = ts_ind_alt[alt_better]
        d_mean_alt = d_mean_alt[alt_better]
        i_alt_best = np.argmin(d_mean_alt)
        ts_ind = ts_ind_alt[i_alt_best]
    elif np.any(alt_same):
        # Default to the first match in the list of time series
        ts_ind_alt = ts_ind_alt[alt_same]
        i_alt_first = np.argmin(ts_ind_alt)
        ts_ind = np.min((ts_ind, ts_ind_alt[i_alt_first]))
    ss_ind = ss_ind_nn_ost[ts_ind]
    return rad, ts_ind, ss_ind


def _across_series_nearest_neighbors(q, tss):
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
        dp = core.mass(q, tss[i])
        ss_ind_nn[i] = np.argmin(dp)
        d[i] = dp[ss_ind_nn[i]]
    return ss_ind_nn, d
