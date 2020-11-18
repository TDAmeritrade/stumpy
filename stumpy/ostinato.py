import numpy as np
from . import core, stump


def ostinato(tss, m):
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


def _get_central_motif(ts, rad, tsi, ssi, m):
    """
    Compare subsequences with the same radius and return the most central motif

    Parameters
    ----------
    ts : list
        List of time series for which to find the most central motif

    rad : float
        Best radius found by a consensus search algorithm

    tsi : int
        Index of time series in which `rad` was found first

    ssi : int
        Start index of subsequence in `tsi` that has radius `rad`

    m : int
        Window size

    Returns
    -------
    rad : float
        Radius of the most central consensus motif

    tsi : int
        Index of time series which contains the most central consensus motif

    ssi : int
        Start index of most central consensus motif within the time series `tsi`
        that contains it
    """
    k = len(ts)
    # Get nearest neighbors for ostinato hit
    nn_ost, d_ost = _across_series_nearest_neighbors(ts, tsi, ssi, m)
    # Alternative candidates with same radius
    tsi_alt = np.flatnonzero(np.isclose(d_ost, rad))
    ssi_alt = nn_ost[tsi_alt]
    num_alt = len(tsi_alt)
    d_alt = np.zeros((num_alt, k), dtype=float)
    for i in range(num_alt):
        _, d_alt[i] = _across_series_nearest_neighbors(ts, tsi_alt[i], ssi_alt[i], m)
    d_alt_sum = np.sum(d_alt, axis=1)
    if np.any(d_alt_sum < d_ost.sum()):
        i_alt_best = np.argmin(d_alt_sum)
        return rad, tsi_alt[i_alt_best], ssi_alt[i_alt_best]
    else:
        return rad, tsi, ssi


def _across_series_nearest_neighbors(ts, qtsind, qssind, m):
    k = len(ts)
    q = ts[qtsind][qssind : qssind + m]
    d = np.zeros(k, dtype=float)
    nn = np.zeros(k, dtype=int)
    nn[qtsind] = qssind
    for i in range(k):
        if i != qtsind:
            dp = core.mass(q, ts[i])
            nn[i] = np.argmin(dp)
            d[i] = dp[nn[i]]
    return nn, d
