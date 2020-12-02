# STUMPY
# Copyright 2019 TD Ameritrade. Released under the terms of the 3-Clause BSD license.
# STUMPY is a trademark of TD Ameritrade IP Company, Inc. All rights reserved.

import numpy as np

from . import core, stump


def _get_central_motif(Ts, radius, Ts_idx, subseq_idx, m):
    """
    Compare subsequences with the same radius and return the most central motif

    Parameters
    ----------
    Ts : list
        A list of time series for which to find the most central motif

    radius : float
        Best radius found by a consensus search algorithm

    Ts_idx : int
        The index of time series in `Ts` where the `radius` was first observed

    subseq_idx : int
        The subsequence index in `Ts[ts_idx]` that has radius `radius`

    m : int
        Window size

    Returns
    -------
    radius : float
        Radius of the most central consensus motif

    Ts_idx : int
        The index of time series in `Ts` which contains the most central consensus motif

    subseq_idx : int
        The subsequence index in the time series `Ts[Ts_idx]` that contains the most
        central consensus motif
    """
    k = len(Ts)

    # Ostinato hit: get nearest neighbors' distances and indices
    q_ost = Ts[Ts_idx][subseq_idx : subseq_idx + m]
    subseq_idx_nn_ost, d_ost = _across_series_nearest_neighbors(q_ost, Ts)

    # Alternative candidates: Distance to ostinato hit equals best radius
    Ts_idx_alt = np.flatnonzero(np.isclose(d_ost, radius))
    subseq_idx_alt = subseq_idx_nn_ost[Ts_idx_alt]
    d_alt = np.zeros((len(Ts_idx_alt), k), dtype=float)
    for i, (Tsi, subseqi) in enumerate(zip(Ts_idx_alt, subseq_idx_alt)):
        q = Ts[Tsi][subseqi : subseqi + m]
        _, d_alt[i] = _across_series_nearest_neighbors(q, Ts)
    rad_alt = np.max(d_alt, axis=1)
    d_mean_alt = np.mean(d_alt, axis=1)

    # Alternatives with same radius and lower mean distance
    alt_better = np.logical_and(
        np.isclose(rad_alt, radius),
        d_mean_alt < d_ost.mean(),
    )
    # Alternatives with same radius and same mean distance
    alt_same = np.logical_and(
        np.isclose(rad_alt, radius),
        np.isclose(d_mean_alt, d_ost.mean()),
    )
    if np.any(alt_better):
        Ts_idx_alt = Ts_idx_alt[alt_better]
        d_mean_alt = d_mean_alt[alt_better]
        i_alt_best = np.argmin(d_mean_alt)
        Ts_idx = Ts_idx_alt[i_alt_best]
    elif np.any(alt_same):
        # Default to the first match in the list of time series
        Ts_idx_alt = Ts_idx_alt[alt_same]
        i_alt_first = np.argmin(Ts_idx_alt)
        Ts_idx = np.min((Ts_idx, Ts_idx_alt[i_alt_first]))
    subseq_idx = subseq_idx_nn_ost[Ts_idx]
    return radius, Ts_idx, subseq_idx


def _across_series_nearest_neighbors(q, Ts):
    """
    For multiple time series find, per individual time series, the subsequences closest
    to a query.

    Parameters
    ----------
    q : ndarray
        Query array or subsequence

    Ts : list
        A list of time series for which to find the nearest neighbor subsequences that
        are closest to the query `q`

    Returns
    -------
    ss_ind_nn : ndarray
        Indices to subsequences in `tss` that are closest to `q`

    d : ndarray
        Distances to subsequences in `tss` that are closest to `q`
    """
    k = len(Ts)
    d = np.zeros(k, dtype=float)
    subseq_idx_nn = np.zeros(k, dtype=int)
    for i in range(k):
        dp = core.mass(q, Ts[i])
        subseq_idx_nn[i] = np.argmin(dp)
        d[i] = dp[subseq_idx_nn[i]]
    return subseq_idx_nn, d


def _ostinato(Ts, m):
    """
    Find the consensus motif amongst a list of time series

    Parameters
    ----------
    Ts : list
        A list of time series for which to find the consensus motif

    m : int
        Window size

    Returns
    -------
    bsf_radius : float
        The (best-so-far) Radius of the consensus motif

    Ts_idx : int
        The time series index in `Ts` which contains the most central consensus motif

    subseq_idx : int
        The subsequence index within time series `Ts[Ts_idx]` the contains most
        central consensus motif

    Notes
    -----
    `DOI: 10.1109/ICDM.2019.00140 \
    <https://www.cs.ucr.edu/~eamonn/consensus_Motif_ICDM_Long_version.pdf>`__

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
    M_Ts = [None] * len(Ts)
    Σ_Ts = [None] * len(Ts)
    for i, T in enumerate(Ts):
        Ts[i], M_Ts[i], Σ_Ts[i] = core.preprocess(T, m)

    bsf_radius = np.inf
    Ts_idx = 0
    subseq_idx = 0

    k = len(Ts)
    for j in range(k):
        if j < (k - 1):
            h = j + 1
        else:
            h = 0

        mp = stump(Ts[j], m, Ts[h], ignore_trivial=False)
        si = np.argsort(mp[:, 0])
        for q in si:
            radius = mp[q, 0]
            if radius >= bsf_radius:
                break
            for i in range(k):
                if i != j and i != h:
                    QT = core.sliding_dot_product(Ts[j][q : q + m], Ts[i])
                    radius = np.max(
                        (
                            radius,
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
                    if radius >= bsf_radius:
                        break
            if radius < bsf_radius:
                bsf_radius, Ts_idx, subseq_idx = radius, j, q

    return bsf_radius, Ts_idx, subseq_idx


def ostinato(Ts, m):
    """
    Find the consensus motif of multiple time series

    This is a wrapper around the vanilla version of the ostinato algorithm
    which finds the best radius and a helper function that finds the most
    central conserved motif.

    Parameters
    ----------
    Ts : list
        A list of time series for which to find the consensus motif

    m : int
        Window size

    Returns
    -------
    radius : float
        Radius of the most central consensus motif

    Ts_idx : int
        The time series index in `Ts` which contains the most central consensus motif

    subseq_idx : int
        The subsequence index within time series `Ts[Ts_idx]` the contains most central
        consensus motif

    Notes
    -----
    `DOI: 10.1109/ICDM.2019.00140 \
    <https://www.cs.ucr.edu/~eamonn/consensus_Motif_ICDM_Long_version.pdf>`__

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
    radius, Ts_idx, subseq_idx = _ostinato(Ts, m)
    return _get_central_motif(Ts, radius, Ts_idx, subseq_idx, m)
