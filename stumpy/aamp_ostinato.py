# STUMPY
# Copyright 2019 TD Ameritrade. Released under the terms of the 3-Clause BSD license.
# STUMPY is a trademark of TD Ameritrade IP Company, Inc. All rights reserved.

import numpy as np

from . import core, aamp, aamped


def _aamp_across_series_nearest_neighbors(
    Ts, Ts_idx, subseq_idx, m, Ts_squared, Ts_subseq_isfinite
):
    """
    For multiple time series find, per individual time series, the subsequences closest
    to a given query.

    Parameters
    ----------
    Ts : list
        A list of time series for which to find the nearest neighbor subsequences that
        are closest to the query subsequence `Ts[Ts_idx][subseq_idx : subseq_idx + m]`

    Ts_idx : int
        The index of time series in `Ts` which contains the query subsequence

    subseq_idx : int
        The subsequence index in the time series `Ts[Ts_idx]` that contains the query
        subsequence

    m : int
        Window size

    Ts_squared : list
        A list of rolling window `T_squared` for each time series in `Ts`

    Ts_subseq_isfinite : list
        A list of rolling window `T_subseq_isfinite` for each time series in `Ts`

    Returns
    -------
    nns_radii : numpy.ndarray
        Radii to subsequences in each time series of `Ts` that are closest to the
        query subsequence `Ts[Ts_idx][subseq_idx : subseq_idx + m]`

    nns_subseq_idx : numpy.ndarray
        Indices to subsequences in each time series of `Ts` that are closest to the
        query subsequence `Ts[Ts_idx][subseq_idx : subseq_idx + m]`
    """
    k = len(Ts)
    Q = Ts[Ts_idx][subseq_idx : subseq_idx + m]
    Q_squared = np.sum(Q * Q)
    nns_radii = np.zeros(k, dtype=np.float64)
    nns_subseq_idx = np.zeros(k, dtype=np.int64)

    for i in range(k):
        if np.any(~np.isfinite(Q)):  # pragma: no cover
            distance_profile = np.empty(Ts[i].shape[0] - m + 1, dtype=np.float64)
            distance_profile[:] = np.inf
        else:
            QT = core.sliding_dot_product(
                Ts[Ts_idx][subseq_idx : subseq_idx + m], Ts[i]
            )
            distance_profile = core._mass_absolute(Q_squared, Ts_squared[i], QT)
            distance_profile[~Ts_subseq_isfinite[i]] = np.inf
        nns_subseq_idx[i] = np.argmin(distance_profile)
        nns_radii[i] = distance_profile[nns_subseq_idx[i]]

    return nns_radii, nns_subseq_idx


def _get_aamp_central_motif(
    Ts, bsf_radius, bsf_Ts_idx, bsf_subseq_idx, m, Ts_squared, Ts_subseq_isfinite
):
    """
    Compare subsequences with the same radius and return the most central motif (i.e.,
    having the smallest average nearest neighbor radii)

    Parameters
    ----------
    Ts : list
        A list of time series for which to find the most central motif

    bsf_radius : float
        Best-so-far sradius found by a consensus search algorithm

    bsf_Ts_idx : int
        The index of time series in `Ts` where the `bsf_radius` was first observed

    bsf_subseq_idx : int
        The subsequence index in `Ts[bsf_Ts_idx]` that has radius `bsf_radius`

    m : int
        Window size

    Ts_squared : list
        A list of rolling window `T_squared` for each time series in `Ts`

    Ts_subseq_isfinite : list
        A list of rolling window `T_subseq_isfinite` for each time series in `Ts`

    Returns
    -------
    bsf_radius : float
        The updated best-so-far radius of the most central consensus motif

    bsf_Ts_idx : int
        The updated index of time series in `Ts` which contains the most central
        consensus motif

    bsf_subseq_idx : int
        The updated subsequence index in the time series `Ts[bsf_Ts_idx]` that contains
        the most central consensus motif
    """
    bsf_nns_radii, bsf_nns_subseq_idx = _aamp_across_series_nearest_neighbors(
        Ts, bsf_Ts_idx, bsf_subseq_idx, m, Ts_squared, Ts_subseq_isfinite
    )
    bsf_nns_mean_radii = bsf_nns_radii.mean()

    candidate_nns_Ts_idx = np.flatnonzero(np.isclose(bsf_nns_radii, bsf_radius))
    candidate_nns_subseq_idx = bsf_nns_subseq_idx[candidate_nns_Ts_idx]

    for Ts_idx, subseq_idx in zip(candidate_nns_Ts_idx, candidate_nns_subseq_idx):
        candidate_nns_radii, _ = _aamp_across_series_nearest_neighbors(
            Ts, Ts_idx, subseq_idx, m, Ts_squared, Ts_subseq_isfinite
        )
        if (
            np.isclose(candidate_nns_radii.max(), bsf_radius)
            and candidate_nns_radii.mean() < bsf_nns_mean_radii
        ):
            bsf_Ts_idx = Ts_idx
            bsf_subseq_idx = subseq_idx
            bsf_nns_mean_radii = candidate_nns_radii.mean()

    return bsf_radius, bsf_Ts_idx, bsf_subseq_idx


def _aamp_ostinato(
    Ts,
    m,
    Ts_squared,
    Ts_subseq_isfinite,
    dask_client=None,
    device_id=None,
    mp_func=aamp,
):
    """
    Find the consensus motif amongst a list of time series

    Parameters
    ----------
    Ts : list
        A list of time series for which to find the consensus motif

    m : int
        Window size

    Ts_squared : list
        A list of rolling window `T_squared` for each time series in `Ts`

    Ts_subseq_isfinite : list
        A list of rolling window `T_subseq_isfinite` for each time series in `Ts`

    dask_client : client, default None
        A Dask Distributed client that is connected to a Dask scheduler and
        Dask workers. Setting up a Dask distributed cluster is beyond the
        scope of this library. Please refer to the Dask Distributed
        documentation.

    device_id : int or list, default None
        The (GPU) device number to use. The default value is `0`. A list of
        valid device ids (int) may also be provided for parallel GPU-STUMP
        computation. A list of all valid device ids can be obtained by
        executing `[device.id for device in numba.cuda.list_devices()]`.

    mp_func : object, default stump
        Specify a custom matrix profile function to use for computing matrix profiles

    Returns
    -------
    bsf_radius : float
        The (best-so-far) Radius of the consensus motif

    bsf_Ts_idx : int
        The time series index in `Ts` which contains the consensus motif

    bsf_subseq_idx : int
        The subsequence index within time series `Ts[bsf_Ts_idx]` the contains the
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
    bsf_radius = np.inf
    bsf_Ts_idx = 0
    bsf_subseq_idx = 0

    partial_mp_func = core._get_partial_mp_func(
        mp_func, dask_client=dask_client, device_id=device_id
    )

    k = len(Ts)
    for j in range(k):
        if j < (k - 1):
            h = j + 1
        else:
            h = 0

        mp = partial_mp_func(Ts[j], m, Ts[h], ignore_trivial=False)
        si = np.argsort(mp[:, 0])
        for q in si:
            Q = Ts[j][q : q + m]
            Q_squared = np.sum(Q * Q)
            radius = mp[q, 0]
            if radius >= bsf_radius:
                break
            for i in range(k):
                if i != j and i != h:
                    if np.any(~np.isfinite(Q)):  # pragma: no cover
                        distance_profile = np.empty(Ts[i].shape[0] - m + 1)
                        distance_profile[:] = np.inf
                    else:
                        QT = core.sliding_dot_product(Ts[j][q : q + m], Ts[i])
                        distance_profile = core._mass_absolute(
                            Q_squared, Ts_squared[i], QT
                        )
                        distance_profile[~Ts_subseq_isfinite[i]] = np.inf
                    radius = np.max((radius, np.min(distance_profile)))
                    if radius >= bsf_radius:
                        break
            if radius < bsf_radius:
                bsf_radius, bsf_Ts_idx, bsf_subseq_idx = radius, j, q

    return bsf_radius, bsf_Ts_idx, bsf_subseq_idx


def aamp_ostinato(Ts, m):
    """
    Find the non-normalized (i.e., without z-normalization) consensus motif of multiple
    time series

    This is a wrapper around the vanilla version of the ostinato algorithm
    which finds the best radius and a helper function that finds the most
    central conserved motif.

    Parameters
    ----------
    Ts : list
        A list of time series for which to find the most central consensus motif

    m : int
        Window size

    Returns
    -------
    central_radius : float
        Radius of the most central consensus motif

    central_Ts_idx : int
        The time series index in `Ts` which contains the most central consensus motif

    central_subseq_idx : int
        The subsequence index within time series `Ts[central_motif_Ts_idx]` the contains
        most central consensus motif

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
    Ts_squared = [None] * len(Ts)
    Ts_subseq_isfinite = [None] * len(Ts)
    for i, T in enumerate(Ts):
        Ts[i], Ts_subseq_isfinite[i] = core.preprocess_non_normalized(T, m)
        Ts_squared[i] = np.sum(core.rolling_window(Ts[i] * Ts[i], m), axis=1)

    bsf_radius, bsf_Ts_idx, bsf_subseq_idx = _aamp_ostinato(
        Ts, m, Ts_squared, Ts_subseq_isfinite
    )

    (central_radius, central_Ts_idx, central_subseq_idx,) = _get_aamp_central_motif(
        Ts, bsf_radius, bsf_Ts_idx, bsf_subseq_idx, m, Ts_squared, Ts_subseq_isfinite
    )

    return central_radius, central_Ts_idx, central_subseq_idx


def aamp_ostinatoed(dask_client, Ts, m):
    """
    Find the non-normalized (i.e., without z-normalization) consensus motif of multiple
    time series with a distributed dask cluster

    This is a wrapper around the vanilla version of the ostinato algorithm
    which finds the best radius and a helper function that finds the most
    central conserved motif.

    Parameters
    ----------
    dask_client : client
        A Dask Distributed client that is connected to a Dask scheduler and
        Dask workers. Setting up a Dask distributed cluster is beyond the
        scope of this library. Please refer to the Dask Distributed
        documentation.

    Ts : list
        A list of time series for which to find the most central consensus motif

    m : int
        Window size

    Returns
    -------
    central_radius : float
        Radius of the most central consensus motif

    central_Ts_idx : int
        The time series index in `Ts` which contains the most central consensus motif

    central_subseq_idx : int
        The subsequence index within time series `Ts[central_motif_Ts_idx]` the contains
        most central consensus motif

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
    Ts_squared = [None] * len(Ts)
    Ts_subseq_isfinite = [None] * len(Ts)
    for i, T in enumerate(Ts):
        Ts[i], Ts_subseq_isfinite[i] = core.preprocess_non_normalized(T, m)
        Ts_squared[i] = np.sum(core.rolling_window(Ts[i] * Ts[i], m), axis=1)

    bsf_radius, bsf_Ts_idx, bsf_subseq_idx = _aamp_ostinato(
        Ts, m, Ts_squared, Ts_subseq_isfinite, dask_client=dask_client, mp_func=aamped
    )

    (central_radius, central_Ts_idx, central_subseq_idx,) = _get_aamp_central_motif(
        Ts, bsf_radius, bsf_Ts_idx, bsf_subseq_idx, m, Ts_squared, Ts_subseq_isfinite
    )

    return central_radius, central_Ts_idx, central_subseq_idx
