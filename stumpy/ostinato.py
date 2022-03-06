# STUMPY
# Copyright 2019 TD Ameritrade. Released under the terms of the 3-Clause BSD license.
# STUMPY is a trademark of TD Ameritrade IP Company, Inc. All rights reserved.

import numpy as np

from . import core, stump, stumped
from .aamp_ostinato import aamp_ostinato, aamp_ostinatoed


def _across_series_nearest_neighbors(Ts, Ts_idx, subseq_idx, m, M_Ts, Σ_Ts):
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

    M_Ts : list
        A list of rolling window means for each time series in `Ts`

    Σ_Ts : list
        A list of rolling window standard deviations for each time series in `Ts`

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
    nns_radii = np.zeros(k, dtype=np.float64)
    nns_subseq_idx = np.zeros(k, dtype=np.int64)

    for i in range(k):
        QT = core.sliding_dot_product(Ts[Ts_idx][subseq_idx : subseq_idx + m], Ts[i])
        distance_profile = core._mass(
            Q,
            Ts[i],
            QT,
            M_Ts[Ts_idx][subseq_idx],
            Σ_Ts[Ts_idx][subseq_idx],
            M_Ts[i],
            Σ_Ts[i],
        )
        nns_subseq_idx[i] = np.argmin(distance_profile)
        nns_radii[i] = distance_profile[nns_subseq_idx[i]]

    return nns_radii, nns_subseq_idx


def _get_central_motif(Ts, bsf_radius, bsf_Ts_idx, bsf_subseq_idx, m, M_Ts, Σ_Ts):
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

    M_Ts : list
        A list of rolling window means for each time series in `Ts`

    Σ_Ts : list
        A list of rolling window standard deviations for each time series in `Ts`

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
    bsf_nns_radii, bsf_nns_subseq_idx = _across_series_nearest_neighbors(
        Ts, bsf_Ts_idx, bsf_subseq_idx, m, M_Ts, Σ_Ts
    )
    bsf_nns_mean_radii = bsf_nns_radii.mean()

    candidate_nns_Ts_idx = np.flatnonzero(np.isclose(bsf_nns_radii, bsf_radius))
    candidate_nns_subseq_idx = bsf_nns_subseq_idx[candidate_nns_Ts_idx]

    for Ts_idx, subseq_idx in zip(candidate_nns_Ts_idx, candidate_nns_subseq_idx):
        candidate_nns_radii, _ = _across_series_nearest_neighbors(
            Ts, Ts_idx, subseq_idx, m, M_Ts, Σ_Ts
        )
        if (
            np.isclose(candidate_nns_radii.max(), bsf_radius)
            and candidate_nns_radii.mean() < bsf_nns_mean_radii
        ):
            bsf_Ts_idx = Ts_idx
            bsf_subseq_idx = subseq_idx
            bsf_nns_mean_radii = candidate_nns_radii.mean()

    return bsf_radius, bsf_Ts_idx, bsf_subseq_idx


def _ostinato(Ts, m, M_Ts, Σ_Ts, dask_client=None, device_id=None, mp_func=stump):
    """
    Find the consensus motif amongst a list of time series

    Parameters
    ----------
    Ts : list
        A list of time series for which to find the consensus motif

    m : int
        Window size

    M_Ts : list
        A list of rolling window means for each time series in `Ts`

    Σ_Ts : list
        A list of rolling window standard deviations for each time series in `Ts`

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
                bsf_radius, bsf_Ts_idx, bsf_subseq_idx = radius, j, q

    return bsf_radius, bsf_Ts_idx, bsf_subseq_idx


@core.non_normalized(aamp_ostinato)
def ostinato(Ts, m, normalize=True, p=2.0):
    """
    Find the z-normalized consensus motif of multiple time series

    This is a wrapper around the vanilla version of the ostinato algorithm
    which finds the best radius and a helper function that finds the most
    central conserved motif.

    Parameters
    ----------
    Ts : list
        A list of time series for which to find the most central consensus motif

    m : int
        Window size

    normalize : bool, default True
        When set to `True`, this z-normalizes subsequences prior to computing distances.
        Otherwise, this function gets re-routed to its complementary non-normalized
        equivalent set in the `@core.non_normalized` function decorator.

    p : float, default 2.0
        The p-norm to apply for computing the Minkowski distance. This parameter is
        ignored when `normalize == True`.

    Returns
    -------
    central_radius : float
        Radius of the most central consensus motif

    central_Ts_idx : int
        The time series index in `Ts` which contains the most central consensus motif

    central_subseq_idx : int
        The subsequence index within time series `Ts[central_motif_Ts_idx]` the contains
        most central consensus motif

    See Also
    --------
    stumpy.ostinatoed : Find the z-normalized consensus motif of multiple time series
        with a distributed dask cluster
    stumpy.gpu_ostinato : Find the z-normalized consensus motif of multiple time series
        with one or more GPU devices

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

    Examples
    --------
    >>> stumpy.ostinato(
    ...     [np.array([584., -11., 23., 79., 1001., 0., 19.]),
    ...      np.array([600., -10., 23., 17.]),
    ...      np.array([  1.,   9.,  6.,  0.])],
    ...     m=3)
    (1.2370237678153826, 0, 4)
    """
    M_Ts = [None] * len(Ts)
    Σ_Ts = [None] * len(Ts)
    for i, T in enumerate(Ts):
        Ts[i], M_Ts[i], Σ_Ts[i] = core.preprocess(T, m)

    bsf_radius, bsf_Ts_idx, bsf_subseq_idx = _ostinato(Ts, m, M_Ts, Σ_Ts)

    (
        central_radius,
        central_Ts_idx,
        central_subseq_idx,
    ) = _get_central_motif(Ts, bsf_radius, bsf_Ts_idx, bsf_subseq_idx, m, M_Ts, Σ_Ts)

    return central_radius, central_Ts_idx, central_subseq_idx


@core.non_normalized(aamp_ostinatoed)
def ostinatoed(dask_client, Ts, m, normalize=True, p=2.0):
    """
    Find the z-normalized consensus motif of multiple time series with a distributed
    dask cluster

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

    normalize : bool, default True
        When set to `True`, this z-normalizes subsequences prior to computing distances.
        Otherwise, this function gets re-routed to its complementary non-normalized
        equivalent set in the `@core.non_normalized` function decorator.

    p : float, default 2.0
        The p-norm to apply for computing the Minkowski distance. This parameter is
        ignored when `normalize == True`.

    Returns
    -------
    central_radius : float
        Radius of the most central consensus motif

    central_Ts_idx : int
        The time series index in `Ts` which contains the most central consensus motif

    central_subseq_idx : int
        The subsequence index within time series `Ts[central_motif_Ts_idx]` the contains
        most central consensus motif

    See Also
    --------
    stumpy.ostinato : Find the z-normalized consensus motif of multiple time series
    stumpy.gpu_ostinato : Find the z-normalized consensus motif of multiple time series
        with one or more GPU devices

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

    >>> from dask.distributed import Client
    >>> if __name__ == "__main__":
    ...     dask_client = Client()
    ...     stumpy.ostinatoed(
    ...         dask_client,
    ...         [np.array([584., -11., 23., 79., 1001., 0., 19.]),
    ...          np.array([600., -10., 23., 17.]),
    ...          np.array([  1.,   9.,  6.,  0.])],
    ...         m=3)
    (1.2370237678153826, 0, 4)
    """
    M_Ts = [None] * len(Ts)
    Σ_Ts = [None] * len(Ts)
    for i, T in enumerate(Ts):
        Ts[i], M_Ts[i], Σ_Ts[i] = core.preprocess(T, m)

    bsf_radius, bsf_Ts_idx, bsf_subseq_idx = _ostinato(
        Ts, m, M_Ts, Σ_Ts, dask_client=dask_client, mp_func=stumped
    )

    (
        central_radius,
        central_Ts_idx,
        central_subseq_idx,
    ) = _get_central_motif(Ts, bsf_radius, bsf_Ts_idx, bsf_subseq_idx, m, M_Ts, Σ_Ts)

    return central_radius, central_Ts_idx, central_subseq_idx
