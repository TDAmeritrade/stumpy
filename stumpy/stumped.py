# STUMPY
# Copyright 2019 TD Ameritrade. Released under the terms of the 3-Clause BSD license.
# STUMPY is a trademark of TD Ameritrade IP Company, Inc. All rights reserved.

import logging

import numpy as np

from . import core, _stump, _get_first_stump_profile, _get_QT

logger = logging.getLogger(__name__)


def stumped(dask_client, T_A, m, T_B=None, ignore_trivial=True):
    """
    Compute the matrix profile with parallelized and distributed STOMP

    This is highly distributed implementation around the Numba JIT-compiled
    parallelized `_stump` function which computes the matrix profile according
    to STOMP.

    Parameters
    ----------
    dask_client : client
        A Dask Distributed client that is connected to a Dask scheduler and
        Dask workers. Setting up a Dask distributed cluster is beyond the
        scope of this library. Please refer to the Dask Distributed
        documentation.

    T_A : ndarray
        The time series or sequence for which to compute the matrix profile

    m : int
        Window size

    T_B : ndarray
        The time series or sequence that contain your query subsequences
        of interest. Default is `None` which corresponds to a self-join.

    ignore_trivial : bool
        Set to `True` if this is a self-join. Otherwise, for AB-join, set this
        to `False`. Default is `True`.

    Returns
    -------
    out : ndarray
        The first column consists of the matrix profile, the second column
        consists of the matrix profile indices, the third column consists of
        the left matrix profile indices, and the fourth column consists of
        the right matrix profile indices.

    Notes
    -----

    `DOI: 10.1109/ICDM.2016.0085 \
    <https://www.cs.ucr.edu/~eamonn/STOMP_GPU_final_submission_camera_ready.pdf>`__

    See Table II

    This is a Dask distributed implementation of stump that scales
    across multiple servers and is a convenience wrapper around the
    parallelized `stump._stump` function

    Timeseries, T_B, will be annotated with the distance location
    (or index) of all its subsequences in another times series, T_A.

    Return: For every subsequence, Q, in T_B, you will get a distance
    and index for the closest subsequence in T_A. Thus, the array
    returned will have length T_B.shape[0]-m+1. Additionally, the
    left and right matrix profiles are also returned.

    Note: Unlike in the Table II where T_A.shape is expected to be equal
    to T_B.shape, this implementation is generalized so that the shapes of
    T_A and T_B can be different. In the case where T_A.shape == T_B.shape,
    then our algorithm reduces down to the same algorithm found in Table II.

    Additionally, unlike STAMP where the exclusion zone is m/2, the default
    exclusion zone for STOMP is m/4 (See Definition 3 and Figure 3).

    For self-joins, set `ignore_trivial = True` in order to avoid the
    trivial match.

    Note that left and right matrix profiles are only available for self-joins.
    """

    T_A = np.asarray(T_A)
    if T_A.ndim != 1:  # pragma: no cover
        raise ValueError(
            f"T_A is {T_A.ndim}-dimensional and must be 1-dimensional. "
            "For multidimensional STUMP use `stumpy.mstump` or `stumpy.mstumped`"
        )
    n = T_A.shape[0]

    T_A = T_A.copy()
    T_A[np.isinf(T_A)] = np.nan
    core.check_dtype(T_A)

    if T_B is None:
        T_B = T_A
        ignore_trivial = True

    T_B = np.asarray(T_B)
    if T_B.ndim != 1:  # pragma: no cover
        raise ValueError(
            f"T_B is {T_B.ndim}-dimensional and must be 1-dimensional. "
            "For multidimensional STUMP use `stumpy.mstump` or `stumpy.mstumped`"
        )
    T_B = T_B.copy()
    T_B[np.isinf(T_B)] = np.nan
    core.check_dtype(T_B)

    core.check_window_size(m)

    if ignore_trivial is False and core.are_arrays_equal(T_A, T_B):  # pragma: no cover
        logger.warning("Arrays T_A, T_B are equal, which implies a self-join.")
        logger.warning("Try setting `ignore_trivial = True`.")

    if ignore_trivial and core.are_arrays_equal(T_A, T_B) is False:  # pragma: no cover
        logger.warning("Arrays T_A, T_B are not equal, which implies an AB-join.")
        logger.warning("Try setting `ignore_trivial = False`.")

    n = T_B.shape[0]
    k = T_A.shape[0] - m + 1
    l = n - m + 1
    excl_zone = int(np.ceil(m / 4))  # See Definition 3 and Figure 3

    M_T, Σ_T = core.compute_mean_std(T_A, m)
    μ_Q, σ_Q = core.compute_mean_std(T_B, m)

    T_A[np.isnan(T_A)] = 0

    out = np.empty((l, 4), dtype=object)
    profile = np.empty((l,), dtype="float64")
    indices = np.empty((l, 3), dtype="int64")

    hosts = list(dask_client.ncores().keys())
    nworkers = len(hosts)

    step = 1 + l // nworkers
    QT_futures = []
    QT_first_futures = []
    for i, start in enumerate(range(0, l, step)):
        profile[start], indices[start, :] = _get_first_stump_profile(
            start, T_A, T_B, m, excl_zone, M_T, Σ_T, ignore_trivial
        )

    T_B[np.isnan(T_B)] = 0

    # Scatter data to Dask cluster
    T_A_future = dask_client.scatter(T_A, broadcast=True)
    T_B_future = dask_client.scatter(T_B, broadcast=True)
    M_T_future = dask_client.scatter(M_T, broadcast=True)
    Σ_T_future = dask_client.scatter(Σ_T, broadcast=True)
    μ_Q_future = dask_client.scatter(μ_Q, broadcast=True)
    σ_Q_future = dask_client.scatter(σ_Q, broadcast=True)

    for i, start in enumerate(range(0, l, step)):
        QT, QT_first = _get_QT(start, T_A, T_B, m)

        QT_future = dask_client.scatter(QT, workers=[hosts[i]])
        QT_first_future = dask_client.scatter(QT_first, workers=[hosts[i]])

        QT_futures.append(QT_future)
        QT_first_futures.append(QT_first_future)

    futures = []
    for i, start in enumerate(range(0, l, step)):
        stop = min(l, start + step)

        futures.append(
            dask_client.submit(
                _stump,
                T_A_future,
                T_B_future,
                m,
                stop,
                excl_zone,
                M_T_future,
                Σ_T_future,
                QT_futures[i],
                QT_first_futures[i],
                μ_Q_future,
                σ_Q_future,
                k,
                ignore_trivial,
                start + 1,
            )
        )

    results = dask_client.gather(futures)
    for i, start in enumerate(range(0, l, step)):
        stop = min(l, start + step)
        profile[start + 1 : stop], indices[start + 1 : stop, :] = results[i]

    out[:, 0] = profile
    out[:, 1:4] = indices

    threshold = 10e-6
    if core.are_distances_too_small(out[:, 0], threshold=threshold):  # pragma: no cover
        logger.warning(f"A large number of values are smaller than {threshold}.")
        logger.warning("For a self-join, try setting `ignore_trivial = True`.")

    return out
