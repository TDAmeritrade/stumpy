# STUMPY
# Copyright 2019 TD Ameritrade. Released under the terms of the 3-Clause BSD license.
# STUMPY is a trademark of TD Ameritrade IP Company, Inc. All rights reserved.

import numpy as np
from . import core
from . import _mstump, _get_first_mstump_profile, _get_multi_QT, _multi_compute_mean_std
import logging

logger = logging.getLogger(__name__)


def mstumped(dask_client, T, m):
    """
    Compute the multi-dimensional matrix profile with parallelized and
    distributed mSTOMP

    This is a highly distributed implementation around the Numba JIT-compiled
    parallelized `_mstump` function which computes the multi-dimensional matrix
    profile according to STOMP. Note that only self-joins are supported.

    Parameters
    ----------
    dask_client : client
        A Dask Distributed client that is connected to a Dask scheduler and
        Dask workers. Setting up a Dask distributed cluster is beyond the
        scope of this library. Please refer to the Dask Distributed
        documentation.

    T : ndarray
        The time series or sequence for which to compute the multi-dimensional
        matrix profile. Each row in `T` represents data from a different
        dimension while each column in `T` represents data from the same
        dimension.
    m : int
        Window size

    Returns
    -------
    P : ndarray
        The multi-dimensioanl matrix profile. Each row of the array corresponds
        to each matrix profile for a given dimension (i.e., the first row is the
        1-D matrix profile and the second row is the 2-D matrix profile).
    I : ndarray
        The multi-dimensional matrix profile index where each row of the array
        correspondsto each matrix profile index for a given dimension.

    Notes
    -----

    `DOI: 10.1109/ICDM.2017.66 \
    <https://www.cs.ucr.edu/~eamonn/Motif_Discovery_ICDM.pdf>`__

    See mSTAMP Algorithm
    """

    hosts = list(dask_client.ncores().keys())
    nworkers = len(hosts)

    T = np.asarray(core.transpose_dataframe(T))

    if T.ndim <= 1:  # pragma: no cover
        err = f"T is {T.ndim}-dimensional and must be greater than 1-dimensional"
        raise ValueError(f"{err}")

    core.check_dtype(T)
    core.check_nan(T)
    core.check_window_size(m)

    d = T.shape[0]
    n = T.shape[1]
    k = n - m + 1
    excl_zone = int(np.ceil(m / 4))  # See Definition 3 and Figure 3

    M_T, Σ_T = _multi_compute_mean_std(T, m)
    μ_Q, σ_Q = _multi_compute_mean_std(T, m)

    P = np.empty((nworkers, d, k), dtype="float64")
    D = np.zeros((nworkers, d, k), dtype="float64")
    D_prime = np.zeros((nworkers, k), dtype="float64")
    I = np.ones((nworkers, d, k), dtype="int64") * -1

    # Scatter data to Dask cluster
    T_future = dask_client.scatter(T, broadcast=True)
    M_T_future = dask_client.scatter(M_T, broadcast=True)
    Σ_T_future = dask_client.scatter(Σ_T, broadcast=True)
    μ_Q_future = dask_client.scatter(μ_Q, broadcast=True)
    σ_Q_future = dask_client.scatter(σ_Q, broadcast=True)

    step = 1 + k // nworkers
    QT_futures = []
    QT_first_futures = []
    P_futures = []
    I_futures = []
    D_futures = []
    D_prime_futures = []

    for i, start in enumerate(range(0, k, step)):
        P[i], I[i] = _get_first_mstump_profile(start, T, m, excl_zone, M_T, Σ_T)

        P_future = dask_client.scatter(P[i], workers=[hosts[i]])
        I_future = dask_client.scatter(I[i], workers=[hosts[i]])
        D_future = dask_client.scatter(D[i], workers=[hosts[i]])
        D_prime_future = dask_client.scatter(D_prime[i], workers=[hosts[i]])

        P_futures.append(P_future)
        I_futures.append(I_future)
        D_futures.append(D_future)
        D_prime_futures.append(D_prime_future)

        QT, QT_first = _get_multi_QT(start, T, m)

        QT_future = dask_client.scatter(QT, workers=[hosts[i]])
        QT_first_future = dask_client.scatter(QT_first, workers=[hosts[i]])

        QT_futures.append(QT_future)
        QT_first_futures.append(QT_first_future)

    futures = []
    for i, start in enumerate(range(0, k, step)):
        stop = min(k, start + step)

        futures.append(
            dask_client.submit(
                _mstump,
                T_future,
                m,
                P_futures[i],
                I_futures[i],
                D_futures[i],
                D_prime_futures[i],
                stop,
                excl_zone,
                M_T_future,
                Σ_T_future,
                QT_futures[i],
                QT_first_futures[i],
                μ_Q_future,
                σ_Q_future,
                k,
                start + 1,
            )
        )

    results = dask_client.gather(futures)
    for i, start in enumerate(range(0, k, step)):
        P[i], I[i] = results[i]
        col_mask = P[0] > P[i]
        P[0, col_mask] = P[i, col_mask]
        I[0, col_mask] = I[i, col_mask]

    return P[0], I[0]
