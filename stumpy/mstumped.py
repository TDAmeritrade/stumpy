# STUMPY
# Copyright 2019 TD Ameritrade. Released under the terms of the 3-Clause BSD license.
# STUMPY is a trademark of TD Ameritrade IP Company, Inc. All rights reserved.

import logging

import numpy as np

from . import _mstump, _get_first_mstump_profile, _get_multi_QT
from . import core

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
        The multi-dimensional matrix profile. Each column of the array corresponds
        to each matrix profile for a given dimension (i.e., the first column is
        the 1-D matrix profile and the second column is the 2-D matrix profile).

    I : ndarray
        The multi-dimensional matrix profile index where each column of the array
        corresponds to each matrix profile index for a given dimension.

    Notes
    -----

    `DOI: 10.1109/ICDM.2017.66 \
    <https://www.cs.ucr.edu/~eamonn/Motif_Discovery_ICDM.pdf>`__

    See mSTAMP Algorithm
    """

    T_A = np.asarray(core.transpose_dataframe(T)).copy()
    T_B = T_A.copy()

    T_A[np.isinf(T_A)] = np.nan
    T_B[np.isinf(T_B)] = np.nan

    core.check_dtype(T_A)
    if T_A.ndim <= 1:  # pragma: no cover
        err = f"T is {T_A.ndim}-dimensional and must be at least 1-dimensional"
        raise ValueError(f"{err}")

    core.check_window_size(m)

    d, n = T_A.shape
    k = n - m + 1
    excl_zone = int(np.ceil(m / 4))  # See Definition 3 and Figure 3

    M_T, Σ_T = core.compute_mean_std(T_A, m)
    μ_Q, σ_Q = core.compute_mean_std(T_B, m)

    T_A[np.isnan(T_A)] = 0

    P = np.empty((d, k), dtype="float64")
    I = np.empty((d, k), dtype="int64")

    hosts = list(dask_client.ncores().keys())
    nworkers = len(hosts)

    step = 1 + k // nworkers

    for i, start in enumerate(range(0, k, step)):
        P[:, start], I[:, start] = _get_first_mstump_profile(
            start, T_A, T_B, m, excl_zone, M_T, Σ_T
        )

    T_B[np.isnan(T_B)] = 0

    # Scatter data to Dask cluster
    T_A_future = dask_client.scatter(T_A, broadcast=True)
    M_T_future = dask_client.scatter(M_T, broadcast=True)
    Σ_T_future = dask_client.scatter(Σ_T, broadcast=True)
    μ_Q_future = dask_client.scatter(μ_Q, broadcast=True)
    σ_Q_future = dask_client.scatter(σ_Q, broadcast=True)

    QT_futures = []
    QT_first_futures = []

    for i, start in enumerate(range(0, k, step)):
        QT, QT_first = _get_multi_QT(start, T_A, m)

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
                T_A_future,
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
                start + 1,
            )
        )

    results = dask_client.gather(futures)
    for i, start in enumerate(range(0, k, step)):
        stop = min(k, start + step)
        P[:, start + 1 : stop], I[:, start + 1 : stop] = results[i]

    return P.T, I.T
