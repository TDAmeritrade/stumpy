# STUMPY
# Copyright 2019 TD Ameritrade. Released under the terms of the 3-Clause BSD license.
# STUMPY is a trademark of TD Ameritrade IP Company, Inc. All rights reserved.

import logging

import numpy as np

from .maamp import _maamp, _get_first_maamp_profile
from .mstump import _get_multi_QT, _preprocess_include
from . import core, config

logger = logging.getLogger(__name__)


def maamped(dask_client, T, m, include=None, discords=False):
    """
    Compute the multi-dimensional non-normalized (i.e., without z-normalization) matrix
    profile with a distributed dask cluster

    This is a highly distributed implementation around the Numba JIT-compiled
    parallelized `_maamp` function which computes the multi-dimensional matrix
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

    include : list, ndarray, default None
        A list of (zero-based) indices corresponding to the dimensions in `T` that
        must be included in the constrained multidimensional motif search.
        For more information, see Section IV D in:

        `DOI: 10.1109/ICDM.2017.66 \
        <https://www.cs.ucr.edu/~eamonn/Motif_Discovery_ICDM.pdf>`__

    discords : bool, default False
        When set to `True`, this reverses the distance matrix which results in a
        multi-dimensional matrix profile that favors larger matrix profile values
        (i.e., discords) rather than smaller values (i.e., motifs). Note that indices
        in `include` are still maintained and respected.

    Returns
    -------
    P : ndarray
        The multi-dimensional matrix profile. Each row of the array corresponds
        to each matrix profile for a given dimension (i.e., the first row is
        the 1-D matrix profile and the second row is the 2-D matrix profile).

    I : ndarray
        The multi-dimensional matrix profile index where each row of the array
        corresponds to each matrix profile index for a given dimension.

    Notes
    -----
    `DOI: 10.1109/ICDM.2017.66 \
    <https://www.cs.ucr.edu/~eamonn/Motif_Discovery_ICDM.pdf>`__

    See mSTAMP Algorithm
    """
    T_A = T
    T_B = T_A

    T_A, T_A_subseq_isfinite = core.preprocess_non_normalized(T_A, m)
    T_B, T_B_subseq_isfinite = core.preprocess_non_normalized(T_B, m)
    T_A_subseq_squared = np.sum(core.rolling_window(T_A * T_A, m), axis=2)
    T_B_subseq_squared = np.sum(core.rolling_window(T_B * T_B, m), axis=2)

    if T_A.ndim <= 1:  # pragma: no cover
        err = f"T is {T_A.ndim}-dimensional and must be at least 1-dimensional"
        raise ValueError(f"{err}")

    core.check_window_size(m, max_size=min(T_A.shape[1], T_B.shape[1]))

    if include is not None:
        include = _preprocess_include(include)

    d, n = T_B.shape
    k = n - m + 1
    excl_zone = int(
        np.ceil(m / config.STUMPY_EXCL_ZONE_DENOM)
    )  # See Definition 3 and Figure 3

    P = np.empty((d, k), dtype="float64")
    I = np.empty((d, k), dtype="int64")

    hosts = list(dask_client.ncores().keys())
    nworkers = len(hosts)

    step = 1 + k // nworkers

    for i, start in enumerate(range(0, k, step)):
        P[:, start], I[:, start] = _get_first_maamp_profile(
            start,
            T_A,
            T_B,
            m,
            excl_zone,
            T_B_subseq_isfinite,
            include,
            discords,
        )

    # Scatter data to Dask cluster
    T_A_future = dask_client.scatter(T_A, broadcast=True, hash=False)
    T_A_subseq_isfinite_future = dask_client.scatter(
        T_A_subseq_isfinite, broadcast=True, hash=False
    )
    T_B_subseq_isfinite_future = dask_client.scatter(
        T_B_subseq_isfinite, broadcast=True, hash=False
    )
    T_A_subseq_squared_future = dask_client.scatter(
        T_A_subseq_squared, broadcast=True, hash=False
    )
    T_B_subseq_squared_future = dask_client.scatter(
        T_B_subseq_squared, broadcast=True, hash=False
    )

    QT_futures = []
    QT_first_futures = []

    for i, start in enumerate(range(0, k, step)):
        QT, QT_first = _get_multi_QT(start, T_A, m)

        QT_future = dask_client.scatter(QT, workers=[hosts[i]], hash=False)
        QT_first_future = dask_client.scatter(QT_first, workers=[hosts[i]], hash=False)

        QT_futures.append(QT_future)
        QT_first_futures.append(QT_first_future)

    futures = []
    for i, start in enumerate(range(0, k, step)):
        stop = min(k, start + step)

        futures.append(
            dask_client.submit(
                _maamp,
                T_A_future,
                m,
                stop,
                excl_zone,
                T_A_subseq_isfinite_future,
                T_B_subseq_isfinite_future,
                T_A_subseq_squared_future,
                T_B_subseq_squared_future,
                QT_futures[i],
                QT_first_futures[i],
                k,
                start + 1,
                include,
                discords,
            )
        )

    results = dask_client.gather(futures)
    for i, start in enumerate(range(0, k, step)):
        stop = min(k, start + step)
        P[:, start + 1 : stop], I[:, start + 1 : stop] = results[i]

    # Delete data from Dask cluster
    dask_client.cancel(T_A_future)
    dask_client.cancel(T_A_subseq_isfinite_future)
    dask_client.cancel(T_B_subseq_isfinite_future)
    dask_client.cancel(T_A_subseq_squared_future)
    dask_client.cancel(T_B_subseq_squared_future)
    for QT_future in QT_futures:
        dask_client.cancel(QT_future)
    for QT_first_future in QT_first_futures:
        dask_client.cancel(QT_first_future)
    for future in futures:
        dask_client.cancel(future)

    return P, I
