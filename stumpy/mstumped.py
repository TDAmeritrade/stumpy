# STUMPY
# Copyright 2019 TD Ameritrade. Released under the terms of the 3-Clause BSD license.
# STUMPY is a trademark of TD Ameritrade IP Company, Inc. All rights reserved.

import logging

import numpy as np

from .mstump import (
    _mstump,
    _get_first_mstump_profile,
    _get_multi_QT,
    _preprocess_include,
)
from . import core, config
from .maamped import maamped

logger = logging.getLogger(__name__)


@core.non_normalized(maamped)
def mstumped(dask_client, T, m, include=None, discords=False, normalize=True):
    """
    Compute the multi-dimensional z-normalized matrix profile with a distributed
    dask cluster

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

    T : numpy.ndarray
        The time series or sequence for which to compute the multi-dimensional
        matrix profile. Each row in `T` represents data from a different
        dimension while each column in `T` represents data from the same
        dimension.

    m : int
        Window size

    include : list, numpy.ndarray, default None
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

    normalize : bool, default True
        When set to `True`, this z-normalizes subsequences prior to computing distances.
        Otherwise, this function gets re-routed to its complementary non-normalized
        equivalent set in the `@core.non_normalized` function decorator.

    Returns
    -------
    P : numpy.ndarray
        The multi-dimensional matrix profile. Each row of the array corresponds
        to each matrix profile for a given dimension (i.e., the first row is
        the 1-D matrix profile and the second row is the 2-D matrix profile).

    I : numpy.ndarray
        The multi-dimensional matrix profile index where each row of the array
        corresponds to each matrix profile index for a given dimension.

    See Also
    --------
    stumpy.mstump : Compute the multi-dimensional z-normalized matrix profile
    stumpy.subspace : Compute the k-dimensional matrix profile subspace for a given
        subsequence index and its nearest neighbor index
    stumpy.mdl : Compute the number of bits needed to compress one array with another
        using the minimum description length (MDL)

    Notes
    -----
    `DOI: 10.1109/ICDM.2017.66 \
    <https://www.cs.ucr.edu/~eamonn/Motif_Discovery_ICDM.pdf>`__

    See mSTAMP Algorithm

    Examples
    --------
    >>> from dask.distributed import Client
    >>> if __name__ == "__main__":
    ...     dask_client = Client()
    ...     stumpy.mstumped(
    ...         np.array([[584., -11., 23., 79., 1001., 0., -19.],
    ...                   [  1.,   2.,  4.,  8.,   16., 0.,  32.]]),
    ...         m=3)
    (array([[0.        , 1.43947142, 0.        , 2.69407392, 0.11633857],
            [0.777905  , 2.36179922, 1.50004632, 2.92246722, 0.777905  ]]),
     array([[2, 4, 0, 1, 0],
            [4, 4, 0, 1, 0]]))
    """
    T_A = T
    T_B = T_A

    T_A, M_T, Σ_T = core.preprocess(T_A, m)
    T_B, μ_Q, σ_Q = core.preprocess(T_B, m)

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

    P = np.empty((d, k), dtype=np.float64)
    I = np.empty((d, k), dtype=np.int64)

    hosts = list(dask_client.ncores().keys())
    nworkers = len(hosts)

    step = 1 + k // nworkers

    for i, start in enumerate(range(0, k, step)):
        P[:, start], I[:, start] = _get_first_mstump_profile(
            start, T_A, T_B, m, excl_zone, M_T, Σ_T, μ_Q, σ_Q, include, discords
        )

    # Scatter data to Dask cluster
    T_A_future = dask_client.scatter(T_A, broadcast=True, hash=False)
    M_T_future = dask_client.scatter(M_T, broadcast=True, hash=False)
    Σ_T_future = dask_client.scatter(Σ_T, broadcast=True, hash=False)
    μ_Q_future = dask_client.scatter(μ_Q, broadcast=True, hash=False)
    σ_Q_future = dask_client.scatter(σ_Q, broadcast=True, hash=False)

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
    dask_client.cancel(M_T_future)
    dask_client.cancel(Σ_T_future)
    dask_client.cancel(μ_Q_future)
    dask_client.cancel(σ_Q_future)
    for QT_future in QT_futures:
        dask_client.cancel(QT_future)
    for QT_first_future in QT_first_futures:
        dask_client.cancel(QT_first_future)
    for future in futures:
        dask_client.cancel(future)

    return P, I
