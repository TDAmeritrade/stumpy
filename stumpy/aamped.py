# STUMPY
# Copyright 2019 TD Ameritrade. Released under the terms of the 3-Clause BSD license.
# STUMPY is a trademark of TD Ameritrade IP Company, Inc. All rights reserved.

import logging

import numpy as np

from . import core, _aamp

logger = logging.getLogger(__name__)


def aamped(dask_client, T_A, m, T_B=None, ignore_trivial=True):
    """
    Compute the matrix profile with parallelized AAMP, which uses non-normalized
    Euclidean distances for computing a matrix profile.

    This is highly distributed implementation around the Numba JIT-compiled
    parallelized `_aamp` function which computes the non-normalized matrix profile
    according to AAMP.

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
        consists of the matrix profile indices.

    Notes
    -----
    `arXiv:1901.05708 \
    <https://arxiv.org/pdf/1901.05708.pdf>`__

    See Algorithm 1

    Note that we have extended this algorithm for AB-joins as well.
    """
    T_A = np.asarray(T_A)
    T_A = T_A.copy()

    if T_B is None:
        T_B = T_A.copy()
        ignore_trivial = True
    else:
        T_B = np.asarray(T_B)
        T_B = T_B.copy()
        ignore_trivial = False

    if T_A.ndim != 1:  # pragma: no cover
        raise ValueError(f"T_A is {T_A.ndim}-dimensional and must be 1-dimensional. ")

    if T_B.ndim != 1:  # pragma: no cover
        raise ValueError(f"T_B is {T_B.ndim}-dimensional and must be 1-dimensional. ")

    core.check_dtype(T_A)
    core.check_dtype(T_B)

    core.check_window_size(m)

    if ignore_trivial is False and core.are_arrays_equal(T_A, T_B):  # pragma: no cover
        logger.warning("Arrays T_A, T_B are equal, which implies a self-join.")
        logger.warning("Try setting `ignore_trivial = True`.")

    if ignore_trivial and core.are_arrays_equal(T_A, T_B) is False:  # pragma: no cover
        logger.warning("Arrays T_A, T_B are not equal, which implies an AB-join.")
        logger.warning("Try setting `ignore_trivial = False`.")

    T_A[np.isinf(T_A)] = np.nan
    T_B[np.isinf(T_B)] = np.nan

    T_A_subseq_isfinite = np.all(np.isfinite(core.rolling_window(T_A, m)), axis=1)
    T_B_subseq_isfinite = np.all(np.isfinite(core.rolling_window(T_B, m)), axis=1)

    T_A[np.isnan(T_A)] = 0
    T_B[np.isnan(T_B)] = 0

    n_A = T_A.shape[0]
    n_B = T_B.shape[0]
    l = n_B - m + 1

    excl_zone = int(np.ceil(m / 4))
    out = np.empty((l, 2), dtype=object)

    hosts = list(dask_client.ncores().keys())
    nworkers = len(hosts)

    if ignore_trivial:
        diags = np.arange(excl_zone + 1, n_B - m + 1)
    else:
        diags = np.arange(-(n_B - m + 1) + 1, n_A - m + 1)

    ndist_counts = core._count_diagonal_ndist(diags, m, n_A, n_B)
    diags_ranges = core._get_array_ranges(ndist_counts, nworkers)
    diags_ranges += diags[0]

    # Scatter data to Dask cluster
    T_A_future = dask_client.scatter(T_A, broadcast=True)
    T_B_future = dask_client.scatter(T_B, broadcast=True)
    T_A_subseq_isfinite_future = dask_client.scatter(
        T_A_subseq_isfinite, broadcast=True
    )
    T_B_subseq_isfinite_future = dask_client.scatter(
        T_B_subseq_isfinite, broadcast=True
    )

    diags_futures = []
    for i, host in enumerate(hosts):
        diags_future = dask_client.scatter(
            np.arange(diags_ranges[i, 0], diags_ranges[i, 1]), workers=[host]
        )
        diags_futures.append(diags_future)

    futures = []
    for i in range(len(hosts)):
        futures.append(
            dask_client.submit(
                _aamp,
                T_A_future,
                T_B_future,
                m,
                T_A_subseq_isfinite_future,
                T_B_subseq_isfinite_future,
                diags_futures[i],
                ignore_trivial,
            )
        )

    results = dask_client.gather(futures)
    profile, indices = results[0]
    for i in range(1, len(hosts)):
        P, I = results[i]
        for col in range(P.shape[0]):  # pragma: no cover
            if P[col] < profile[col]:
                profile[col] = P[col]
                indices[col] = I[col]

    out[:, 0] = profile
    out[:, 1] = indices

    threshold = 10e-6
    if core.are_distances_too_small(out[:, 0], threshold=threshold):  # pragma: no cover
        logger.warning(f"A large number of values are smaller than {threshold}.")
        logger.warning("For a self-join, try setting `ignore_trivial = True`.")

    return out
