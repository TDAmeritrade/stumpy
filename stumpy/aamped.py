# STUMPY
# Copyright 2019 TD Ameritrade. Released under the terms of the 3-Clause BSD license.
# STUMPY is a trademark of TD Ameritrade IP Company, Inc. All rights reserved.

import logging

import numpy as np

from . import core, config
from .aamp import _aamp

logger = logging.getLogger(__name__)


def aamped(dask_client, T_A, m, T_B=None, ignore_trivial=True, p=2.0):
    """
    Compute the non-normalized (i.e., without z-normalization) matrix profile

    This is a highly distributed implementation around the Numba JIT-compiled
    parallelized `_aamp` function which computes the non-normalized matrix profile
    according to AAMP.

    Parameters
    ----------
    dask_client : client
        A Dask Distributed client that is connected to a Dask scheduler and
        Dask workers. Setting up a Dask distributed cluster is beyond the
        scope of this library. Please refer to the Dask Distributed
        documentation.

    T_A : numpy.ndarray
        The time series or sequence for which to compute the matrix profile

    m : int
        Window size

    T_B : numpy.ndarray, default None
        The time series or sequence that will be used to annotate T_A. For every
        subsequence in T_A, its nearest neighbor in T_B will be recorded. Default is
        `None` which corresponds to a self-join.

    ignore_trivial : bool, default True
        Set to `True` if this is a self-join. Otherwise, for AB-join, set this
        to `False`. Default is `True`.

    p : float, default 2.0
        The p-norm to apply for computing the Minkowski distance.

    Returns
    -------
    out : numpy.ndarray
        The first column consists of the matrix profile, the second column
        consists of the matrix profile indices.

    Notes
    -----
    `arXiv:1901.05708 \
    <https://arxiv.org/pdf/1901.05708.pdf>`__

    See Algorithm 1

    Note that we have extended this algorithm for AB-joins as well.
    """
    if T_B is None:
        T_B = T_A.copy()
        ignore_trivial = True

    T_A, T_A_subseq_isfinite = core.preprocess_non_normalized(T_A, m)
    T_B, T_B_subseq_isfinite = core.preprocess_non_normalized(T_B, m)

    if T_A.ndim != 1:  # pragma: no cover
        raise ValueError(f"T_A is {T_A.ndim}-dimensional and must be 1-dimensional. ")

    if T_B.ndim != 1:  # pragma: no cover
        raise ValueError(f"T_B is {T_B.ndim}-dimensional and must be 1-dimensional. ")

    core.check_window_size(m, max_size=min(T_A.shape[0], T_B.shape[0]))

    if ignore_trivial is False and core.are_arrays_equal(T_A, T_B):  # pragma: no cover
        logger.warning("Arrays T_A, T_B are equal, which implies a self-join.")
        logger.warning("Try setting `ignore_trivial = True`.")

    if ignore_trivial and core.are_arrays_equal(T_A, T_B) is False:  # pragma: no cover
        logger.warning("Arrays T_A, T_B are not equal, which implies an AB-join.")
        logger.warning("Try setting `ignore_trivial = False`.")

    n_A = T_A.shape[0]
    n_B = T_B.shape[0]
    l = n_A - m + 1

    excl_zone = int(np.ceil(m / config.STUMPY_EXCL_ZONE_DENOM))
    out = np.empty((l, 4), dtype=object)

    hosts = list(dask_client.ncores().keys())
    nworkers = len(hosts)

    if ignore_trivial:
        diags = np.arange(excl_zone + 1, n_A - m + 1, dtype=np.int64)
    else:
        diags = np.arange(-(n_A - m + 1) + 1, n_B - m + 1, dtype=np.int64)

    ndist_counts = core._count_diagonal_ndist(diags, m, n_A, n_B)
    diags_ranges = core._get_array_ranges(ndist_counts, nworkers, False)
    diags_ranges += diags[0]

    # Scatter data to Dask cluster
    T_A_future = dask_client.scatter(T_A, broadcast=True, hash=False)
    T_B_future = dask_client.scatter(T_B, broadcast=True, hash=False)
    T_A_subseq_isfinite_future = dask_client.scatter(
        T_A_subseq_isfinite, broadcast=True, hash=False
    )
    T_B_subseq_isfinite_future = dask_client.scatter(
        T_B_subseq_isfinite, broadcast=True, hash=False
    )

    diags_futures = []
    for i, host in enumerate(hosts):
        diags_future = dask_client.scatter(
            np.arange(diags_ranges[i, 0], diags_ranges[i, 1], dtype=np.int64),
            workers=[host],
            hash=False,
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
                p,
                diags_futures[i],
                ignore_trivial,
            )
        )

    results = dask_client.gather(futures)
    profile, indices = results[0]
    for i in range(1, len(hosts)):
        P, I = results[i]
        for col in range(P.shape[1]):  # pragma: no cover
            cond = P[:, col] < profile[:, col]
            profile[:, col] = np.where(cond, P[:, col], profile[:, col])
            indices[:, col] = np.where(cond, I[:, col], indices[:, col])

    out[:, 0] = profile[:, 0]
    out[:, 1:4] = indices

    # Delete data from Dask cluster
    dask_client.cancel(T_A_future)
    dask_client.cancel(T_B_future)
    dask_client.cancel(T_A_subseq_isfinite_future)
    dask_client.cancel(T_B_subseq_isfinite_future)
    for diags_future in diags_futures:
        dask_client.cancel(diags_future)
    for future in futures:
        dask_client.cancel(future)

    core._check_P(out)

    return out
