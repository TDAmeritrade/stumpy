# STUMPY
# Copyright 2019 TD Ameritrade. Released under the terms of the 3-Clause BSD license.
# STUMPY is a trademark of TD Ameritrade IP Company, Inc. All rights reserved.

import logging

import numpy as np

from . import core, _stump

logger = logging.getLogger(__name__)


def stumped(dask_client, T_A, m, T_B=None, ignore_trivial=True):
    """
    Compute the z-normalized matrix profile with a distributed dask cluster

    This is a highly distributed implementation around the Numba JIT-compiled
    parallelized `_stump` function which computes the matrix profile according
    to STOMPopt with Pearson correlations.

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
    `DOI: 10.1007/s10115-017-1138-x \
    <https://www.cs.ucr.edu/~eamonn/ten_quadrillion.pdf>`__

    See Section 4.5

    The above reference outlines a general approach for traversing the distance
    matrix in a diagonal fashion rather than in a row-wise fashion.

    `DOI: 10.1145/3357223.3362721 \
    <https://www.cs.ucr.edu/~eamonn/public/GPU_Matrix_profile_VLDB_30DraftOnly.pdf>`__

    See Section 3.1 and Section 3.3

    The above reference outlines the use of the Pearson correlation via Welford's
    centered sum-of-products along each diagonal of the distance matrix in place of the
    sliding window dot product found in the original STOMP method.

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
    if T_B is None:
        T_B = T_A
        ignore_trivial = True

    (
        T_A,
        M_T,
        Σ_T_inverse,
        M_T_m_1,
        T_A_subseq_isfinite,
        T_A_subseq_isconstant,
    ) = core.preprocess_diagonal(T_A, m)

    (
        T_B,
        μ_Q,
        σ_Q_inverse,
        μ_Q_m_1,
        T_B_subseq_isfinite,
        T_B_subseq_isconstant,
    ) = core.preprocess_diagonal(T_B, m)

    if T_A.ndim != 1:  # pragma: no cover
        raise ValueError(
            f"T_A is {T_A.ndim}-dimensional and must be 1-dimensional. "
            "For multidimensional STUMP use `stumpy.mstump` or `stumpy.mstumped`"
        )

    if T_B.ndim != 1:  # pragma: no cover
        raise ValueError(
            f"T_B is {T_B.ndim}-dimensional and must be 1-dimensional. "
            "For multidimensional STUMP use `stumpy.mstump` or `stumpy.mstumped`"
        )

    core.check_dtype(T_A)
    core.check_dtype(T_B)

    core.check_window_size(m)

    if ignore_trivial is False and core.are_arrays_equal(T_A, T_B):  # pragma: no cover
        logger.warning("Arrays T_A, T_B are equal, which implies a self-join.")
        logger.warning("Try setting `ignore_trivial = True`.")

    if ignore_trivial and core.are_arrays_equal(T_A, T_B) is False:  # pragma: no cover
        logger.warning("Arrays T_A, T_B are not equal, which implies an AB-join.")
        logger.warning("Try setting `ignore_trivial = False`.")

    n_A = T_A.shape[0]
    n_B = T_B.shape[0]
    l = n_B - m + 1

    excl_zone = int(np.ceil(m / 4))
    out = np.empty((l, 4), dtype=object)

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
    M_T_future = dask_client.scatter(M_T, broadcast=True)
    μ_Q_future = dask_client.scatter(μ_Q, broadcast=True)
    Σ_T_inverse_future = dask_client.scatter(Σ_T_inverse, broadcast=True)
    σ_Q_inverse_future = dask_client.scatter(σ_Q_inverse, broadcast=True)
    M_T_m_1_future = dask_client.scatter(M_T_m_1, broadcast=True)
    μ_Q_m_1_future = dask_client.scatter(μ_Q_m_1, broadcast=True)
    T_A_subseq_isfinite_future = dask_client.scatter(
        T_A_subseq_isfinite, broadcast=True
    )
    T_B_subseq_isfinite_future = dask_client.scatter(
        T_B_subseq_isfinite, broadcast=True
    )
    T_A_subseq_isconstant_future = dask_client.scatter(
        T_A_subseq_isconstant, broadcast=True
    )
    T_B_subseq_isconstant_future = dask_client.scatter(
        T_B_subseq_isconstant, broadcast=True
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
                _stump,
                T_A_future,
                T_B_future,
                m,
                M_T_future,
                μ_Q_future,
                Σ_T_inverse_future,
                σ_Q_inverse_future,
                M_T_m_1_future,
                μ_Q_m_1_future,
                T_A_subseq_isfinite_future,
                T_B_subseq_isfinite_future,
                T_A_subseq_isconstant_future,
                T_B_subseq_isconstant_future,
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

    threshold = 10e-6
    if core.are_distances_too_small(out[:, 0], threshold=threshold):  # pragma: no cover
        logger.warning(f"A large number of values are smaller than {threshold}.")
        logger.warning("For a self-join, try setting `ignore_trivial = True`.")

    return out
