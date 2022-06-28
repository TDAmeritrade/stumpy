# STUMPY
# Copyright 2019 TD Ameritrade. Released under the terms of the 3-Clause BSD license.
# STUMPY is a trademark of TD Ameritrade IP Company, Inc. All rights reserved.

import logging

import numpy as np

from . import core, config
from .stump import _stump
from .aamped import aamped

logger = logging.getLogger(__name__)


@core.non_normalized(aamped)
def stumped(dask_client, T_A, m, T_B=None, ignore_trivial=True, normalize=True, p=2.0):
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

    normalize : bool, default True
        When set to `True`, this z-normalizes subsequences prior to computing distances.
        Otherwise, this function gets re-routed to its complementary non-normalized
        equivalent set in the `@core.non_normalized` function decorator.

    p : float, default 2.0
        The p-norm to apply for computing the Minkowski distance. This parameter is
        ignored when `normalize == True`.

    Returns
    -------
    out : numpy.ndarray
        The first column consists of the matrix profile, the second column
        consists of the matrix profile indices, the third column consists of
        the left matrix profile indices, and the fourth column consists of
        the right matrix profile indices.

    See Also
    --------
    stumpy.stump : Compute the z-normalized matrix profile
        cluster
    stumpy.gpu_stump : Compute the z-normalized matrix profile with one or more GPU
        devices
    stumpy.scrump : Compute an approximate z-normalized matrix profile

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

    Timeseries, T_A, will be annotated with the distance location
    (or index) of all its subsequences in another times series, T_B.

    Return: For every subsequence, Q, in T_A, you will get a distance
    and index for the closest subsequence in T_B. Thus, the array
    returned will have length T_A.shape[0]-m+1. Additionally, the
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

    Examples
    --------
    >>> from dask.distributed import Client
    >>> if __name__ == "__main__":
    ...     dask_client = Client()
    ...     stumpy.stumped(
    ...         dask_client,
    ...         np.array([584., -11., 23., 79., 1001., 0., -19.]),
    ...         m=3)
    array([[0.11633857113691416, 4, -1, 4],
           [2.694073918063438, 3, -1, 3],
           [3.0000926340485923, 0, 0, 4],
           [2.694073918063438, 1, 1, -1],
           [0.11633857113691416, 0, 0, -1]], dtype=object)
    """
    if T_B is None:
        T_B = T_A
        ignore_trivial = True

    (
        T_A,
        μ_Q,
        σ_Q_inverse,
        μ_Q_m_1,
        T_A_subseq_isfinite,
        T_A_subseq_isconstant,
    ) = core.preprocess_diagonal(T_A, m)

    (
        T_B,
        M_T,
        Σ_T_inverse,
        M_T_m_1,
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
    M_T_future = dask_client.scatter(M_T, broadcast=True, hash=False)
    μ_Q_future = dask_client.scatter(μ_Q, broadcast=True, hash=False)
    Σ_T_inverse_future = dask_client.scatter(Σ_T_inverse, broadcast=True, hash=False)
    σ_Q_inverse_future = dask_client.scatter(σ_Q_inverse, broadcast=True, hash=False)
    M_T_m_1_future = dask_client.scatter(M_T_m_1, broadcast=True, hash=False)
    μ_Q_m_1_future = dask_client.scatter(μ_Q_m_1, broadcast=True, hash=False)
    T_A_subseq_isfinite_future = dask_client.scatter(
        T_A_subseq_isfinite, broadcast=True, hash=False
    )
    T_B_subseq_isfinite_future = dask_client.scatter(
        T_B_subseq_isfinite, broadcast=True, hash=False
    )
    T_A_subseq_isconstant_future = dask_client.scatter(
        T_A_subseq_isconstant, broadcast=True, hash=False
    )
    T_B_subseq_isconstant_future = dask_client.scatter(
        T_B_subseq_isconstant, broadcast=True, hash=False
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

    # Delete data from Dask cluster
    dask_client.cancel(T_A_future)
    dask_client.cancel(T_B_future)
    dask_client.cancel(M_T_future)
    dask_client.cancel(μ_Q_future)
    dask_client.cancel(Σ_T_inverse_future)
    dask_client.cancel(σ_Q_inverse_future)
    dask_client.cancel(M_T_m_1_future)
    dask_client.cancel(μ_Q_m_1_future)
    dask_client.cancel(T_A_subseq_isfinite_future)
    dask_client.cancel(T_B_subseq_isfinite_future)
    dask_client.cancel(T_A_subseq_isconstant_future)
    dask_client.cancel(T_B_subseq_isconstant_future)
    for diags_future in diags_futures:
        dask_client.cancel(diags_future)
    for future in futures:
        dask_client.cancel(future)

    core._check_P(out)

    return out
