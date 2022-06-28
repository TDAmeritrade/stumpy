# STUMPY
# Copyright 2019 TD Ameritrade. Released under the terms of the 3-Clause BSD license.
# STUMPY is a trademark of TD Ameritrade IP Company, Inc. All rights reserved.

import logging

import numpy as np
from numba import njit, prange
import numba

from . import core, config
from .aamp import aamp

logger = logging.getLogger(__name__)


@njit(
    # "(f8[:], f8[:], i8, f8[:], f8[:], f8[:], f8[:], f8[:], f8[:], f8[:], f8[:],"
    # "b1[:], b1[:], b1[:], b1[:], i8[:], i8, i8, i8, f8[:, :, :], i8[:, :, :], b1)",
    fastmath=True,
)
def _compute_diagonal(
    T_A,
    T_B,
    m,
    M_T,
    μ_Q,
    Σ_T_inverse,
    σ_Q_inverse,
    cov_a,
    cov_b,
    cov_c,
    cov_d,
    T_A_subseq_isfinite,
    T_B_subseq_isfinite,
    T_A_subseq_isconstant,
    T_B_subseq_isconstant,
    diags,
    diags_start_idx,
    diags_stop_idx,
    thread_idx,
    ρ,
    I,
    ignore_trivial,
):
    """
    Compute (Numba JIT-compiled) and update the Pearson correlation, ρ, and I
    sequentially along individual diagonals using a single thread and avoiding race
    conditions

    Parameters
    ----------
    T_A : numpy.ndarray
        The time series or sequence for which to compute the matrix profile

    T_B : numpy.ndarray
        The time series or sequence that will be used to annotate T_A. For every
        subsequence in T_A, its nearest neighbor in T_B will be recorded.

    m : int
        Window size

    M_T : numpy.ndarray
        Sliding mean of time series, `T`

    μ_Q : numpy.ndarray
        Mean of the query sequence, `Q`, relative to the current sliding window

    Σ_T_inverse : numpy.ndarray
        Inverse sliding standard deviation of time series, `T`

    σ_Q_inverse : numpy.ndarray
        Inverse standard deviation of the query sequence, `Q`, relative to the current
        sliding window

    cov_a : numpy.ndarray
        The first covariance term relating T_A[i + k + m - 1] and M_T_m_1[i + k]

    cov_b : numpy.ndarray
        The second covariance term relating T_B[i + m - 1] and μ_Q_m_1[i]

    cov_c : numpy.ndarray
        The third covariance term relating T_A[i + k - 1] and M_T_m_1[i + k]

    cov_d : numpy.ndarray
        The fourth covariance term relating T_B[i - 1] and μ_Q_m_1[i]

    μ_Q_m_1 : numpy.ndarray
        Mean of the query sequence, `Q`, relative to the current sliding window and
        using a window size of `m-1`

    T_A_subseq_isfinite : numpy.ndarray
        A boolean array that indicates whether a subsequence in `T_A` contains a
        `np.nan`/`np.inf` value (False)

    T_B_subseq_isfinite : numpy.ndarray
        A boolean array that indicates whether a subsequence in `T_B` contains a
        `np.nan`/`np.inf` value (False)

    T_A_subseq_isconstant : numpy.ndarray
        A boolean array that indicates whether a subsequence in `T_A` is constant (True)

    T_B_subseq_isconstant : numpy.ndarray
        A boolean array that indicates whether a subsequence in `T_B` is constant (True)

    diags : numpy.ndarray
        The diagonal indices

    diags_start_idx : int
        The starting (inclusive) diagonal index

    diags_stop_idx : int
        The stopping (exclusive) diagonal index

    thread_idx : int
        The thread index

    ρ : numpy.ndarray
        The Pearson correlations

    I : numpy.ndarray
        The matrix profile indices

    ignore_trivial : bool
        Set to `True` if this is a self-join. Otherwise, for AB-join, set this to
        `False`. Default is `True`.

    Returns
    -------
    None

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
    """
    n_A = T_A.shape[0]
    n_B = T_B.shape[0]
    m_inverse = 1.0 / m
    constant = (m - 1) * m_inverse * m_inverse  # (m - 1)/(m * m)

    for diag_idx in range(diags_start_idx, diags_stop_idx):
        k = diags[diag_idx]

        if k >= 0:
            iter_range = range(0, min(n_A - m + 1, n_B - m + 1 - k))
        else:
            iter_range = range(-k, min(n_A - m + 1, n_B - m + 1 - k))

        for i in iter_range:
            if i == 0 or (k < 0 and i == -k):
                cov = (
                    np.dot(
                        (T_B[i + k : i + k + m] - M_T[i + k]), (T_A[i : i + m] - μ_Q[i])
                    )
                    * m_inverse
                )
            else:
                # The next lines are equivalent and left for reference
                # cov = cov + constant * (
                #     (T_B[i + k + m - 1] - M_T_m_1[i + k])
                #     * (T_A[i + m - 1] - μ_Q_m_1[i])
                #     - (T_B[i + k - 1] - M_T_m_1[i + k]) * (T_A[i - 1] - μ_Q_m_1[i])
                # )
                cov = cov + constant * (
                    cov_a[i + k] * cov_b[i] - cov_c[i + k] * cov_d[i]
                )

            if T_B_subseq_isfinite[i + k] and T_A_subseq_isfinite[i]:
                # Neither subsequence contains NaNs
                if T_B_subseq_isconstant[i + k] or T_A_subseq_isconstant[i]:
                    pearson = 0.5
                else:
                    pearson = cov * Σ_T_inverse[i + k] * σ_Q_inverse[i]

                if T_B_subseq_isconstant[i + k] and T_A_subseq_isconstant[i]:
                    pearson = 1.0

                if pearson > ρ[thread_idx, i, 0]:
                    ρ[thread_idx, i, 0] = pearson
                    I[thread_idx, i, 0] = i + k

                if ignore_trivial:  # self-joins only
                    if pearson > ρ[thread_idx, i + k, 0]:
                        ρ[thread_idx, i + k, 0] = pearson
                        I[thread_idx, i + k, 0] = i

                    if i < i + k:
                        # left pearson correlation and left matrix profile index
                        if pearson > ρ[thread_idx, i + k, 1]:
                            ρ[thread_idx, i + k, 1] = pearson
                            I[thread_idx, i + k, 1] = i

                        # right pearson correlation and right matrix profile index
                        if pearson > ρ[thread_idx, i, 2]:
                            ρ[thread_idx, i, 2] = pearson
                            I[thread_idx, i, 2] = i + k

    return


@njit(
    # "(f8[:], f8[:], i8, f8[:], f8[:], f8[:], f8[:], f8[:], f8[:], b1[:], b1[:],"
    # "b1[:], b1[:], i8[:], b1)",
    parallel=True,
    fastmath=True,
)
def _stump(
    T_A,
    T_B,
    m,
    M_T,
    μ_Q,
    Σ_T_inverse,
    σ_Q_inverse,
    M_T_m_1,
    μ_Q_m_1,
    T_A_subseq_isfinite,
    T_B_subseq_isfinite,
    T_A_subseq_isconstant,
    T_B_subseq_isconstant,
    diags,
    ignore_trivial,
):
    """
    A Numba JIT-compiled version of STOMPopt with Pearson correlations for parallel
    computation of the matrix profile, matrix profile indices, left matrix profile
    indices, and right matrix profile indices.

    Parameters
    ----------
    T_A : numpy.ndarray
        The time series or sequence for which to compute the matrix profile

    T_B : numpy.ndarray
        The time series or sequence that will be used to annotate T_A. For every
        subsequence in T_A, its nearest neighbor in T_B will be recorded.

    m : int
        Window size

    M_T : numpy.ndarray
        Sliding mean of time series, `T`

    μ_Q : numpy.ndarray
        Mean of the query sequence, `Q`, relative to the current sliding window

    Σ_T_inverse : numpy.ndarray
        Inverse sliding standard deviation of time series, `T`

    σ_Q_inverse : numpy.ndarray
        Inverse standard deviation of the query sequence, `Q`, relative to the current
        sliding window

    M_T_m_1 : numpy.ndarray
        Sliding mean of time series, `T`, using a window size of `m-1`

    μ_Q_m_1 : numpy.ndarray
        Mean of the query sequence, `Q`, relative to the current sliding window and
        using a window size of `m-1`

    T_A_subseq_isfinite : numpy.ndarray
        A boolean array that indicates whether a subsequence in `T_A` contains a
        `np.nan`/`np.inf` value (False)

    T_B_subseq_isfinite : numpy.ndarray
        A boolean array that indicates whether a subsequence in `T_B` contains a
        `np.nan`/`np.inf` value (False)

    T_A_subseq_isconstant : numpy.ndarray
        A boolean array that indicates whether a subsequence in `T_A` is constant (True)

    T_B_subseq_isconstant : numpy.ndarray
        A boolean array that indicates whether a subsequence in `T_B` is constant (True)

    diags : numpy.ndarray
        The diagonal indices

    ignore_trivial : bool
        Set to `True` if this is a self-join. Otherwise, for AB-join, set this to
        `False`. Default is `True`.

    Returns
    -------
    profile : numpy.ndarray
        Matrix profile

    indices : numpy.ndarray
        The first column consists of the matrix profile indices, the second
        column consists of the left matrix profile indices, and the third
        column consists of the right matrix profile indices.

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
    """
    n_A = T_A.shape[0]
    n_B = T_B.shape[0]
    l = n_A - m + 1
    n_threads = numba.config.NUMBA_NUM_THREADS
    ρ = np.full((n_threads, l, 3), -np.inf, dtype=np.float64)
    I = np.full((n_threads, l, 3), -1, dtype=np.int64)

    ndist_counts = core._count_diagonal_ndist(diags, m, n_A, n_B)
    diags_ranges = core._get_array_ranges(ndist_counts, n_threads, False)

    cov_a = T_B[m - 1 :] - M_T_m_1[:-1]
    cov_b = T_A[m - 1 :] - μ_Q_m_1[:-1]
    # The next lines are equivalent and left for reference
    # cov_c = np.roll(T_A, 1)
    # cov_ = cov_c[:M_T_m_1.shape[0]] - M_T_m_1[:]
    cov_c = np.empty(M_T_m_1.shape[0], dtype=np.float64)
    cov_c[1:] = T_B[: M_T_m_1.shape[0] - 1]
    cov_c[0] = T_B[-1]
    cov_c[:] = cov_c - M_T_m_1
    # The next lines are equivalent and left for reference
    # cov_d = np.roll(T_B, 1)
    # cov_d = cov_d[:μ_Q_m_1.shape[0]] - μ_Q_m_1[:]
    cov_d = np.empty(μ_Q_m_1.shape[0], dtype=np.float64)
    cov_d[1:] = T_A[: μ_Q_m_1.shape[0] - 1]
    cov_d[0] = T_A[-1]
    cov_d[:] = cov_d - μ_Q_m_1

    for thread_idx in prange(n_threads):
        # Compute and update cov, I within a single thread to avoiding race conditions
        _compute_diagonal(
            T_A,
            T_B,
            m,
            M_T,
            μ_Q,
            Σ_T_inverse,
            σ_Q_inverse,
            cov_a,
            cov_b,
            cov_c,
            cov_d,
            T_A_subseq_isfinite,
            T_B_subseq_isfinite,
            T_A_subseq_isconstant,
            T_B_subseq_isconstant,
            diags,
            diags_ranges[thread_idx, 0],
            diags_ranges[thread_idx, 1],
            thread_idx,
            ρ,
            I,
            ignore_trivial,
        )

    # Reduction of results from all threads
    for thread_idx in range(1, n_threads):
        for i in prange(l):
            if ρ[0, i, 0] < ρ[thread_idx, i, 0]:
                ρ[0, i, 0] = ρ[thread_idx, i, 0]
                I[0, i, 0] = I[thread_idx, i, 0]
            # left pearson correlation and left matrix profile indices
            if ρ[0, i, 1] < ρ[thread_idx, i, 1]:
                ρ[0, i, 1] = ρ[thread_idx, i, 1]
                I[0, i, 1] = I[thread_idx, i, 1]
            # right pearson correlation and right matrix profile indices
            if ρ[0, i, 2] < ρ[thread_idx, i, 2]:
                ρ[0, i, 2] = ρ[thread_idx, i, 2]
                I[0, i, 2] = I[thread_idx, i, 2]

    # Convert pearson correlations to distances
    p_norm = np.abs(2 * m * (1 - ρ[0, :, :]))
    for i in prange(p_norm.shape[0]):
        if p_norm[i, 0] < config.STUMPY_P_NORM_THRESHOLD:
            p_norm[i, 0] = 0.0
        if p_norm[i, 1] < config.STUMPY_P_NORM_THRESHOLD:
            p_norm[i, 1] = 0.0
        if p_norm[i, 2] < config.STUMPY_P_NORM_THRESHOLD:
            p_norm[i, 2] = 0.0
    P = np.sqrt(p_norm)

    return P[:, :], I[0, :, :]


@core.non_normalized(aamp)
def stump(T_A, m, T_B=None, ignore_trivial=True, normalize=True, p=2.0):
    """
    Compute the z-normalized matrix profile

    This is a convenience wrapper around the Numba JIT-compiled parallelized
    `_stump` function which computes the matrix profile according to STOMPopt with
    Pearson correlations.

    Parameters
    ----------
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
    stumpy.stumped : Compute the z-normalized matrix profile with a distributed dask
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
    >>> import stumpy
    >>> stumpy.stump(np.array([584., -11., 23., 79., 1001., 0., -19.]), m=3)
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

    if ignore_trivial:
        diags = np.arange(excl_zone + 1, n_A - m + 1, dtype=np.int64)
    else:
        diags = np.arange(-(n_A - m + 1) + 1, n_B - m + 1, dtype=np.int64)

    P, I = _stump(
        T_A,
        T_B,
        m,
        M_T,
        μ_Q,
        Σ_T_inverse,
        σ_Q_inverse,
        M_T_m_1,
        μ_Q_m_1,
        T_A_subseq_isfinite,
        T_B_subseq_isfinite,
        T_A_subseq_isconstant,
        T_B_subseq_isconstant,
        diags,
        ignore_trivial,
    )

    out[:, 0] = P[:, 0]
    out[:, 1:] = I

    core._check_P(out)

    return out
