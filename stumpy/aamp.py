# STUMPY
# Copyright 2019 TD Ameritrade. Released under the terms of the 3-Clause BSD license.
# STUMPY is a trademark of TD Ameritrade IP Company, Inc. All rights reserved.

import logging

import numpy as np
from numba import njit, prange
import numba

from . import core, config

logger = logging.getLogger(__name__)


@njit(
    # "(f8[:], f8[:], i8, b1[:], b1[:], f8, i8[:], i8, i8, i8, f8[:, :, :],"
    # "i8[:, :, :], b1)",
    fastmath=True,
)
def _compute_diagonal(
    T_A,
    T_B,
    m,
    T_A_subseq_isfinite,
    T_B_subseq_isfinite,
    p,
    diags,
    diags_start_idx,
    diags_stop_idx,
    thread_idx,
    P,
    I,
    ignore_trivial,
):
    """
    Compute (Numba JIT-compiled) and update P, I along a single diagonal using a single
    thread and avoiding race conditions

    Parameters
    ----------
    T_A : numpy.ndarray
        The time series or sequence for which to compute the matrix profile

    T_B : numpy.ndarray
        The time series or sequence that will be used to annotate T_A. For every
        subsequence in T_A, its nearest neighbor in T_B will be recorded.

    m : int
        Window size

    P : numpy.ndarray
        Matrix profile

    I : numpy.ndarray
        Matrix profile indices

    T_A_subseq_isfinite : numpy.ndarray
        A boolean array that indicates whether a subsequence in `T_A` contains a
        `np.nan`/`np.inf` value (False)

    T_B_subseq_isfinite : numpy.ndarray
        A boolean array that indicates whether a subsequence in `T_B` contains a
        `np.nan`/`np.inf` value (False)

    p : float
        The p-norm to apply for computing the Minkowski distance.

    diags : numpy.ndarray
        The diag of diagonals to process and compute

    diags_start_idx : int
        The start index for a range of diagonal diag to process and compute

    diags_stop_idx : int
        The (exclusive) stop index for a range of diagonal diag to process and compute

    thread_idx : int
        The thread index

    ignore_trivial : bool
        Set to `True` if this is a self-join. Otherwise, for AB-join, set this to
        `False`. Default is `True`.

    Returns
    -------
    None
    """
    n_A = T_A.shape[0]
    n_B = T_B.shape[0]

    for diag_idx in range(diags_start_idx, diags_stop_idx):
        k = diags[diag_idx]

        if k >= 0:
            iter_range = range(0, min(n_A - m + 1, n_B - m + 1 - k))
        else:
            iter_range = range(-k, min(n_A - m + 1, n_B - m + 1 - k))

        for i in iter_range:
            if i == 0 or (k < 0 and i == -k):
                p_norm = (
                    np.linalg.norm(T_B[i + k : i + k + m] - T_A[i : i + m], ord=p) ** p
                )
            else:
                p_norm = np.abs(
                    p_norm
                    - np.absolute(T_B[i + k - 1] - T_A[i - 1]) ** p
                    + np.absolute(T_B[i + k + m - 1] - T_A[i + m - 1]) ** p
                )

            if p_norm < config.STUMPY_P_NORM_THRESHOLD:
                p_norm = 0.0

            if T_A_subseq_isfinite[i] and T_B_subseq_isfinite[i + k]:
                # Neither subsequence contains NaNs
                if p_norm < P[thread_idx, i, 0]:
                    P[thread_idx, i, 0] = p_norm
                    I[thread_idx, i, 0] = i + k

                if ignore_trivial:
                    if p_norm < P[thread_idx, i + k, 0]:
                        P[thread_idx, i + k, 0] = p_norm
                        I[thread_idx, i + k, 0] = i

                    if i < i + k:
                        # left matrix profile and left matrix profile index
                        if p_norm < P[thread_idx, i + k, 1]:
                            P[thread_idx, i + k, 1] = p_norm
                            I[thread_idx, i + k, 1] = i

                        # right matrix profile and right matrix profile index
                        if p_norm < P[thread_idx, i, 2]:
                            P[thread_idx, i, 2] = p_norm
                            I[thread_idx, i, 2] = i + k

    return


@njit(
    # "(f8[:], f8[:], i8, b1[:], b1[:], i8[:], b1)",
    parallel=True,
    fastmath=True,
)
def _aamp(
    T_A, T_B, m, T_A_subseq_isfinite, T_B_subseq_isfinite, p, diags, ignore_trivial
):
    """
    A Numba JIT-compiled version of AAMP for parallel computation of the matrix
    profile and matrix profile indices.

    Parameters
    ----------
    T_A : numpy.ndarray
        The time series or sequence for which to compute the matrix profile

    T_B : numpy.ndarray
        The time series or sequence that will be used to annotate T_A. For every
        subsequence in T_A, its nearest neighbor in T_B will be recorded.

    m : int
        Window size

    T_A_subseq_isfinite : numpy.ndarray
        A boolean array that indicates whether a subsequence in `T_A` contains a
        `np.nan`/`np.inf` value (False)

    T_B_subseq_isfinite : numpy.ndarray
        A boolean array that indicates whether a subsequence in `T_B` contains a
        `np.nan`/`np.inf` value (False)

    p : float
        The p-norm to apply for computing the Minkowski distance.

    diags : numpy.ndarray
        The diag of diagonals to process and compute

    ignore_trivial : bool
        Set to `True` if this is a self-join. Otherwise, for AB-join, set this to
        `False`. Default is `True`.

    Returns
    -------
    P : numpy.ndarray
        Matrix profile

    I : numpy.ndarray
        Matrix profile indices

    Notes
    -----
    `DOI: 10.1109/ICDM.2018.00099 \
    <https://www.cs.ucr.edu/~eamonn/SCRIMP_ICDM_camera_ready_updated.pdf>`__

    See Algorithm 1
    """
    n_A = T_A.shape[0]
    n_B = T_B.shape[0]
    l = n_A - m + 1
    n_threads = numba.config.NUMBA_NUM_THREADS
    P = np.full((n_threads, l, 3), np.inf, dtype=np.float64)
    I = np.full((n_threads, l, 3), -1, dtype=np.int64)

    ndist_counts = core._count_diagonal_ndist(diags, m, n_A, n_B)
    diags_ranges = core._get_array_ranges(ndist_counts, n_threads, False)

    for thread_idx in prange(n_threads):
        # Compute and update P, I within a single thread while avoiding race conditions
        _compute_diagonal(
            T_A,
            T_B,
            m,
            T_A_subseq_isfinite,
            T_B_subseq_isfinite,
            p,
            diags,
            diags_ranges[thread_idx, 0],
            diags_ranges[thread_idx, 1],
            thread_idx,
            P,
            I,
            ignore_trivial,
        )

    # Reduction of results from all threads
    for thread_idx in range(1, n_threads):
        for i in prange(l):
            if P[0, i, 0] > P[thread_idx, i, 0]:
                P[0, i, 0] = P[thread_idx, i, 0]
                I[0, i, 0] = I[thread_idx, i, 0]
            # left matrix profile and left matrix profile indices
            if P[0, i, 1] > P[thread_idx, i, 1]:
                P[0, i, 1] = P[thread_idx, i, 1]
                I[0, i, 1] = I[thread_idx, i, 1]
            # right matrix profile and right matrix profile indices
            if P[0, i, 2] > P[thread_idx, i, 2]:
                P[0, i, 2] = P[thread_idx, i, 2]
                I[0, i, 2] = I[thread_idx, i, 2]

    return np.power(P[0, :, :], 1.0 / p), I[0, :, :]


def aamp(T_A, m, T_B=None, ignore_trivial=True, p=2.0):
    """
    Compute the non-normalized (i.e., without z-normalization) matrix profile

    This is a convenience wrapper around the Numba JIT-compiled parallelized
    `_aamp` function which computes the matrix profile according to AAMP.

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

    if ignore_trivial:
        diags = np.arange(excl_zone + 1, n_A - m + 1, dtype=np.int64)
    else:
        diags = np.arange(-(n_A - m + 1) + 1, n_B - m + 1, dtype=np.int64)

    P, I = _aamp(
        T_A, T_B, m, T_A_subseq_isfinite, T_B_subseq_isfinite, p, diags, ignore_trivial
    )

    out[:, 0] = P[:, 0]
    out[:, 1:] = I[:, :]

    return out
