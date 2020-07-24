# STUMPY
# Copyright 2019 TD Ameritrade. Released under the terms of the 3-Clause BSD license.
# STUMPY is a trademark of TD Ameritrade IP Company, Inc. All rights reserved.

import logging

import numpy as np
from numba import njit, prange, config

from . import core
from .scrump import _get_orders_ranges

logger = logging.getLogger(__name__)


@njit(fastmath=True)
def _compute_diagonal(
    T_A,
    T_B,
    m,
    orders,
    orders_start_idx,
    orders_stop_idx,
    subseq_A_isnan,
    subseq_B_isnan,
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
    T_A : ndarray
        The time series or sequence for which to compute the matrix profile

    T_B : ndarray
        The time series or sequence that contain your query subsequences
        of interest

    m : int
        Window size

    orders : ndarray
        The order of diagonals to process and compute

    orders_start_idx : int
        The start index for a range of diagonal order to process and compute

    orders_stop_idx : int
        The (exclusive) stop index for a range of diagonal order to process and compute

    P : ndarray
        Matrix profile

    I : ndarray
        Matrix profile indices

    subseq_A_isnan : ndarray
        A boolean array that indicates whether a subsequence in `T_A` contains a
        `np.nan` value

    subseq_B_isnan : ndarray
        A boolean array that indicates whether a subsequence in `T_B` contains a
        `np.nan` value

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

    for order_idx in range(orders_start_idx, orders_stop_idx):
        k = orders[order_idx]

        if k >= 0:
            iter_range = range(0, min(n_B - m + 1, n_A - m + 1 - k))
        else:
            iter_range = range(-k, min(n_B - m + 1, n_A - m + 1 - k))

        for i in iter_range:
            if i == 0 or i == k or (k < 0 and i == -k):
                D_squared = np.linalg.norm(T_A[i + k : i + k + m] - T_B[i : i + m]) ** 2
            else:
                D_squared = np.abs(
                    D_squared
                    - (T_A[i + k - 1] - T_B[i - 1]) ** 2
                    + (T_A[i + k + m - 1] - T_B[i + m - 1]) ** 2
                )

            if D_squared < core.D_SQUARED_THRESHOLD:
                D_squared = 0.0

            if not (subseq_A_isnan[i + k] or subseq_B_isnan[i]):
                # Neither subsequence contains NaNs
                if D_squared < P[thread_idx, i]:
                    P[thread_idx, i] = D_squared
                    I[thread_idx, i] = i + k

                if ignore_trivial and D_squared < P[thread_idx, i + k]:
                    P[thread_idx, i + k] = D_squared
                    I[thread_idx, i + k] = i
    return


@njit(parallel=True, fastmath=True)
def _aamp(
    T_A, T_B, m, orders, orders_ranges, subseq_A_isnan, subseq_B_isnan, ignore_trivial
):
    """
    A Numba JIT-compiled version of AAMP for parallel computation of the matrix
    profile and matrix profile indices.

    Parameters
    ----------
    T_A : ndarray
        The time series or sequence for which to compute the matrix profile

    T_B : ndarray
        The time series or sequence that contain your query subsequences
        of interest

    m : int
        Window size

    orders : ndarray
        The order of diagonals to process and compute

    orders_ranges : ndarray
        The start (column 1) and (exclusive) stop (column 2) order indices for
        each thread

    subseq_A_isnan : ndarray
        A boolean array that indicates whether a subsequence in `T_A` contains a
        `np.nan` value

    subseq_B_isnan : ndarray
        A boolean array that indicates whether a subsequence in `T_B` contains a
        `np.nan` value

    ignore_trivial : bool
        Set to `True` if this is a self-join. Otherwise, for AB-join, set this to
        `False`. Default is `True`.

    Returns
    -------
    P : ndarray
        Matrix profile

    I : ndarray
        Matrix profile indices

    Notes
    -----
    `DOI: 10.1109/ICDM.2018.00099 \
    <https://www.cs.ucr.edu/~eamonn/SCRIMP_ICDM_camera_ready_updated.pdf>`__

    See Algorithm 1
    """
    n_B = T_B.shape[0]
    l = n_B - m + 1
    n_threads = config.NUMBA_NUM_THREADS
    P = np.empty((n_threads, l))
    I = np.empty((n_threads, l), np.int64)

    P[:, :] = np.inf
    I[:, :] = -1

    for thread_idx in prange(n_threads):
        # Compute and pdate P, I within a single thread while avoiding race conditions
        _compute_diagonal(
            T_A,
            T_B,
            m,
            orders,
            orders_ranges[thread_idx, 0],
            orders_ranges[thread_idx, 1],
            subseq_A_isnan,
            subseq_B_isnan,
            thread_idx,
            P,
            I,
            ignore_trivial,
        )

    for thread_idx in range(n_threads):
        print(P[thread_idx])

    # Reduction of results from all threads
    for thread_idx in range(1, n_threads):
        for i in prange(l):
            if P[0, i] > P[thread_idx, i]:
                P[0, i] = P[thread_idx, i]
                I[0, i] = I[thread_idx, i]

    print(np.sqrt(P[0]))

    return np.sqrt(P[0]), I[0]


def aamp(T_A, m, T_B=None, ignore_trivial=True):
    """
    Compute the matrix profile with parallelized AAMP, which uses non-normalized
    Euclidean distances for computing a matrix profile.

    This is a convenience wrapper around the Numba JIT-compiled parallelized
    `_aamp` function which computes the matrix profile according to AAMP.

    Parameters
    ----------
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

    subseq_A_isnan = np.any(np.isnan(core.rolling_window(T_A, m)), axis=1)
    subseq_B_isnan = np.any(np.isnan(core.rolling_window(T_B, m)), axis=1)

    T_A[np.isnan(T_A)] = 0
    T_B[np.isnan(T_B)] = 0

    n_A = T_A.shape[0]
    n_B = T_B.shape[0]
    l = n_B - m + 1
    n_threads = config.NUMBA_NUM_THREADS

    excl_zone = int(np.ceil(m / 4))
    out = np.empty((l, 2), dtype=object)

    if ignore_trivial:
        orders = np.arange(excl_zone + 1, n_B - m + 1)
    else:
        orders = np.arange(-(n_B - m + 1) + 1, n_A - m + 1)

    orders_ranges = _get_orders_ranges(n_threads, m, n_A, n_B, orders, 0, 1.0)

    P, I = _aamp(
        T_A,
        T_B,
        m,
        orders,
        orders_ranges,
        subseq_A_isnan,
        subseq_B_isnan,
        ignore_trivial,
    )

    out[:, 0] = P
    out[:, 1] = I

    return out
