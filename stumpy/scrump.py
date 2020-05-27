# STUMPY
# Copyright 2019 TD Ameritrade. Released under the terms of the 3-Clause BSD license.
# STUMPY is a trademark of TD Ameritrade IP Company, Inc. All rights reserved.

import logging

import numpy as np
from numba import njit, prange, config

from . import core

logger = logging.getLogger(__name__)


@njit(fastmath=True)
def _get_max_order_idx(m, n_A, n_B, orders, start, percentage):
    """
    Determine the order index for when the desired percentage of distances is computed

     Parameters
    ----------
    m : int
        Window size

    n_A : int
        The length of the time series or sequence for which to compute the matrix
        profile `T_A`

    n_B : int
        The length of the time series or sequence that contain your query subsequences
        of interest `T_B`

    orders : ndarray
        The order of diagonals to process and compute

    start : int
        The (inclusive) order index from which to start

    percentage : float
        Approximate percentage completed. The value is between 0.0 and 1.0.

    Returns
    -------
    max_order_id : int
        The order index that corresponds to desired percentage of distances to compute

    n_dist_computed : int
        The number of distances computed
    """

    max_n_dist = 0
    for order_idx in range(orders.shape[0]):
        k = orders[order_idx]
        if k >= 0:
            max_n_dist += min(n_A - m + 1 - k, n_B - m + 1)
        else:
            max_n_dist += min(n_A - m + 1, n_B - m + 1 + k)

    n_dist_computed = 0
    for order_idx in range(start, orders.shape[0]):
        k = orders[order_idx]
        if k >= 0:
            n_dist_computed += min(n_A - m + 1 - k, n_B - m + 1)
        else:
            n_dist_computed += min(n_A - m + 1, n_B - m + 1 + k)

        if n_dist_computed / max_n_dist > percentage:  # pragma: no cover
            break

    max_order_idx = order_idx + 1

    return max_order_idx, n_dist_computed


@njit(fastmath=True)
def _get_orders_ranges(n_split, m, n_A, n_B, orders, start, percentage):
    """
    For the desired percentage of distances to be computed from orders, split the
    orders into `n_split` chunks, and determine the appropriate start and stop
    (exclusive) order index ranges for each chunk

     Parameters
    ----------
    n_split : int
        The number of chunks to split the percentage of desired orders into

    m : int
        Window size

    n_A : int
        The length of the time series or sequence for which to compute the matrix
        profile `T_A`

    n_B : int
        The length of the time series or sequence that contain your query subsequences
        of interest `T_B`

    orders : ndarray
        The order of diagonals to process and compute

    start : int
        The (inclusive) order index from which to start

    percentage : float
        Approximate percentage completed. The value is between 0.0 and 1.0.

    Returns
    -------
    ranges : int
        The start (column 1) and (exclusive) stop (column 2) orders index ranges
        that corresponds to a desired percentage of distances to compute
    """

    max_order_idx, n_dist_computed = _get_max_order_idx(
        m, n_A, n_B, orders, start, percentage
    )

    orders_ranges = np.zeros((n_split, 2), np.int64)
    ranges_idx = 0
    range_start_idx = start
    max_n_dist_per_range = n_dist_computed / n_split

    n_dist_computed = 0
    for order_idx in range(start, max_order_idx):
        k = orders[order_idx]
        if k >= 0:
            n_dist_computed += min(n_A - m + 1 - k, n_B - m + 1)
        else:
            n_dist_computed += min(n_A - m + 1, n_B - m + 1 + k)

        if n_dist_computed > max_n_dist_per_range:  # pragma: no cover
            orders_ranges[ranges_idx, 0] = range_start_idx
            orders_ranges[ranges_idx, 1] = min(
                order_idx + 1, max_order_idx
            )  # Exclusive stop index
            # Reset and Update
            range_start_idx = order_idx + 1
            ranges_idx += 1
            n_dist_computed = 0

    # Handle final range outside of for loop if it was not saturated
    if ranges_idx < orders_ranges.shape[0]:
        orders_ranges[ranges_idx, 0] = range_start_idx
        orders_ranges[ranges_idx, 1] = max_order_idx

    return orders_ranges


@njit(fastmath=True)
def _compute_diagonal(
    T_A,
    T_B,
    m,
    M_T,
    Σ_T,
    μ_Q,
    σ_Q,
    orders,
    orders_start_idx,
    orders_stop_idx,
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

    M_T : ndarray
        Sliding mean of time series, `T_A`

    Σ_T : ndarray
        Sliding standard deviation of time series, `T_A`

    μ_Q : ndarray
        Mean of the query sequence, `Q`, relative to the current sliding window in `T_B`

    σ_Q : ndarray
        Standard deviation of the query sequence, `Q`, relative to the current
        sliding window in `T_B`

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
                QT = np.dot(T_A[i + k : i + k + m], T_B[i : i + m])
            else:
                QT = (
                    QT
                    - T_A[i + k - 1] * T_B[i - 1]
                    + T_A[i + k + m - 1] * T_B[i + m - 1]
                )

            D_squared = core._calculate_squared_distance(
                m, QT, M_T[i + k], Σ_T[i + k], μ_Q[i], σ_Q[i],
            )

            if D_squared < P[thread_idx, i]:
                P[thread_idx, i] = D_squared
                I[thread_idx, i] = i + k

            if ignore_trivial and D_squared < P[thread_idx, i + k]:
                P[thread_idx, i + k] = D_squared
                I[thread_idx, i + k] = i


@njit(parallel=True, fastmath=True)
def _scrump(T_A, T_B, m, M_T, Σ_T, μ_Q, σ_Q, orders, orders_ranges, ignore_trivial):
    """
    A Numba JIT-compiled version of SCRIMP for parallel computation of the matrix
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

    M_T : ndarray
        Sliding mean of time series, `T_A`

    Σ_T : ndarray
        Sliding standard deviation of time series, `T_A`

    μ_Q : ndarray
        Mean of the query sequence, `Q`, relative to the current sliding window in `T_B`

    σ_Q : ndarray
        Standard deviation of the query sequence, `Q`, relative to the current
        sliding window in `T_B`

    orders : ndarray
        The order of diagonals to process and compute

    orders_ranges : ndarray
        The start (column 1) and (exclusive) stop (column 2) order indices for
        each thread

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

    n = T_B.shape[0]
    l = n - m + 1
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
            M_T,
            Σ_T,
            μ_Q,
            σ_Q,
            orders,
            orders_ranges[thread_idx, 0],
            orders_ranges[thread_idx, 1],
            thread_idx,
            P,
            I,
            ignore_trivial,
        )

    # Reduction of results from all threads
    for thread_idx in range(1, n_threads):
        for i in prange(l):
            if P[0, i] > P[thread_idx, i]:
                P[0, i] = P[thread_idx, i]
                I[0, i] = I[thread_idx, i]

    return np.sqrt(P[0]), I[0]


@njit(parallel=True, fastmath=True)
def _prescrump(
    T_A,
    T_B,
    m,
    M_T,
    Σ_T,
    μ_Q,
    σ_Q,
    QT,
    i,
    s,
    squared_distance_profile,
    P_squared,
    I,
    excl_zone=None,
):
    """
    A Numba JIT-compiled implementation of the preSCRIMP algorithm.

    Parameters
    ----------
    T_A : ndarray
        The time series or sequence for which to compute the matrix profile

    T_B : ndarray
        The time series or sequence that contain your query subsequences
        of interest

    m : int
        Window size

    M_T : ndarray
        Sliding window mean for T_A

    Σ_T : ndarray
        Sliding window standard deviation for T_A

    μ_Q : ndarray
        Mean of the query sequence, `Q`, relative to the current sliding window in `T_B`

    σ_Q : ndarray
        Standard deviation of the query sequence, `Q`, relative to the current
        sliding window in `T_B`

    QT : ndarray
        Sliding dot product between `Q` in `T_B` and `T_A`

    i : int
        The subsequence index in `T_B` that corresponds to `Q`

    s : int
        The sampling interval that defaults to `int(np.ceil(m / 4))`

    squared_distance_profile : ndarray
        A reusable array to store the computed squared distance profile

    P_squared : ndarray
        The squared matrix profile

    I : ndarray
        The matrix profile indices

    excl_zone : int
        The half width for the exclusion zone relative to the `i`.

    Notes
    -----
    `DOI: 10.1109/ICDM.2018.00099 \
    <https://www.cs.ucr.edu/~eamonn/SCRIMP_ICDM_camera_ready_updated.pdf>`__

    See Algorithm 2
    """

    l = QT.shape[0]
    # Update P[i] relative to all T[j : j + m]
    Q = T_B[i : i + m]
    squared_distance_profile[:] = core._mass(Q, T_A, QT, μ_Q[i], σ_Q[i], M_T, Σ_T)
    squared_distance_profile[:] = np.square(squared_distance_profile)
    if excl_zone is not None:
        zone_start = max(0, i - excl_zone)
        zone_stop = min(l, i + excl_zone)
        squared_distance_profile[zone_start : zone_stop + 1] = np.inf
    I[i] = np.argmin(squared_distance_profile)
    P_squared[i] = squared_distance_profile[I[i]]
    if P_squared[i] == np.inf:  # pragma: no cover
        I[i] = -1

    j = I[i]
    # Given the squared distance, work backwards and compute QT
    QT_j = (m - P_squared[i] / 2.0) * (Σ_T[j] * σ_Q[i]) + (m * M_T[j] * μ_Q[i])
    QT_j_prime = QT_j
    for k in range(1, min(s, l - max(i, j))):
        QT_j = (
            QT_j
            - T_A[i + k - 1] * T_B[j + k - 1]
            + T_A[i + k + m - 1] * T_B[j + k + m - 1]
        )
        D_squared = core._calculate_squared_distance(
            m, QT_j, M_T[i + k], Σ_T[i + k], μ_Q[j + k], σ_Q[j + k],
        )
        if D_squared < P_squared[i + k]:
            P_squared[i + k] = D_squared
            I[i + k] = j + k
        if D_squared < P_squared[j + k]:
            P_squared[j + k] = D_squared
            I[j + k] = i + k
    QT_j = QT_j_prime
    for k in range(1, min(s, i + 1, j + 1)):
        QT_j = QT_j - T_A[i - k + m] * T_B[j - k + m] + T_A[i - k] * T_B[j - k]
        D_squared = core._calculate_squared_distance(
            m, QT_j, M_T[i - k], Σ_T[i - k], μ_Q[j - k], σ_Q[j - k],
        )
        if D_squared < P_squared[i - k]:
            P_squared[i - k] = D_squared
            I[i - k] = j - k
        if D_squared < P_squared[j - k]:
            P_squared[j - k] = D_squared
            I[j - k] = i - k

    return


def prescrump(T_A, m, T_B=None, s=None):
    """
    This is a convenience wrapper around the Numba JIT-compiled parallelized
    `_prescrump` function which computes the approximate matrix profile according
    to the preSCRIMP algorithm

    Parameters
    ----------
    T_A : ndarray
        The time series or sequence for which to compute the matrix profile

    m : int
        Window size

    T_B : ndarray
        The time series or sequence that contain your query subsequences
        of interest

    s : int
        The sampling interval that defaults to `int(np.ceil(m / 4))`

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

    See Algorithm 2
    """

    T_A = np.asarray(T_A)
    T_A = T_A.copy()
    T_A[np.isinf(T_A)] = np.nan
    core.check_dtype(T_A)

    if T_B is None:
        T_B = T_A
        excl_zone = int(np.ceil(m / 4))
    else:
        excl_zone = None

    T_B = np.asarray(T_B)
    T_B = T_B.copy()
    T_B[np.isinf(T_B)] = np.nan
    core.check_dtype(T_B)

    core.check_window_size(m)

    M_T, Σ_T = core.compute_mean_std(T_A, m)
    μ_Q, σ_Q = core.compute_mean_std(T_B, m)

    T_A[np.isnan(T_A)] = 0
    T_B[np.isnan(T_B)] = 0

    n_A = T_A.shape[0]
    n_B = T_B.shape[0]
    l = n_B - m + 1
    P_squared = np.empty(l)
    I = np.empty(l, dtype=np.int64)
    squared_distance_profile = np.empty(n_A - m + 1)

    if s is None:  # pragma: no cover
        s = excl_zone

    P_squared[:] = np.inf
    I[:] = -1

    for i in np.random.permutation(range(0, l, s)):
        QT = core.sliding_dot_product(T_B[i : i + m], T_A)
        _prescrump(
            T_A,
            T_B,
            m,
            M_T,
            Σ_T,
            μ_Q,
            σ_Q,
            QT,
            i,
            s,
            squared_distance_profile,
            P_squared,
            I,
            excl_zone,
        )

    P = np.sqrt(P_squared)

    return P, I


def scrump(
    T_A, m, T_B=None, ignore_trivial=True, percentage=0.01, pre_scrump=False, s=None
):
    """
    Compute the matrix profile with parallelized SCRIMP. This returns a generator that
    can be incrementally iterated on. For SCRIMP++, set `pre_scrump=True`.

    This is a convenience wrapper around the Numba JIT-compiled parallelized
    `_scrump` function which computes the matrix profile according to SCRIMP.

    Parameters
    ----------
    T_A : ndarray
        The time series or sequence for which to compute the matrix profile

    T_B : ndarray
        The time series or sequence that contain your query subsequences
        of interest

    m : int
        Window size

    ignore_trivial : bool
        Set to `True` if this is a self-join. Otherwise, for AB-join, set this to
        `False`. Default is `True`.

    percentage : float
        Approximate percentage completed. The value is between 0.0 and 1.0.

    pre_scrump : bool
        A flag for whether or not to perform the PreSCRIMP calculation prior to
        computing SCRIMP. If set to `True`, this is equivalent to computing
        SCRIMP++

    s : int
        The size of the PreSCRIMP fixed interval. If `pre-scrump=True` and `s=None`,
        then `s` will automatically be set to `s=int(np.ceil(m/4))`, the size of
        the exclusion zone.

    Returns
    -------
    out : ndarray
        Matrix profile and matrix profile indices

    Notes
    -----
    `DOI: 10.1109/ICDM.2018.00099 \
    <https://www.cs.ucr.edu/~eamonn/SCRIMP_ICDM_camera_ready_updated.pdf>`__

    See Algorithm 1 and Algorithm 2
    """

    if T_B is None:
        T_B = T_A
        ignore_trivial = True

    T_A, M_T, Σ_T = core.preprocess(T_A, m)
    T_B, μ_Q, σ_Q = core.preprocess(T_B, m)

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

    out = np.empty((l, 2), dtype=object)
    out[:, 0] = np.inf
    out[:, 1] = -1

    excl_zone = int(np.ceil(m / 4))

    if s is None:
        s = excl_zone

    if pre_scrump:
        if ignore_trivial:
            P, I = prescrump(T_A, m, s=s)
        else:
            P, I = prescrump(T_A, m, T_B=T_B, s=s)
        for i in range(P.shape[0]):
            if out[i, 0] > P[i]:
                out[i, 0] = P[i]
                out[i, 1] = I[i]

    if ignore_trivial:
        orders = np.random.permutation(range(excl_zone + 1, n_B - m + 1))
    else:
        orders = np.random.permutation(range(-(n_B - m + 1) + 1, n_A - m + 1))

    n_threads = config.NUMBA_NUM_THREADS
    percentage = min(percentage, 1.0)
    percentage = max(percentage, 0.0)
    generator_rounds = int(np.ceil(1.0 / percentage))
    start = 0
    for round in range(generator_rounds):
        orders_ranges = _get_orders_ranges(
            n_threads, m, n_A, n_B, orders, start, percentage
        )

        P, I = _scrump(
            T_A, T_B, m, M_T, Σ_T, μ_Q, σ_Q, orders, orders_ranges, ignore_trivial
        )
        start = orders_ranges[:, 1].max()

        # Update matrix profile and indices
        for i in range(out.shape[0]):
            if out[i, 0] > P[i]:
                out[i, 0] = P[i]
                out[i, 1] = I[i]

        yield out
