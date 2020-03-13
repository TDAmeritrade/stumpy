# STUMPY
# Copyright 2019 TD Ameritrade. Released under the terms of the 3-Clause BSD license.
# STUMPY is a trademark of TD Ameritrade IP Company, Inc. All rights reserved.

import logging

import numpy as np
from numba import njit, prange

from . import core, stamp

logger = logging.getLogger(__name__)


def _get_first_stump_profile(start, T_A, T_B, m, excl_zone, M_T, Σ_T, ignore_trivial):
    """
    Compute the matrix profile, matrix profile index, left matrix profile
    index, and right matrix profile index for given window within the times
    series or sequence that is denote by the `start` index. Essentially, this
    is a convenience wrapper around `stamp.mass`

    Parameters
    ----------
    start : int
        The window index to calculate the first matrix profile, matrix profile
        index, left matrix profile index, and right matrix profile index for.

    T_A : ndarray
        The time series or sequence for which the matrix profile index will
        be returned

    T_B : ndarray
        The time series or sequence that contains your query subsequences

    m : int
        Window size

    excl_zone : int
        The half width for the exclusion zone relative to the `start`.

    M_T : ndarray
        Sliding mean for `T_A`

    Σ_T : ndarray
        Sliding standard deviation for `T_A`

    ignore_trivial : bool
        `True` if this is a self join and `False` otherwise (i.e., AB-join).

    Returns
    -------
    P : float64
        Matrix profile for the window with index equal to `start`

    I : Tuple[int64, int64, int64]
        Matrix profile index, left matrix profile index, and right matrix profile
        index for the window with index equal to `start`. The left and right matrix
        profile indices are automatically set to `-1` for self-joins (i.e., when
        `ignore_trivial` is set to `True`.
    """

    # Handle first subsequence, add exclusionary zone
    if ignore_trivial:
        P, I = stamp.mass(T_B[start : start + m], T_A, M_T, Σ_T, start, excl_zone)
        PL, IL = stamp.mass(
            T_B[start : start + m], T_A, M_T, Σ_T, start, excl_zone, left=True
        )
        PR, IR = stamp.mass(
            T_B[start : start + m], T_A, M_T, Σ_T, start, excl_zone, right=True
        )
    else:
        P, I = stamp.mass(T_B[start : start + m], T_A, M_T, Σ_T)
        # No left and right matrix profile available
        IL = -1
        IR = -1

    return P, (I, IL, IR)


def _get_QT(start, T_A, T_B, m):
    """
    Compute the sliding dot product between the query, `T_B`, (from
    [start:start+m]) and the time series, `T_A`. Additionally, compute
    QT for the first window.

    Parameters
    ----------
    start : int
        The window index for T_B from which to calculate the QT dot product

    T_A : ndarray
        The time series or sequence for which to compute the dot product

    T_B : ndarray
        The time series or sequence that contain your query subsequence
        of interest

    m : int
        Window size

    Returns
    -------
    QT : ndarray
        Given `start`, return the corresponding QT

    QT_first : ndarray
         QT for the first window
    """

    QT = core.sliding_dot_product(T_B[start : start + m], T_A)
    QT_first = core.sliding_dot_product(T_A[:m], T_B)

    return QT, QT_first


@njit(parallel=True, fastmath=True)
def _stump(
    T_A,
    T_B,
    m,
    range_stop,
    excl_zone,
    M_T,
    Σ_T,
    QT,
    QT_first,
    μ_Q,
    σ_Q,
    k,
    ignore_trivial=True,
    range_start=1,
):
    """
    A Numba JIT-compiled version of STOMP for parallel computation of the
    matrix profile, matrix profile indices, left matrix profile indices,
    and right matrix profile indices.

    Parameters
    ----------
    T_A : ndarray
        The time series or sequence for which to compute the matrix profile

    T_B : ndarray
        The time series or sequence that contain your query subsequences
        of interest

    m : int
        Window size

    range_stop : int
        The index value along T_B for which to stop the matrix profile
        calculation. This parameter is here for consistency with the
        distributed `stumped` algorithm.

    excl_zone : int
        The half width for the exclusion zone relative to the current
        sliding window

    M_T : ndarray
        Sliding mean of time series, `T`

    Σ_T : ndarray
        Sliding standard deviation of time series, `T`

    QT : ndarray
        Dot product between some query sequence,`Q`, and time series, `T`

    QT_first : ndarray
        QT for the first window relative to the current sliding window

    μ_Q : ndarray
        Mean of the query sequence, `Q`, relative to the current sliding window

    σ_Q : ndarray
        Standard deviation of the query sequence, `Q`, relative to the current
        sliding window

    k : int
        The total number of sliding windows to iterate over

    ignore_trivial : bool
        Set to `True` if this is a self-join. Otherwise, for AB-join, set this to
        `False`. Default is `True`.

    range_start : int
        The starting index value along T_B for which to start the matrix
        profile calculation. Default is 1.

    Returns
    -------
    profile : ndarray
        Matrix profile

    indices : ndarray
        The first column consists of the matrix profile indices, the second
        column consists of the left matrix profile indices, and the third
        column consists of the right matrix profile indices.

    Notes
    -----
    `DOI: 10.1109/ICDM.2016.0085 \
    <https://www.cs.ucr.edu/~eamonn/STOMP_GPU_final_submission_camera_ready.pdf>`__

    See Table II

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

    QT_odd = QT.copy()
    QT_even = QT.copy()
    profile = np.empty((range_stop - range_start,))  # float64
    indices = np.empty((range_stop - range_start, 3))  # int64

    for i in range(range_start, range_stop):
        # Numba's prange requires incrementing a range by 1 so replace
        # `for j in range(k-1,0,-1)` with its incrementing compliment
        for rev_j in prange(1, k):
            j = k - rev_j
            # GPU Stomp Parallel Implementation with Numba
            # DOI: 10.1109/ICDM.2016.0085
            # See Figure 5
            if i % 2 == 0:
                # Even
                QT_even[j] = (
                    QT_odd[j - 1]
                    - T_B[i - 1] * T_A[j - 1]
                    + T_B[i + m - 1] * T_A[j + m - 1]
                )
            else:
                # Odd
                QT_odd[j] = (
                    QT_even[j - 1]
                    - T_B[i - 1] * T_A[j - 1]
                    + T_B[i + m - 1] * T_A[j + m - 1]
                )

        if i % 2 == 0:
            QT_even[0] = QT_first[i]
            D = core._calculate_squared_distance_profile(
                m, QT_even, μ_Q[i], σ_Q[i], M_T, Σ_T
            )
        else:
            QT_odd[0] = QT_first[i]
            D = core._calculate_squared_distance_profile(
                m, QT_odd, μ_Q[i], σ_Q[i], M_T, Σ_T
            )

        if ignore_trivial:
            zone_start = max(0, i - excl_zone)
            zone_stop = min(k, i + excl_zone)
            D[zone_start : zone_stop + 1] = np.inf

        I = np.argmin(D)
        P = np.sqrt(D[I])
        if P == np.inf:
            I = -1

        # Get left and right matrix profiles
        IL = -1
        PL = np.inf
        if ignore_trivial and i > 0:
            IL = np.argmin(D[:i])
            PL = D[IL]
        if PL == np.inf or zone_start <= IL < zone_stop:
            IL = -1

        IR = -1
        PR = np.inf
        if ignore_trivial and i + 1 < D.shape[0]:
            IR = i + 1 + np.argmin(D[i + 1 :])
            PR = D[IR]
        if PR == np.inf or zone_start <= IR < zone_stop:
            IR = -1

        # Only a part of the profile/indices array are passed
        profile[i - range_start] = P
        indices[i - range_start] = I, IL, IR

    return profile, indices


def stump(T_A, m, T_B=None, ignore_trivial=True):
    """
    Compute the matrix profile with parallelized STOMP

    This is a convenience wrapper around the Numba JIT-compiled parallelized
    `_stump` function which computes the matrix profile according to STOMP.

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
        consists of the matrix profile indices, the third column consists of
        the left matrix profile indices, and the fourth column consists of
        the right matrix profile indices.

    Notes
    -----

    `DOI: 10.1109/ICDM.2016.0085 \
    <https://www.cs.ucr.edu/~eamonn/STOMP_GPU_final_submission_camera_ready.pdf>`__

    See Table II

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

    T_A = np.asarray(T_A)
    if T_A.ndim != 1:  # pragma: no cover
        raise ValueError(
            f"T_A is {T_A.ndim}-dimensional and must be 1-dimensional. "
            "For multidimensional STUMP use `stumpy.mstump` or `stumpy.mstumped`"
        )
    n = T_A.shape[0]

    T_A = T_A.copy()
    T_A[np.isinf(T_A)] = np.nan
    core.check_dtype(T_A)

    if T_B is None:
        T_B = T_A
        ignore_trivial = True

    T_B = np.asarray(T_B)
    T_B = T_B.copy()
    if T_B.ndim != 1:  # pragma: no cover
        raise ValueError(
            f"T_B is {T_B.ndim}-dimensional and must be 1-dimensional. "
            "For multidimensional STUMP use `stumpy.mstump` or `stumpy.mstumped`"
        )
    T_B[np.isinf(T_B)] = np.nan
    core.check_dtype(T_B)

    core.check_window_size(m)

    if ignore_trivial is False and core.are_arrays_equal(T_A, T_B):  # pragma: no cover
        logger.warning("Arrays T_A, T_B are equal, which implies a self-join.")
        logger.warning("Try setting `ignore_trivial = True`.")

    if ignore_trivial and core.are_arrays_equal(T_A, T_B) is False:  # pragma: no cover
        logger.warning("Arrays T_A, T_B are not equal, which implies an AB-join.")
        logger.warning("Try setting `ignore_trivial = False`.")

    n = T_B.shape[0]
    k = T_A.shape[0] - m + 1
    l = n - m + 1
    excl_zone = int(np.ceil(m / 4))  # See Definition 3 and Figure 3

    M_T, Σ_T = core.compute_mean_std(T_A, m)
    μ_Q, σ_Q = core.compute_mean_std(T_B, m)

    T_A[np.isnan(T_A)] = 0

    out = np.empty((l, 4), dtype=object)
    profile = np.empty((l,), dtype="float64")
    indices = np.empty((l, 3), dtype="int64")

    start = 0
    stop = l

    profile[start], indices[start, :] = _get_first_stump_profile(
        start, T_A, T_B, m, excl_zone, M_T, Σ_T, ignore_trivial
    )

    T_B[
        np.isnan(T_B)
    ] = 0  # Remove all nan values from T_B only after first profile is calculated

    QT, QT_first = _get_QT(start, T_A, T_B, m)

    profile[start + 1 : stop], indices[start + 1 : stop, :] = _stump(
        T_A,
        T_B,
        m,
        stop,
        excl_zone,
        M_T,
        Σ_T,
        QT,
        QT_first,
        μ_Q,
        σ_Q,
        k,
        ignore_trivial,
        start + 1,
    )

    out[:, 0] = profile
    out[:, 1:4] = indices

    threshold = 10e-6
    if core.are_distances_too_small(out[:, 0], threshold=threshold):  # pragma: no cover
        logger.warning(f"A large number of values are smaller than {threshold}.")
        logger.warning("For a self-join, try setting `ignore_trivial = True`.")

    return out
