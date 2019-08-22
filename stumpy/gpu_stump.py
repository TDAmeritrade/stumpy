# STUMPY
# Copyright 2019 TD Ameritrade. Released under the terms of the 3-Clause BSD license.
# STUMPY is a trademark of TD Ameritrade IP Company, Inc. All rights reserved.

import numpy as np

try:
    import cupy as cp
except ModuleNotFoundError:  # pragma: no cover
    import numpy as cp
from . import core
from . import _get_first_stump_profile, _get_QT
import logging

logger = logging.getLogger(__name__)


def _gpu_calculate_squared_distance_profile(m, QT, μ_Q, σ_Q, M_T, Σ_T):
    """
    A CuPy implementation for GPU computation of the squared
    distance profile according to:

    `DOI: 10.1109/ICDM.2016.0179 \
    <https://www.cs.ucr.edu/~eamonn/PID4481997_extend_Matrix%20Profile_I.pdf>`__

    See Equation on Page 4

    Parameters
    ----------
    QT : ndarray
        Dot product between the query sequence,`Q`, and time series, `T`

    μ_Q : ndarray
        Mean of the query sequence, `Q`

    σ_Q : ndarray
        Standard deviation of the query sequence, `Q`

    M_T : ndarray
        Sliding mean of time series, `T`

    Σ_T : ndarray
        Sliding standard deviation of time series, `T`

    Returns
    -------
    D_squared : ndarray
        Squared z-normalized Eucldiean distances. The normal distances can
        be obtained by calculating taking the square root.
    """

    denom = m * σ_Q * Σ_T
    denom[denom == 0] = 1e-10  # Avoid divide by zero
    D_squared = cp.abs(2 * m * (1.0 - (QT - m * μ_Q * M_T) / denom))

    return D_squared


def _gpu_stump(
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
    A CuPy version of GPU-STOMP for GPU computation of the
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
        profile claculation. Default is 1.

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

    QT = cp.asarray(QT)
    μ_Q = cp.asarray(μ_Q)
    σ_Q = cp.asarray(σ_Q)
    M_T = cp.asarray(M_T)
    Σ_T = cp.asarray(Σ_T)
    QT_odd = QT.copy()
    QT_even = QT.copy()
    profile = cp.empty((range_stop - range_start,))  # float64
    indices = cp.empty((range_stop - range_start, 3))  # int64

    for i in range(range_start, range_stop):
        # Numba's prange requires incrementing a range by 1 so replace
        # `for j in range(k-1,0,-1)` with its incrementing compliment
        for rev_j in range(1, k):
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
            D = _gpu_calculate_squared_distance_profile(
                m, QT_even, μ_Q[i], σ_Q[i], M_T, Σ_T
            )
        else:
            QT_odd[0] = QT_first[i]
            D = _gpu_calculate_squared_distance_profile(
                m, QT_odd, μ_Q[i], σ_Q[i], M_T, Σ_T
            )

        if ignore_trivial:
            zone_start = max(0, i - excl_zone)
            zone_stop = min(k, i + excl_zone)
            D[zone_start:zone_stop] = cp.inf
        I = cp.argmin(D)
        P = cp.sqrt(D[I])

        # Get left and right matrix profiles for self-joins
        if ignore_trivial and i > 0:
            IL = cp.argmin(D[:i])
            if zone_start <= IL < zone_stop:  # pragma: no cover
                IL = -1
        else:
            IL = -1

        if ignore_trivial and i + 1 < D.shape[0]:
            IR = i + 1 + cp.argmin(D[i + 1 :])
            if zone_start <= IR < zone_stop:  # pragma: no cover
                IR = -1
        else:
            IR = -1

        # Only a part of the profile/indices array are passed
        profile[i - range_start] = P
        indices[i - range_start, 0] = I
        indices[i - range_start, 1] = IL
        indices[i - range_start, 2] = IR

    profile = cp.asnumpy(profile)
    indices = cp.asnumpy(indices)
    return profile, indices


def gpu_stump(T_A, m, T_B=None, ignore_trivial=True):
    """
    Compute the matrix profile with GPU-STOMP

    This is a convenience wrapper around the CuPy `_gpu_stump` function which
    computes the matrix profile according to GPU-STOMP.

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
    core.check_dtype(T_A)
    if T_B is None:  # Self join!
        T_B = T_A
        ignore_trivial = True
    T_B = np.asarray(T_B)
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

    out = np.empty((l, 4), dtype=object)
    profile = np.empty((l,), dtype="float64")
    indices = np.empty((l, 3), dtype="int64")

    start = 0
    stop = l

    profile[start], indices[start, :] = _get_first_stump_profile(
        start, T_A, T_B, m, excl_zone, M_T, Σ_T, ignore_trivial
    )

    QT, QT_first = _get_QT(start, T_A, T_B, m)

    profile[start + 1 : stop], indices[start + 1 : stop, :] = _gpu_stump(
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
