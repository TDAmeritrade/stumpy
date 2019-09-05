# STUMPY
# Copyright 2019 TD Ameritrade. Released under the terms of the 3-Clause BSD license.
# STUMPY is a trademark of TD Ameritrade IP Company, Inc. All rights reserved.

import numpy as np
from numba import cuda, int64
import math
from . import core
from . import _get_first_stump_profile, _get_QT
import logging

logger = logging.getLogger(__name__)


@cuda.jit
def _get_QT_kernel(i, T_A, T_B, m, QT_even, QT_odd, k):  # pragma: no gpu cover
    """
    A Numba implementation for GPU computation of the squared
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
    start = cuda.grid(1)
    stride = cuda.gridsize(1)

    for j in range(start, k, stride):
        if j >= 1:
            if i % 2 == 0:
                QT_even[j] = (
                    QT_odd[j - 1]
                    - T_B[i - 1] * T_A[j - 1]
                    + T_B[i + m - 1] * T_A[j + m - 1]
                )
            else:
                QT_odd[j] = (
                    QT_even[j - 1]
                    - T_B[i - 1] * T_A[j - 1]
                    + T_B[i + m - 1] * T_A[j + m - 1]
                )


@cuda.jit
def _calculate_squared_distance_kernel(
    i, m, M_T, Σ_T, QT, QT_first, μ_Q, σ_Q, D, denom
):  # pragma: no gpu cover
    """
    A Numba implementation for GPU computation of the squared
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
    start = cuda.grid(1)
    stride = cuda.gridsize(1)

    for j in range(start, D.shape[0], stride):
        denom[j] = m * σ_Q[i] * Σ_T[j]
        if denom[j] == 0:
            denom[j] = 1e-10

        QT[0] = QT_first[i]
        D[j] = abs(2 * m * (1.0 - (QT[j] - m * μ_Q[i] * M_T[j]) / denom[j]))


@cuda.jit
def _ignore_trivial(k, D, zone_start, zone_stop):  # pragma: no gpu cover
    start = cuda.grid(1)
    stride = cuda.gridsize(1)

    for j in range(start, k, stride):
        if j >= zone_start and j < zone_stop:
            D[j] = np.inf


@cuda.jit
def _get_PI_kernel(
    i, D, ignore_trivial, zone_start, zone_stop, profile, indices
):  # pragma: no gpu cover
    IL = cuda.local.array(1, dtype=int64)
    IR = cuda.local.array(1, dtype=int64)

    IL[0] = -1
    IR[0] = -1

    start = cuda.grid(1)
    stride = cuda.gridsize(1)

    for j in range(start, D.shape[0], stride):
        pass


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
    threads_per_block=1024,
):  # pragma: no gpu cover
    """
    """
    device_T_A = cuda.to_device(T_A)
    device_T_B = cuda.to_device(T_B)
    device_M_T = cuda.to_device(M_T)
    device_Σ_T = cuda.to_device(Σ_T)
    device_QT_odd = cuda.to_device(QT)
    device_QT_even = cuda.to_device(QT)
    device_QT_first = cuda.to_device(QT_first)
    device_μ_Q = cuda.to_device(μ_Q)
    device_σ_Q = cuda.to_device(σ_Q)
    device_D_even = cuda.device_array(k, dtype=np.float64)
    device_D_odd = cuda.device_array(k, dtype=np.float64)
    device_denom_even = cuda.device_array(QT.shape, dtype=np.float64)
    device_denom_odd = cuda.device_array(QT.shape, dtype=np.float64)

    profile = np.empty((range_stop - range_start,))  # float64
    indices = np.empty((range_stop - range_start, 3))  # int64

    profile[:] = np.inf
    indices[:, :] = -1
    device_profile = cuda.to_device(profile)
    device_indices = cuda.to_device(indices)

    blocks_per_grid = math.ceil(k / threads_per_block)

    for i in range(range_start, range_stop):
        zone_start = max(0, i - excl_zone)
        zone_stop = min(k, i + excl_zone)

        if i % 2 == 0:
            # Even
            device_QT = device_QT_even
            device_D = device_D_even
            device_denom = device_denom_even
        else:
            # Odd
            device_QT = device_QT_odd
            device_D = device_D_odd
            device_denom = device_denom_odd

        _get_QT_kernel[blocks_per_grid, threads_per_block](
            i, device_T_A, device_T_B, m, device_QT_even, device_QT_odd, k
        )

        _calculate_squared_distance_kernel[blocks_per_grid, threads_per_block](
            i,
            m,
            device_M_T,
            device_Σ_T,
            device_QT,
            device_QT_first,
            device_μ_Q,
            device_σ_Q,
            device_D,
            device_denom,
        )

        if ignore_trivial:
            _ignore_trivial[blocks_per_grid, threads_per_block](
                k, device_D, zone_start, zone_stop
            )

        _get_PI_kernel[blocks_per_grid, threads_per_block](
            i,
            device_D,
            ignore_trivial,
            zone_start,
            zone_stop,
            device_profile,
            device_indices,
        )

        D = device_D.copy_to_host()

        P = np.inf
        for j in range(k):
            if D[j] < P:
                P = D[j]
                I = j

        # Get left and right matrix profiles for self-joins
        if ignore_trivial and i > 0:
            IL = np.argmin(D[:i])
            if zone_start <= IL < zone_stop:  # pragma: no cover
                IL = -1
        else:
            IL = -1

        if ignore_trivial and i + 1 < D.shape[0]:
            IR = i + 1 + np.argmin(D[i + 1 :])
            if zone_start <= IR < zone_stop:  # pragma: no cover
                IR = -1
        else:
            IR = -1

        profile[i - range_start] = np.sqrt(P)
        indices[i - range_start] = I, IL, IR

    return profile, indices


def gpu_stump(
    T_A, m, T_B=None, ignore_trivial=True, threads_per_block=1024
):  # pragma: no gpu cover
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
        threads_per_block,
    )

    out[:, 0] = profile
    out[:, 1:4] = indices

    threshold = 10e-6
    if core.are_distances_too_small(out[:, 0], threshold=threshold):  # pragma: no cover
        logger.warning(f"A large number of values are smaller than {threshold}.")
        logger.warning("For a self-join, try setting `ignore_trivial = True`.")

    return out
