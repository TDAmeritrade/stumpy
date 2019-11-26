# STUMPY
# Copyright 2019 TD Ameritrade. Released under the terms of the 3-Clause BSD license.
# STUMPY is a trademark of TD Ameritrade IP Company, Inc. All rights reserved.

import numpy as np
from numba import cuda
import math
from . import core
from . import _get_QT
import logging

logger = logging.getLogger(__name__)
THREADS_PER_BLOCK = 512


@cuda.jit(
    "(i8, f8[:], f8[:], i8,  f8[:], f8[:], f8[:], f8[:], f8[:],"
    "f8[:], f8[:], i8, b1, i8, f8[:, :], f8[:, :], b1)"
)
def _compute_and_update_PI_kernel(
    i,
    T_A,
    T_B,
    m,
    QT_even,
    QT_odd,
    QT_first,
    M_T,
    Σ_T,
    μ_Q,
    σ_Q,
    k,
    ignore_trivial,
    excl_zone,
    profile,
    indices,
    compute_QT,
):
    """
    A Numba CUDA kernel to update the matrix profile and matrix profile indices

    Parameters
    ----------
    i : int
        sliding window `i`

    T_A : ndarray
        The time series or sequence for which to compute the dot product

    T_B : ndarray
        The time series or sequence that contain your query subsequence
        of interest

    m : int
        Window size

    QT_even : ndarray
        The input QT array (dot product between the query sequence,`Q`, and
        time series, `T`) to use when `i` is even

    QT_odd : ndarray
        The input QT array (dot product between the query sequence,`Q`, and
        time series, `T`) to use when `i` is odd

    QT_first : ndarray
        Dot product between the first query sequence,`Q`, and time series, `T`

    M_T : ndarray
        Sliding mean of time series, `T`

    Σ_T : ndarray
        Sliding standard deviation of time series, `T`

    μ_Q : ndarray
        Mean of the query sequence, `Q`

    σ_Q : ndarray
        Standard deviation of the query sequence, `Q`

    k : int
        The total number of sliding windows to iterate over

    ignore_trivial : bool
        Set to `True` if this is a self-join. Otherwise, for AB-join, set this to
        `False`.

    excl_zone : int
        The half width for the exclusion zone relative to the current
        sliding window

    profile : ndarray
        Matrix profile. The first column consists of the global matrix profile,
        the second column consists of the left matrix profile, and the third
        column consists of the right matrix profile.

    indices : ndarray
        The first column consists of the matrix profile indices, the second
        column consists of the left matrix profile indices, and the third
        column consists of the right matrix profile indices.

    compute_QT : bool
        A boolean flag for whether or not to compute QT

    Returns
    -------
    None

    Notes
    -----
    `DOI: 10.1109/ICDM.2016.0085 \
    <https://www.cs.ucr.edu/~eamonn/STOMP_GPU_final_submission_camera_ready.pdf>`__

    See Table II, Figure 5, and Figure 6
    """

    start = cuda.grid(1)
    stride = cuda.gridsize(1)

    if i % 2 == 0:
        QT_out = QT_even
        QT_in = QT_odd
    else:
        QT_out = QT_odd
        QT_in = QT_even

    for j in range(start, QT_out.shape[0], stride):
        zone_start = max(0, j - excl_zone)
        zone_stop = min(k, j + excl_zone)
        denom = m * σ_Q[i] * Σ_T[j]
        if denom == 0:  # pragma: no cover
            denom = 1e-10

        if compute_QT:
            QT_out[j] = (
                QT_in[j - 1] - T_B[i - 1] * T_A[j - 1] + T_B[i + m - 1] * T_A[j + m - 1]
            )

            QT_out[0] = QT_first[i]
        D = abs(2 * m * (1.0 - (QT_out[j] - m * μ_Q[i] * M_T[j]) / denom))

        if ignore_trivial:
            if i < zone_stop and i >= zone_start:
                D = np.inf
            if D < profile[j, 1] and i < j:
                profile[j, 1] = D
                indices[j, 1] = i
            if D < profile[j, 2] and i > j:
                profile[j, 2] = D
                indices[j, 2] = i

        if D < profile[j, 0]:
            profile[j, 0] = D
            indices[j, 0] = i


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
    threads_per_block=THREADS_PER_BLOCK,
):
    """
    A Numba CUDA version of STOMP for parallel computation of the
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

    threads_per_block : int
        The number of GPU threads to use for all kernels. The default value is
        set in `THREADS_PER_BLOCK=512`.

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

    See Table II, Figure 5, and Figure 6

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

    device_T_A = cuda.to_device(T_A)
    device_T_B = cuda.to_device(T_B)
    device_M_T = cuda.to_device(M_T)
    device_Σ_T = cuda.to_device(Σ_T)
    device_QT_odd = cuda.to_device(QT)
    device_QT_even = cuda.to_device(QT)
    device_QT_first = cuda.to_device(QT_first)
    device_μ_Q = cuda.to_device(μ_Q)
    device_σ_Q = cuda.to_device(σ_Q)

    profile = np.empty((k, 3))  # float64
    indices = np.empty((k, 3))  # int64

    profile[:] = np.inf
    indices[:, :] = -1
    device_profile = cuda.to_device(profile)
    device_indices = cuda.to_device(indices)

    blocks_per_grid = math.ceil(k / threads_per_block)

    _compute_and_update_PI_kernel[blocks_per_grid, threads_per_block](
        range_start - 1,
        device_T_A,
        device_T_B,
        m,
        device_QT_even,
        device_QT_odd,
        device_QT_first,
        device_M_T,
        device_Σ_T,
        device_μ_Q,
        device_σ_Q,
        k,
        ignore_trivial,
        excl_zone,
        device_profile,
        device_indices,
        False,
    )

    for i in range(range_start, range_stop):
        _compute_and_update_PI_kernel[blocks_per_grid, threads_per_block](
            i,
            device_T_A,
            device_T_B,
            m,
            device_QT_even,
            device_QT_odd,
            device_QT_first,
            device_M_T,
            device_Σ_T,
            device_μ_Q,
            device_σ_Q,
            k,
            ignore_trivial,
            excl_zone,
            device_profile,
            device_indices,
            True,
        )

    profile = device_profile.copy_to_host()[:, 0]
    indices = device_indices.copy_to_host()
    profile = np.sqrt(profile)

    return profile, indices


def gpu_stump(
    T_A, m, T_B=None, ignore_trivial=True, threads_per_block=THREADS_PER_BLOCK
):
    """
    Compute the matrix profile with GPU-STOMP

    This is a convenience wrapper around the Numba `cuda.jit` `_gpu_stump` function
    which computes the matrix profile according to GPU-STOMP.

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

    threads_per_block : int
        The number of GPU threads to use for all kernels. The default value is
        set in `THREADS_PER_BLOCK=512`.

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

    See Table II, Figure 5, and Figure 6

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
    core.check_nan(T_A)
    if T_B is None:  # Self join!
        T_B = T_A
        ignore_trivial = True
    T_B = np.asarray(T_B)

    core.check_dtype(T_B)
    core.check_nan(T_B)
    core.check_window_size(m)

    if ignore_trivial is False and core.are_arrays_equal(T_A, T_B):  # pragma: no cover
        logger.warning("Arrays T_A, T_B are equal, which implies a self-join.")
        logger.warning("Try setting `ignore_trivial = True`.")

    if ignore_trivial and core.are_arrays_equal(T_A, T_B) is False:  # pragma: no cover
        logger.warning("Arrays T_A, T_B are not equal, which implies an AB-join.")
        logger.warning("Try setting `ignore_trivial = False`.")

    # Swap T_A and T_B for GPU implementation
    # This keeps the API identical to and compatible with `stumpy.stump`
    tmp_T = T_A
    T_A = T_B
    T_B = tmp_T

    n = T_B.shape[0]
    k = T_A.shape[0] - m + 1
    l = n - m + 1
    excl_zone = int(np.ceil(m / 4))  # See Definition 3 and Figure 3

    M_T, Σ_T = core.compute_mean_std(T_A, m)
    μ_Q, σ_Q = core.compute_mean_std(T_B, m)

    out = np.empty((k, 4), dtype=object)
    profile = np.empty((k,), dtype="float64")
    indices = np.empty((k, 3), dtype="int64")

    start = 0
    stop = l

    QT, QT_first = _get_QT(start, T_A, T_B, m)
    profile[:], indices[:, :] = _gpu_stump(
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
