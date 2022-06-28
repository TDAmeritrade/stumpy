# STUMPY
# Copyright 2019 TD Ameritrade. Released under the terms of the 3-Clause BSD license.
# STUMPY is a trademark of TD Ameritrade IP Company, Inc. All rights reserved.
import logging
import math
import multiprocessing as mp
import os

import numpy as np
from numba import cuda

from . import core, config
from .gpu_aamp import gpu_aamp

logger = logging.getLogger(__name__)


@cuda.jit(
    "(i8, f8[:], f8[:], i8,  f8[:], f8[:], f8[:], f8[:], f8[:],"
    "f8[:], f8[:], i8, b1, i8, f8[:, :], i8[:, :], b1)"
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

    T_A : numpy.ndarray
        The time series or sequence for which to compute the dot product

    T_B : numpy.ndarray
        The time series or sequence that will be used to annotate T_A. For every
        subsequence in T_A, its nearest neighbor in T_B will be recorded.

    m : int
        Window size

    QT_even : numpy.ndarray
        The input QT array (dot product between the query sequence,`Q`, and
        time series, `T`) to use when `i` is even

    QT_odd : numpy.ndarray
        The input QT array (dot product between the query sequence,`Q`, and
        time series, `T`) to use when `i` is odd

    QT_first : numpy.ndarray
        Dot product between the first query sequence,`Q`, and time series, `T`

    M_T : numpy.ndarray
        Sliding mean of time series, `T`

    Σ_T : numpy.ndarray
        Sliding standard deviation of time series, `T`

    μ_Q : numpy.ndarray
        Mean of the query sequence, `Q`

    σ_Q : numpy.ndarray
        Standard deviation of the query sequence, `Q`

    k : int
        The total number of sliding windows to iterate over

    ignore_trivial : bool
        Set to `True` if this is a self-join. Otherwise, for AB-join, set this to
        `False`.

    excl_zone : int
        The half width for the exclusion zone relative to the current
        sliding window

    profile : numpy.ndarray
        Matrix profile. The first column consists of the global matrix profile,
        the second column consists of the left matrix profile, and the third
        column consists of the right matrix profile.

    indices : numpy.ndarray
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

        if compute_QT:
            QT_out[j] = (
                QT_in[j - 1] - T_B[i - 1] * T_A[j - 1] + T_B[i + m - 1] * T_A[j + m - 1]
            )

            QT_out[0] = QT_first[i]
        if math.isinf(M_T[j]) or math.isinf(μ_Q[i]):
            p_norm = np.inf
        else:
            if (
                σ_Q[i] < config.STUMPY_STDDEV_THRESHOLD
                or Σ_T[j] < config.STUMPY_STDDEV_THRESHOLD
            ):
                p_norm = m
            else:
                denom = m * σ_Q[i] * Σ_T[j]
                if math.fabs(denom) < config.STUMPY_DENOM_THRESHOLD:  # pragma nocover
                    denom = config.STUMPY_DENOM_THRESHOLD
                p_norm = abs(2 * m * (1.0 - (QT_out[j] - m * μ_Q[i] * M_T[j]) / denom))

            if (
                σ_Q[i] < config.STUMPY_STDDEV_THRESHOLD
                and Σ_T[j] < config.STUMPY_STDDEV_THRESHOLD
            ) or p_norm < config.STUMPY_P_NORM_THRESHOLD:
                p_norm = 0

        if ignore_trivial:
            if i <= zone_stop and i >= zone_start:
                p_norm = np.inf
            if p_norm < profile[j, 1] and i < j:
                profile[j, 1] = p_norm
                indices[j, 1] = i
            if p_norm < profile[j, 2] and i > j:
                profile[j, 2] = p_norm
                indices[j, 2] = i

        if p_norm < profile[j, 0]:
            profile[j, 0] = p_norm
            indices[j, 0] = i


def _gpu_stump(
    T_A_fname,
    T_B_fname,
    m,
    range_stop,
    excl_zone,
    M_T_fname,
    Σ_T_fname,
    QT_fname,
    QT_first_fname,
    μ_Q_fname,
    σ_Q_fname,
    k,
    ignore_trivial=True,
    range_start=1,
    device_id=0,
):
    """
    A Numba CUDA version of STOMP for parallel computation of the
    matrix profile, matrix profile indices, left matrix profile indices,
    and right matrix profile indices.

    Parameters
    ----------
    T_A_fname : str
        The file name for the time series or sequence for which to compute
        the matrix profile

    T_B_fname : str
        The file name for the time series or sequence that will be used to annotate T_A.
        For every subsequence in T_A, its nearest neighbor in T_B will be recorded.

    m : int
        Window size

    range_stop : int
        The index value along T_B for which to stop the matrix profile
        calculation. This parameter is here for consistency with the
        distributed `stumped` algorithm.

    excl_zone : int
        The half width for the exclusion zone relative to the current
        sliding window

    M_T_fname : str
        The file name for the sliding mean of time series, `T`

    Σ_T_fname : str
        The file name for the sliding standard deviation of time series, `T`

    QT_fname : str
        The file name for the dot product between some query sequence,`Q`,
        and time series, `T`

    QT_first_fname : str
        The file name for the QT for the first window relative to the current
        sliding window

    μ_Q_fname : str
        The file name for the mean of the query sequence, `Q`, relative to
        the current sliding window

    σ_Q_fname : str
        The file name for the standard deviation of the query sequence, `Q`,
        relative to the current sliding window

    k : int
        The total number of sliding windows to iterate over

    ignore_trivial : bool
        Set to `True` if this is a self-join. Otherwise, for AB-join, set this to
        `False`. Default is `True`.

    range_start : int
        The starting index value along T_B for which to start the matrix
        profile calculation. Default is 1.

    device_id : int
        The (GPU) device number to use. The default value is `0`.

    Returns
    -------
    profile_fname : str
        The file name for the matrix profile

    indices_fname : str
        The file name for the matrix profile indices. The first column of the
        array consists of the matrix profile indices, the second column consists
        of the left matrix profile indices, and the third column consists of the
        right matrix profile indices.

    Notes
    -----
    `DOI: 10.1109/ICDM.2016.0085 \
    <https://www.cs.ucr.edu/~eamonn/STOMP_GPU_final_submission_camera_ready.pdf>`__

    See Table II, Figure 5, and Figure 6

    Timeseries, T_A, will be annotated with the distance location
    (or index) of all its subsequences in another times series, T_B.

    Return: For every subsequence, Q, in T_A, you will get a distance
    and index for the closest subsequence in T_A. Thus, the array
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
    threads_per_block = config.STUMPY_THREADS_PER_BLOCK
    blocks_per_grid = math.ceil(k / threads_per_block)

    T_A = np.load(T_A_fname, allow_pickle=False)
    T_B = np.load(T_B_fname, allow_pickle=False)
    QT = np.load(QT_fname, allow_pickle=False)
    QT_first = np.load(QT_first_fname, allow_pickle=False)
    M_T = np.load(M_T_fname, allow_pickle=False)
    Σ_T = np.load(Σ_T_fname, allow_pickle=False)
    μ_Q = np.load(μ_Q_fname, allow_pickle=False)
    σ_Q = np.load(σ_Q_fname, allow_pickle=False)

    with cuda.gpus[device_id]:
        device_T_A = cuda.to_device(T_A)
        device_QT_odd = cuda.to_device(QT)
        device_QT_even = cuda.to_device(QT)
        device_QT_first = cuda.to_device(QT_first)
        device_μ_Q = cuda.to_device(μ_Q)
        device_σ_Q = cuda.to_device(σ_Q)
        if ignore_trivial:
            device_T_B = device_T_A
            device_M_T = device_μ_Q
            device_Σ_T = device_σ_Q
        else:
            device_T_B = cuda.to_device(T_B)
            device_M_T = cuda.to_device(M_T)
            device_Σ_T = cuda.to_device(Σ_T)

        profile = np.full((k, 3), np.inf, dtype=np.float64)
        indices = np.full((k, 3), -1, dtype=np.int64)

        device_profile = cuda.to_device(profile)
        device_indices = cuda.to_device(indices)
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

        profile = device_profile.copy_to_host()
        indices = device_indices.copy_to_host()
        profile = np.sqrt(profile)

        profile_fname = core.array_to_temp_file(profile)
        indices_fname = core.array_to_temp_file(indices)

    return profile_fname, indices_fname


@core.non_normalized(gpu_aamp)
def gpu_stump(
    T_A, m, T_B=None, ignore_trivial=True, device_id=0, normalize=True, p=2.0
):
    """
    Compute the z-normalized matrix profile with one or more GPU devices

    This is a convenience wrapper around the Numba `cuda.jit` `_gpu_stump` function
    which computes the matrix profile according to GPU-STOMP. The default number of
    threads-per-block is set to `512` and may be changed by setting the global parameter
    `config.STUMPY_THREADS_PER_BLOCK` to an appropriate number based on your GPU
    hardware.

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

    device_id : int or list, default 0
        The (GPU) device number to use. The default value is `0`. A list of
        valid device ids (int) may also be provided for parallel GPU-STUMP
        computation. A list of all valid device ids can be obtained by
        executing `[device.id for device in numba.cuda.list_devices()]`.

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
    stumpy.stumped : Compute the z-normalized matrix profile with a distributed dask
        cluster
    stumpy.scrump : Compute an approximate z-normalized matrix profile

    Notes
    -----
    `DOI: 10.1109/ICDM.2016.0085 \
    <https://www.cs.ucr.edu/~eamonn/STOMP_GPU_final_submission_camera_ready.pdf>`__

    See Table II, Figure 5, and Figure 6

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
    >>> from numba import cuda
    >>> if __name__ == "__main__":
    ...     all_gpu_devices = [device.id for device in cuda.list_devices()]
    ...     stumpy.gpu_stump(
    ...         np.array([584., -11., 23., 79., 1001., 0., -19.]),
    ...         m=3,
    ...         device_id=all_gpu_devices)
    array([[0.11633857113691416, 4, -1, 4],
           [2.694073918063438, 3, -1, 3],
           [3.0000926340485923, 0, 0, 4],
           [2.694073918063438, 1, 1, -1],
           [0.11633857113691416, 0, 0, -1]], dtype=object)
    """
    if T_B is None:  # Self join!
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

    core.check_window_size(m, max_size=min(T_A.shape[0], T_B.shape[0]))

    if ignore_trivial is False and core.are_arrays_equal(T_A, T_B):  # pragma: no cover
        logger.warning("Arrays T_A, T_B are equal, which implies a self-join.")
        logger.warning("Try setting `ignore_trivial = True`.")

    if ignore_trivial and core.are_arrays_equal(T_A, T_B) is False:  # pragma: no cover
        logger.warning("Arrays T_A, T_B are not equal, which implies an AB-join.")
        logger.warning("Try setting `ignore_trivial = False`.")

    n = T_B.shape[0]
    k = T_A.shape[0] - m + 1
    l = n - m + 1
    excl_zone = int(
        np.ceil(m / config.STUMPY_EXCL_ZONE_DENOM)
    )  # See Definition 3 and Figure 3

    T_A_fname = core.array_to_temp_file(T_A)
    T_B_fname = core.array_to_temp_file(T_B)
    M_T_fname = core.array_to_temp_file(M_T)
    Σ_T_fname = core.array_to_temp_file(Σ_T)
    μ_Q_fname = core.array_to_temp_file(μ_Q)
    σ_Q_fname = core.array_to_temp_file(σ_Q)

    out = np.empty((k, 4), dtype=object)

    if isinstance(device_id, int):
        device_ids = [device_id]
    else:
        device_ids = device_id

    profile = [None] * len(device_ids)
    indices = [None] * len(device_ids)

    for _id in device_ids:
        with cuda.gpus[_id]:
            if (
                cuda.current_context().__class__.__name__ != "FakeCUDAContext"
            ):  # pragma: no cover
                cuda.current_context().deallocations.clear()

    step = 1 + l // len(device_ids)

    # Start process pool for multi-GPU request
    if len(device_ids) > 1:  # pragma: no cover
        mp.set_start_method("spawn", force=True)
        pool = mp.Pool(processes=len(device_ids))
        results = [None] * len(device_ids)

    QT_fnames = []
    QT_first_fnames = []

    for idx, start in enumerate(range(0, l, step)):
        stop = min(l, start + step)

        QT, QT_first = core._get_QT(start, T_A, T_B, m)
        QT_fname = core.array_to_temp_file(QT)
        QT_first_fname = core.array_to_temp_file(QT_first)
        QT_fnames.append(QT_fname)
        QT_first_fnames.append(QT_first_fname)

        if len(device_ids) > 1 and idx < len(device_ids) - 1:  # pragma: no cover
            # Spawn and execute in child process for multi-GPU request
            results[idx] = pool.apply_async(
                _gpu_stump,
                (
                    T_A_fname,
                    T_B_fname,
                    m,
                    stop,
                    excl_zone,
                    M_T_fname,
                    Σ_T_fname,
                    QT_fname,
                    QT_first_fname,
                    μ_Q_fname,
                    σ_Q_fname,
                    k,
                    ignore_trivial,
                    start + 1,
                    device_ids[idx],
                ),
            )
        else:
            # Execute last chunk in parent process
            # Only parent process is executed when a single GPU is requested
            profile[idx], indices[idx] = _gpu_stump(
                T_A_fname,
                T_B_fname,
                m,
                stop,
                excl_zone,
                M_T_fname,
                Σ_T_fname,
                QT_fname,
                QT_first_fname,
                μ_Q_fname,
                σ_Q_fname,
                k,
                ignore_trivial,
                start + 1,
                device_ids[idx],
            )

    # Clean up process pool for multi-GPU request
    if len(device_ids) > 1:  # pragma: no cover
        pool.close()
        pool.join()

        # Collect results from spawned child processes if they exist
        for idx, result in enumerate(results):
            if result is not None:
                profile[idx], indices[idx] = result.get()

    os.remove(T_A_fname)
    os.remove(T_B_fname)
    os.remove(M_T_fname)
    os.remove(Σ_T_fname)
    os.remove(μ_Q_fname)
    os.remove(σ_Q_fname)
    for QT_fname in QT_fnames:
        os.remove(QT_fname)
    for QT_first_fname in QT_first_fnames:
        os.remove(QT_first_fname)

    for idx in range(len(device_ids)):
        profile_fname = profile[idx]
        indices_fname = indices[idx]
        profile[idx] = np.load(profile_fname, allow_pickle=False)
        indices[idx] = np.load(indices_fname, allow_pickle=False)
        os.remove(profile_fname)
        os.remove(indices_fname)

    for i in range(1, len(device_ids)):
        # Update all matrix profiles and matrix profile indices
        # (global, left, right) and store in profile[0] and indices[0]
        for col in range(profile[0].shape[1]):  # pragma: no cover
            cond = profile[0][:, col] < profile[i][:, col]
            profile[0][:, col] = np.where(cond, profile[0][:, col], profile[i][:, col])
            indices[0][:, col] = np.where(cond, indices[0][:, col], indices[i][:, col])

    out[:, 0] = profile[0][:, 0]
    out[:, 1:4] = indices[0][:, :]

    core._check_P(out)

    return out
