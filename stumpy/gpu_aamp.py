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

logger = logging.getLogger(__name__)


@cuda.jit(
    "(i8, f8[:], f8[:], i8, f8, f8[:], f8[:], f8[:], b1[:], b1[:],"
    "i8, b1, i8, f8[:, :], i8[:, :], b1)"
)
def _compute_and_update_PI_kernel(
    i,
    T_A,
    T_B,
    m,
    p,
    p_norm_even,
    p_norm_odd,
    p_norm_first,
    T_A_subseq_isfinite,
    T_B_subseq_isfinite,
    k,
    ignore_trivial,
    excl_zone,
    profile,
    indices,
    compute_p_norm,
):
    """
    A Numba CUDA kernel to update the non-normalized (i.e., without z-normalization)
    matrix profile and matrix profile indices

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

    p : float
        The p-norm to apply for computing the Minkowski distance.

    p_norm_even : numpy.ndarray
        The input p-norm array to use when `i` is even

    p_norm_odd : numpy.ndarray
        The input p-norm array to use when `i` is odd

    p_norm_first : numpy.ndarray
        The p-norm between the first query sequence,`Q`, and time series, `T`

    T_A_subseq_isfinite : numpy.ndarray
        A boolean array that indicates whether a subsequence in `T_A` contains a
        `np.nan`/`np.inf` value (False)

    T_B_subseq_isfinite : numpy.ndarray
        A boolean array that indicates whether a subsequence in `T_B` contains a
        `np.nan`/`np.inf` value (False)

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

    compute_p_norm : bool
        A boolean flag for whether or not to compute the p-norm

    Returns
    -------
    None

    Notes
    -----
    `arXiv:1901.05708 \
    <https://arxiv.org/pdf/1901.05708.pdf>`__

    See Algorithm 1

    Note that we have extended this algorithm for AB-joins as well.

    `DOI: 10.1109/ICDM.2016.0085 \
    <https://www.cs.ucr.edu/~eamonn/STOMP_GPU_final_submission_camera_ready.pdf>`__

    See Table II, Figure 5, and Figure 6
    """
    start = cuda.grid(1)
    stride = cuda.gridsize(1)

    if i % 2 == 0:
        p_norm_out = p_norm_even
        p_norm_in = p_norm_odd
    else:
        p_norm_out = p_norm_odd
        p_norm_in = p_norm_even

    for j in range(start, p_norm_out.shape[0], stride):
        zone_start = max(0, j - excl_zone)
        zone_stop = min(k, j + excl_zone)

        if compute_p_norm:
            p_norm_out[j] = (
                p_norm_in[j - 1]
                - abs(T_B[i - 1] - T_A[j - 1]) ** p
                + abs(T_B[i + m - 1] - T_A[j + m - 1]) ** p
            )
            p_norm_out[0] = p_norm_first[i]
        if not T_B_subseq_isfinite[i] or not T_A_subseq_isfinite[j]:
            p_norm = np.inf
        else:
            p_norm = p_norm_out[j]

        if p_norm < config.STUMPY_P_NORM_THRESHOLD:
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


def _gpu_aamp(
    T_A_fname,
    T_B_fname,
    m,
    range_stop,
    excl_zone,
    T_A_subseq_isfinite_fname,
    T_B_subseq_isfinite_fname,
    p,
    p_norm_fname,
    p_norm_first_fname,
    k,
    ignore_trivial=True,
    range_start=1,
    device_id=0,
):
    """
    A Numba CUDA version of AAMP for parallel computation of the non-normalized (i.e.,
    without z-normalization) matrix profile, matrix profile indices, left matrix profile
    indices, and right matrix profile indices.

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

    T_A_subseq_isfinite_fname : str
        The file name for the boolean array that indicates whether a subsequence in
        `T_A` contains a `np.nan`/`np.inf` value (False)

    T_B_subseq_isfinite_fname : str
        The file name for the boolean array that indicates whether a subsequence in
        `T_B` contains a `np.nan`/`np.inf` value (False)

    p : float
        The p-norm to apply for computing the Minkowski distance.

    p_norm_fname : str
        The file name for the p-norm between some query sequence,`Q`,
        and time series, `T`

    p_norm_first_fname : str
        The file name for the p-norm for the first window relative to the current
        sliding window

    k : int
        The total number of sliding windows to iterate over

    ignore_trivial : bool, default True
        Set to `True` if this is a self-join. Otherwise, for AB-join, set this to
        `False`. Default is `True`.

    range_start : int, default 1
        The starting index value along T_B for which to start the matrix
        profile calculation. Default is 1.

    device_id : int, default 0
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
    `arXiv:1901.05708 \
    <https://arxiv.org/pdf/1901.05708.pdf>`__

    See Algorithm 1

    Note that we have extended this algorithm for AB-joins as well.

    `DOI: 10.1109/ICDM.2016.0085 \
    <https://www.cs.ucr.edu/~eamonn/STOMP_GPU_final_submission_camera_ready.pdf>`__

    See Table II, Figure 5, and Figure 6
    """
    threads_per_block = config.STUMPY_THREADS_PER_BLOCK
    blocks_per_grid = math.ceil(k / threads_per_block)

    T_A = np.load(T_A_fname, allow_pickle=False)
    T_B = np.load(T_B_fname, allow_pickle=False)
    p_norm = np.load(p_norm_fname, allow_pickle=False)
    p_norm_first = np.load(p_norm_first_fname, allow_pickle=False)
    T_A_subseq_isfinite = np.load(T_A_subseq_isfinite_fname, allow_pickle=False)
    T_B_subseq_isfinite = np.load(T_B_subseq_isfinite_fname, allow_pickle=False)

    with cuda.gpus[device_id]:
        device_T_A = cuda.to_device(T_A)
        device_T_A_subseq_isfinite = cuda.to_device(T_A_subseq_isfinite)
        device_p_norm_odd = cuda.to_device(p_norm)
        device_p_norm_even = cuda.to_device(p_norm)
        device_p_norm_first = cuda.to_device(p_norm_first)
        if ignore_trivial:
            device_T_B = device_T_A
            device_T_B_subseq_isfinite = device_T_A_subseq_isfinite
        else:
            device_T_B = cuda.to_device(T_B)
            device_T_B_subseq_isfinite = cuda.to_device(T_B_subseq_isfinite)

        profile = np.full((k, 3), np.inf, dtype=np.float64)
        indices = np.full((k, 3), -1, dtype=np.int64)

        device_profile = cuda.to_device(profile)
        device_indices = cuda.to_device(indices)
        _compute_and_update_PI_kernel[blocks_per_grid, threads_per_block](
            range_start - 1,
            device_T_A,
            device_T_B,
            m,
            p,
            device_p_norm_even,
            device_p_norm_odd,
            device_p_norm_first,
            device_T_A_subseq_isfinite,
            device_T_B_subseq_isfinite,
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
                p,
                device_p_norm_even,
                device_p_norm_odd,
                device_p_norm_first,
                device_T_A_subseq_isfinite,
                device_T_B_subseq_isfinite,
                k,
                ignore_trivial,
                excl_zone,
                device_profile,
                device_indices,
                True,
            )

        profile = device_profile.copy_to_host()
        indices = device_indices.copy_to_host()
        profile = np.power(profile, 1.0 / p)

        profile_fname = core.array_to_temp_file(profile)
        indices_fname = core.array_to_temp_file(indices)

    return profile_fname, indices_fname


def gpu_aamp(T_A, m, T_B=None, ignore_trivial=True, device_id=0, p=2.0):
    """
    Compute the non-normalized (i.e., without z-normalization) matrix profile with one
    or more GPU devices

    This is a convenience wrapper around the Numba `cuda.jit` `_gpu_aamp` function
    which computes the non-normalized matrix profile according to modified version
    GPU-STOMP. The default number of threads-per-block is set to `512` and may be
    changed by setting the global parameter `config.STUMPY_THREADS_PER_BLOCK` to an
    appropriate number based on your GPU hardware.

    Parameters
    ----------
    T_A : numpy.ndarray
        The time series or sequence for which to compute the matrix profile

    m : int
        Window size

    T_B : numpy.ndarray, default None
        The time series or sequence that contain your query subsequences
        of interest. Default is `None` which corresponds to a self-join.

    ignore_trivial : bool, default True
        Set to `True` if this is a self-join. Otherwise, for AB-join, set this
        to `False`. Default is `True`.

    device_id : int or list, default 0
        The (GPU) device number to use. The default value is `0`. A list of
        valid device ids (int) may also be provided for parallel GPU-STUMP
        computation. A list of all valid device ids can be obtained by
        executing `[device.id for device in numba.cuda.list_devices()]`.

    p : float, default 2.0
        The p-norm to apply for computing the Minkowski distance.

    Returns
    -------
    out : numpy.ndarray
        The first column consists of the matrix profile, the second column
        consists of the matrix profile indices, the third column consists of
        the left matrix profile indices, and the fourth column consists of
        the right matrix profile indices.

    Notes
    -----
    `arXiv:1901.05708 \
    <https://arxiv.org/pdf/1901.05708.pdf>`__

    See Algorithm 1

    Note that we have extended this algorithm for AB-joins as well.

    `DOI: 10.1109/ICDM.2016.0085 \
    <https://www.cs.ucr.edu/~eamonn/STOMP_GPU_final_submission_camera_ready.pdf>`__

    See Table II, Figure 5, and Figure 6
    """
    if T_B is None:  # Self join!
        T_B = T_A
        ignore_trivial = True

    T_A, T_A_subseq_isfinite = core.preprocess_non_normalized(T_A, m)
    T_B, T_B_subseq_isfinite = core.preprocess_non_normalized(T_B, m)

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
    T_A_subseq_isfinite_fname = core.array_to_temp_file(T_A_subseq_isfinite)
    T_B_subseq_isfinite_fname = core.array_to_temp_file(T_B_subseq_isfinite)

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

    p_norm_fnames = []
    p_norm_first_fnames = []

    for idx, start in enumerate(range(0, l, step)):
        stop = min(l, start + step)

        p_norm = np.power(core.mass_absolute(T_B[start : start + m], T_A, p=p), p)
        p_norm_first = np.power(core.mass_absolute(T_A[:m], T_B, p=p), p)

        p_norm_fname = core.array_to_temp_file(p_norm)
        p_norm_first_fname = core.array_to_temp_file(p_norm_first)
        p_norm_fnames.append(p_norm_fname)
        p_norm_first_fnames.append(p_norm_first_fname)

        if len(device_ids) > 1 and idx < len(device_ids) - 1:  # pragma: no cover
            # Spawn and execute in child process for multi-GPU request
            results[idx] = pool.apply_async(
                _gpu_aamp,
                (
                    T_A_fname,
                    T_B_fname,
                    m,
                    stop,
                    excl_zone,
                    T_A_subseq_isfinite_fname,
                    T_B_subseq_isfinite_fname,
                    p,
                    p_norm_fname,
                    p_norm_first_fname,
                    k,
                    ignore_trivial,
                    start + 1,
                    device_ids[idx],
                ),
            )
        else:
            # Execute last chunk in parent process
            # Only parent process is executed when a single GPU is requested
            profile[idx], indices[idx] = _gpu_aamp(
                T_A_fname,
                T_B_fname,
                m,
                stop,
                excl_zone,
                T_A_subseq_isfinite_fname,
                T_B_subseq_isfinite_fname,
                p,
                p_norm_fname,
                p_norm_first_fname,
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
    os.remove(T_A_subseq_isfinite_fname)
    os.remove(T_B_subseq_isfinite_fname)
    for p_norm_fname in p_norm_fnames:
        os.remove(p_norm_fname)
    for p_norm_first_fname in p_norm_first_fnames:
        os.remove(p_norm_first_fname)

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
