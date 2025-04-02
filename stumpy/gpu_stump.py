# STUMPY
# Copyright 2019 TD Ameritrade. Released under the terms of the 3-Clause BSD license.
# STUMPY is a trademark of TD Ameritrade IP Company, Inc. All rights reserved.
import math
import multiprocessing as mp
import os

import numpy as np
from numba import cuda

from . import config, core
from .gpu_aamp import gpu_aamp
from .mparray import mparray


@cuda.jit(
    "(i8, f8[:], f8[:], i8,  f8[:], f8[:], f8[:], f8[:], f8[:],"
    "f8[:], f8[:], b1[:], b1[:], i8, b1, i8, f8[:, :], f8[:], "
    "f8[:], i8[:, :], i8[:], i8[:], b1, i8[:], i8, i8)"
)
def _compute_and_update_PI_kernel(
    idx,
    T_A,
    T_B,
    m,
    QT_even,
    QT_odd,
    QT_first,
    μ_Q,
    σ_Q,
    M_T,
    Σ_T,
    Q_subseq_isconstant,
    T_subseq_isconstant,
    w,
    ignore_trivial,
    excl_zone,
    profile,
    profile_L,
    profile_R,
    indices,
    indices_L,
    indices_R,
    compute_QT,
    bfs,
    nlevel,
    k,
):
    """
    A Numba CUDA kernel to update the matrix profile and matrix profile indices

    Parameters
    ----------
    idx : int
        The index for sliding window `j` (in  `T_B`)

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

    μ_Q : numpy.ndarray
        Mean of the query sequence, `Q`

    σ_Q : numpy.ndarray
        Standard deviation of the query sequence, `Q`

    M_T : numpy.ndarray
        Sliding mean of time series, `T`

    Σ_T : numpy.ndarray
        Sliding standard deviation of time series, `T`

    Q_subseq_isconstant : numpy.ndarray
        A boolean array that indicates whether the subsequence in `Q` is constant (True)

    T_subseq_isconstant : numpy.ndarray
        A boolean array that indicates whether a subsequence in `T` is constant (True)

    w : int
        The total number of sliding windows to iterate over

    ignore_trivial : bool
        Set to `True` if this is a self-join. Otherwise, for AB-join, set this to
        `False`.

    excl_zone : int
        The half width for the exclusion zone relative to the current
        sliding window

    profile : numpy.ndarray
        The (top-k) matrix profile, sorted in ascending order per row

    profile_L : numpy.ndarray
        The (top-1) left matrix profile

    profile_R : numpy.ndarray
        The (top-1) right matrix profile

    indices : numpy.ndarray
        The (top-k) matrix profile indices

    indices_L : numpy.ndarray
        The (top-1) left matrix profile indices

    indices_R : numpy.ndarray
        The (top-1) right matrix profile indices

    compute_QT : bool
        A boolean flag for whether or not to compute QT

    bfs : numpy.ndarray
        The breadth-first-search indices where the missing leaves of its corresponding
        binary search tree are filled with -1.

    nlevel : int
        The number of levels in the binary search tree from which the array
        `bfs` is obtained.

    k : int
        The number of top `k` smallest distances used to construct the matrix profile.
        Note that this will increase the total computational time and memory usage
        when k > 1.

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

    j = idx
    # The name `i` is reserved to be used as an index for `T_A`

    if j % 2 == 0:
        QT_out = QT_even
        QT_in = QT_odd
    else:
        QT_out = QT_odd
        QT_in = QT_even

    for i in range(start, QT_out.shape[0], stride):
        zone_start = max(0, i - excl_zone)
        zone_stop = min(w, i + excl_zone)

        if compute_QT:
            QT_out[i] = (
                QT_in[i - 1] - T_A[i - 1] * T_B[j - 1] + T_A[i + m - 1] * T_B[j + m - 1]
            )

            QT_out[0] = QT_first[j]
        if math.isinf(μ_Q[i]) or math.isinf(M_T[j]):
            p_norm = np.inf
        elif Q_subseq_isconstant[i] and T_subseq_isconstant[j]:
            p_norm = 0
        elif Q_subseq_isconstant[i] or T_subseq_isconstant[j]:
            p_norm = m
        else:
            denom = (σ_Q[i] * Σ_T[j]) * m
            denom = max(denom, config.STUMPY_DENOM_THRESHOLD)
            ρ = (QT_out[i] - (μ_Q[i] * M_T[j]) * m) / denom
            ρ = min(ρ, 1.0)
            p_norm = 2 * m * (1.0 - ρ)

        if ignore_trivial:
            if j <= zone_stop and j >= zone_start:
                p_norm = np.inf
            if p_norm < profile_L[i] and j < i:
                profile_L[i] = p_norm
                indices_L[i] = j
            if p_norm < profile_R[i] and j > i:
                profile_R[i] = p_norm
                indices_R[i] = j

        if p_norm < profile[i, -1]:
            idx = core._gpu_searchsorted_right(profile[i], p_norm, bfs, nlevel)
            for g in range(k - 1, idx, -1):
                profile[i, g] = profile[i, g - 1]
                indices[i, g] = indices[i, g - 1]

            profile[i, idx] = p_norm
            indices[i, idx] = j


def _gpu_stump(
    T_A_fname,
    T_B_fname,
    m,
    range_stop,
    excl_zone,
    μ_Q_fname,
    σ_Q_fname,
    QT_fname,
    QT_first_fname,
    M_T_fname,
    Σ_T_fname,
    Q_subseq_isconstant_fname,
    T_subseq_isconstant_fname,
    w,
    ignore_trivial=True,
    range_start=1,
    device_id=0,
    k=1,
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

    μ_Q_fname : str
        The file name for the mean of the query sequence, `Q`, relative to
        the current sliding window

    σ_Q_fname : str
        The file name for the standard deviation of the query sequence, `Q`,
        relative to the current sliding window

    QT_fname : str
        The file name for the dot product between some query sequence,`Q`,
        and time series, `T`

    QT_first_fname : str
        The file name for the QT for the first window relative to the current
        sliding window

    M_T_fname : str
        The file name for the sliding mean of time series, `T`

    Σ_T_fname : str
        The file name for the sliding standard deviation of time series, `T`

    Q_subseq_isconstant_fname : str
        The file name for the rolling isconstant in `Q`

    T_subseq_isconstant_fname : str
        The file name for the rolling isconstant in `T`

    w : int
        The total number of sliding windows to iterate over

    ignore_trivial : bool
        Set to `True` if this is a self-join. Otherwise, for AB-join, set this to
        `False`. Default is `True`.

    range_start : int
        The starting index value along T_B for which to start the matrix
        profile calculation. Default is 1.

    device_id : int
        The (GPU) device number to use. The default value is `0`.

    k : int
        The number of top `k` smallest distances used to construct the matrix profile.
        Note that this will increase the total computational time and memory usage
        when k > 1.

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
    blocks_per_grid = math.ceil(w / threads_per_block)

    T_A = np.load(T_A_fname, allow_pickle=False)
    T_B = np.load(T_B_fname, allow_pickle=False)
    QT = np.load(QT_fname, allow_pickle=False)
    QT_first = np.load(QT_first_fname, allow_pickle=False)
    μ_Q = np.load(μ_Q_fname, allow_pickle=False)
    σ_Q = np.load(σ_Q_fname, allow_pickle=False)
    M_T = np.load(M_T_fname, allow_pickle=False)
    Σ_T = np.load(Σ_T_fname, allow_pickle=False)
    Q_subseq_isconstant = np.load(Q_subseq_isconstant_fname, allow_pickle=False)
    T_subseq_isconstant = np.load(T_subseq_isconstant_fname, allow_pickle=False)

    nlevel = np.floor(np.log2(k) + 1).astype(np.int64)
    # number of levels in binary search tree from which `bfs` is constructed.

    with cuda.gpus[device_id]:
        device_T_A = cuda.to_device(T_A)
        device_QT_odd = cuda.to_device(QT)
        device_QT_even = cuda.to_device(QT)
        device_QT_first = cuda.to_device(QT_first)
        device_μ_Q = cuda.to_device(μ_Q)
        device_σ_Q = cuda.to_device(σ_Q)
        device_Q_subseq_isconstant = cuda.to_device(Q_subseq_isconstant)

        if ignore_trivial:
            device_T_B = device_T_A
            device_M_T = device_μ_Q
            device_Σ_T = device_σ_Q
            device_T_subseq_isconstant = device_Q_subseq_isconstant
        else:
            device_T_B = cuda.to_device(T_B)
            device_M_T = cuda.to_device(M_T)
            device_Σ_T = cuda.to_device(Σ_T)
            device_T_subseq_isconstant = cuda.to_device(T_subseq_isconstant)

        profile = np.full((w, k), np.inf, dtype=np.float64)
        indices = np.full((w, k), -1, dtype=np.int64)

        profile_L = np.full(w, np.inf, dtype=np.float64)
        indices_L = np.full(w, -1, dtype=np.int64)

        profile_R = np.full(w, np.inf, dtype=np.float64)
        indices_R = np.full(w, -1, dtype=np.int64)

        device_profile = cuda.to_device(profile)
        device_profile_L = cuda.to_device(profile_L)
        device_profile_R = cuda.to_device(profile_R)
        device_indices = cuda.to_device(indices)
        device_indices_L = cuda.to_device(indices_L)
        device_indices_R = cuda.to_device(indices_R)
        device_bfs = cuda.to_device(core._bfs_indices(k, fill_value=-1))

        _compute_and_update_PI_kernel[blocks_per_grid, threads_per_block](
            range_start - 1,
            device_T_A,
            device_T_B,
            m,
            device_QT_even,
            device_QT_odd,
            device_QT_first,
            device_μ_Q,
            device_σ_Q,
            device_M_T,
            device_Σ_T,
            device_Q_subseq_isconstant,
            device_T_subseq_isconstant,
            w,
            ignore_trivial,
            excl_zone,
            device_profile,
            device_profile_L,
            device_profile_R,
            device_indices,
            device_indices_L,
            device_indices_R,
            False,
            device_bfs,
            nlevel,
            k,
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
                device_μ_Q,
                device_σ_Q,
                device_M_T,
                device_Σ_T,
                device_Q_subseq_isconstant,
                device_T_subseq_isconstant,
                w,
                ignore_trivial,
                excl_zone,
                device_profile,
                device_profile_L,
                device_profile_R,
                device_indices,
                device_indices_L,
                device_indices_R,
                True,
                device_bfs,
                nlevel,
                k,
            )

        profile = device_profile.copy_to_host()
        profile_L = device_profile_L.copy_to_host()
        profile_R = device_profile_R.copy_to_host()
        indices = device_indices.copy_to_host()
        indices_L = device_indices_L.copy_to_host()
        indices_R = device_indices_R.copy_to_host()

        profile[:, :] = np.sqrt(profile)
        profile_L[:] = np.sqrt(profile_L)
        profile_R[:] = np.sqrt(profile_R)

        profile_fname = core.array_to_temp_file(profile)
        profile_L_fname = core.array_to_temp_file(profile_L)
        profile_R_fname = core.array_to_temp_file(profile_R)
        indices_fname = core.array_to_temp_file(indices)
        indices_L_fname = core.array_to_temp_file(indices_L)
        indices_R_fname = core.array_to_temp_file(indices_R)

    return (
        profile_fname,
        profile_L_fname,
        profile_R_fname,
        indices_fname,
        indices_L_fname,
        indices_R_fname,
    )


@core.non_normalized(gpu_aamp)
def gpu_stump(
    T_A,
    m,
    T_B=None,
    ignore_trivial=True,
    device_id=0,
    normalize=True,
    p=2.0,
    k=1,
    T_A_subseq_isconstant=None,
    T_B_subseq_isconstant=None,
):
    """
    Compute the z-normalized matrix profile with one or more GPU devices

    This is a convenience wrapper around the Numba ``cuda.jit`` ``_gpu_stump`` function
    which computes the matrix profile according to GPU-STOMP. The default number of
    threads-per-block is set to ``512`` and may be changed by setting the global
    parameter ``config.STUMPY_THREADS_PER_BLOCK`` to an appropriate number based on
    your GPU hardware.

    Parameters
    ----------
    T_A : numpy.ndarray
        The time series or sequence for which to compute the matrix profile.

    m : int
        Window size.

    T_B : numpy.ndarray, default None
        The time series or sequence that will be used to annotate ``T_A``. For every
        subsequence in ``T_A``, its nearest neighbor in ``T_B`` will be recorded.
        Default is  ``None`` which corresponds to a self-join.

    ignore_trivial : bool, default True
        Set to ``True`` if this is a self-join. Otherwise, for AB-join, set this
        to ``False``.

    device_id : int or list, default 0
        The (GPU) device number to use. The default value is ``0``. A list of
        valid device ids (``int``) may also be provided for parallel GPU-STUMP
        computation. A list of all valid device ids can be obtained by
        executing ``[device.id for device in numba.cuda.list_devices()]``.

    normalize : bool, default True
        When set to ``True``, this z-normalizes subsequences prior to computing
        distances. Otherwise, this function gets re-routed to its complementary
        non-normalized equivalent set in the ``@core.non_normalized`` function
        decorator.

    p : float, default 2.0
        The p-norm to apply for computing the Minkowski distance. Minkowski distance is
        typically used with ``p`` being ``1`` or ``2``, which correspond to the
        Manhattan distance and the Euclidean distance, respectively. This parameter is
        ignored when ``normalize == True``.

    k : int, default 1
        The number of top ``k`` smallest distances used to construct the matrix
        profile. Note that this will increase the total computational time and memory
        usage when ``k > 1``.

    T_A_subseq_isconstant : numpy.ndarray or function, default None
        A boolean array that indicates whether a subsequence in ``T_A`` is constant
        (``True``). Alternatively, a custom, user-defined function that returns a
        boolean array that indicates whether a subsequence in ``T_A`` is constant
        (``True``). The function must only take two arguments, ``a``, a 1-D array,
        and ``w``, the window size, while additional arguments may be specified
        by currying the user-defined function using ``functools.partial``. Any
        subsequence with at least one ``np.nan``/``np.inf`` will automatically have
        its corresponding value set to ``False`` in this boolean array.

    T_B_subseq_isconstant : numpy.ndarray or function, default None
        A boolean array that indicates whether a subsequence in ``T_B`` is constant
        (``True``). Alternatively, a custom, user-defined function that returns a
        boolean array that indicates whether a subsequence in ``T_B`` is constant
        (``True``). The function must only take two arguments, ``a``, a 1-D array,
        and ``w``, the window size, while additional arguments may be specified
        by currying the user-defined function using ``functools.partial``. Any
        subsequence with at least one ``np.nan``/``np.inf`` will automatically have
        its corresponding value set to ``False`` in this boolean array.

    Returns
    -------
    out : numpy.ndarray
        When ``k = 1`` (default), the first column consists of the matrix profile,
        the second column consists of the matrix profile indices, the third column
        consists of the left matrix profile indices, and the fourth column consists
        of the right matrix profile indices. However, when ``k > 1``, the output array
        will contain exactly ``2 * k + 2`` columns. The first ``k`` columns (i.e.,
        ``out[:, :k]``) consists of the top-k matrix profile, the next set of ``k``
        columns (i.e., ``out[:, k : 2 * k``]) consists of the corresponding top-k matrix
        profile indices, and the last two columns (i.e., ``out[:, 2 * k]`` and
        ``out[:, 2 * k + 1]`` or, equivalently, ``out[:, -2]`` and ``out[:, -1]``)
        correspond to the top-1 left matrix profile indices and the top-1 right matrix
        profile indices, respectively.

        |

        For convenience, the matrix profile (distances) and matrix profile indices can
        also be accessed via their corresponding named array attributes, ``.P_`` and
        ``.I_``,respectively. Similarly, the corresponding left matrix profile indices
        and right matrix profile indices may also be accessed via the ``.left_I_`` and
        ``.right_I_`` array attributes. See examples below.

    See Also
    --------
    stumpy.stump : Compute the z-normalized matrix profile
    stumpy.stumped : Compute the z-normalized matrix profile with a ``dask``/ ``ray``
        cluster
    stumpy.scrump : Compute an approximate z-normalized matrix profile

    Notes
    -----
    `DOI: 10.1109/ICDM.2016.0085 \
    <https://www.cs.ucr.edu/~eamonn/STOMP_GPU_final_submission_camera_ready.pdf>`__

    See Table II, Figure 5, and Figure 6

    Timeseries, ``T_A``, will be annotated with the distance location
    (or index) of all its subsequences in another times series, ``T_B``.

    Return: For every subsequence, ``Q``, in ``T_A``, you will get a distance
    and index for the closest subsequence in ``T_B``. Thus, the array
    returned will have length ``T_A.shape[0] - m + 1``. Additionally, the
    left and right matrix profiles are also returned.

    Note: Unlike in the Table II where ``T_A.shape`` is expected to be equal
    to ``T_B.shape``, this implementation is generalized so that the shapes of
    ``T_A`` and ``T_B`` can be different. In the case where ``T_A.shape == T_B.shape``,
    then our algorithm reduces down to the same algorithm found in Table II.

    Additionally, unlike STAMP where the exclusion zone is ``m``/2, the default
    exclusion zone for STOMP is ``m``/4 (See Definition 3 and Figure 3).

    For self-joins, set ``ignore_trivial = True`` in order to avoid the
    trivial match.

    Note that left and right matrix profiles are only available for self-joins.

    Examples
    --------
    >>> import stumpy
    >>> import numpy as np
    >>> from numba import cuda
    >>> if __name__ == "__main__":
    ...     all_gpu_devices = [device.id for device in cuda.list_devices()]
    ...     mp = stumpy.gpu_stump(
    ...         np.array([584., -11., 23., 79., 1001., 0., -19.]),
    ...         m=3,
    ...         device_id=all_gpu_devices)
    >>>     mp
    mparray([[0.11633857113691416, 4, -1, 4],
             [2.694073918063438, 3, -1, 3],
             [3.0000926340485923, 0, 0, 4],
             [2.694073918063438, 1, 1, -1],
             [0.11633857113691416, 0, 0, -1]], dtype=object)
    >>>
    >>>     mp.P_
    mparray([0.11633857, 2.69407392, 3.00009263, 2.69407392, 0.11633857])
    >>>     mp.I_
    mparray([4, 3, 0, 1, 0])
    """
    if T_B is None:  # Self join!
        T_B = T_A
        ignore_trivial = True
        T_B_subseq_isconstant = T_A_subseq_isconstant

    T_A, μ_Q, σ_Q, Q_subseq_isconstant = core.preprocess(
        T_A, m, T_subseq_isconstant=T_A_subseq_isconstant
    )
    T_B, M_T, Σ_T, T_subseq_isconstant = core.preprocess(
        T_B, m, T_subseq_isconstant=T_B_subseq_isconstant
    )

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

    ignore_trivial = core.check_ignore_trivial(T_A, T_B, ignore_trivial)
    if ignore_trivial:  # self-join
        core.check_window_size(
            m, max_size=min(T_A.shape[0], T_B.shape[0]), n=T_A.shape[0]
        )
    else:  # AB-join
        core.check_window_size(m, max_size=min(T_A.shape[0], T_B.shape[0]))

    n = T_B.shape[0]
    w = T_A.shape[0] - m + 1
    l = n - m + 1
    excl_zone = int(
        np.ceil(m / config.STUMPY_EXCL_ZONE_DENOM)
    )  # See Definition 3 and Figure 3

    T_A_fname = core.array_to_temp_file(T_A)
    T_B_fname = core.array_to_temp_file(T_B)
    μ_Q_fname = core.array_to_temp_file(μ_Q)
    σ_Q_fname = core.array_to_temp_file(σ_Q)
    M_T_fname = core.array_to_temp_file(M_T)
    Σ_T_fname = core.array_to_temp_file(Σ_T)
    Q_subseq_isconstant_fname = core.array_to_temp_file(Q_subseq_isconstant)
    T_subseq_isconstant_fname = core.array_to_temp_file(T_subseq_isconstant)

    if isinstance(device_id, int):
        device_ids = [device_id]
    else:
        device_ids = device_id

    profile = [None] * len(device_ids)
    indices = [None] * len(device_ids)

    profile_L = [None] * len(device_ids)
    indices_L = [None] * len(device_ids)

    profile_R = [None] * len(device_ids)
    indices_R = [None] * len(device_ids)

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
                    μ_Q_fname,
                    σ_Q_fname,
                    QT_fname,
                    QT_first_fname,
                    M_T_fname,
                    Σ_T_fname,
                    Q_subseq_isconstant_fname,
                    T_subseq_isconstant_fname,
                    w,
                    ignore_trivial,
                    start + 1,
                    device_ids[idx],
                    k,
                ),
            )
        else:
            # Execute last chunk in parent process
            # Only parent process is executed when a single GPU is requested
            (
                profile[idx],
                profile_L[idx],
                profile_R[idx],
                indices[idx],
                indices_L[idx],
                indices_R[idx],
            ) = _gpu_stump(
                T_A_fname,
                T_B_fname,
                m,
                stop,
                excl_zone,
                μ_Q_fname,
                σ_Q_fname,
                QT_fname,
                QT_first_fname,
                M_T_fname,
                Σ_T_fname,
                Q_subseq_isconstant_fname,
                T_subseq_isconstant_fname,
                w,
                ignore_trivial,
                start + 1,
                device_ids[idx],
                k,
            )

    # Clean up process pool for multi-GPU request
    if len(device_ids) > 1:  # pragma: no cover
        pool.close()
        pool.join()

        # Collect results from spawned child processes if they exist
        for idx, result in enumerate(results):
            if result is not None:
                (
                    profile[idx],
                    profile_L[idx],
                    profile_R[idx],
                    indices[idx],
                    indices_L[idx],
                    indices_R[idx],
                ) = result.get()

    os.remove(T_A_fname)
    os.remove(T_B_fname)
    os.remove(μ_Q_fname)
    os.remove(σ_Q_fname)
    os.remove(M_T_fname)
    os.remove(Σ_T_fname)
    os.remove(Q_subseq_isconstant_fname)
    os.remove(T_subseq_isconstant_fname)
    for QT_fname in QT_fnames:
        os.remove(QT_fname)
    for QT_first_fname in QT_first_fnames:
        os.remove(QT_first_fname)

    for idx in range(len(device_ids)):
        profile_fname = profile[idx]
        profile_L_fname = profile_L[idx]
        profile_R_fname = profile_R[idx]
        indices_fname = indices[idx]
        indices_L_fname = indices_L[idx]
        indices_R_fname = indices_R[idx]

        profile[idx] = np.load(profile_fname, allow_pickle=False)
        profile_L[idx] = np.load(profile_L_fname, allow_pickle=False)
        profile_R[idx] = np.load(profile_R_fname, allow_pickle=False)
        indices[idx] = np.load(indices_fname, allow_pickle=False)
        indices_L[idx] = np.load(indices_L_fname, allow_pickle=False)
        indices_R[idx] = np.load(indices_R_fname, allow_pickle=False)

        os.remove(profile_fname)
        os.remove(profile_L_fname)
        os.remove(profile_R_fname)
        os.remove(indices_fname)
        os.remove(indices_L_fname)
        os.remove(indices_R_fname)

    for i in range(1, len(device_ids)):  # pragma: no cover
        # Update (top-k) matrix profile and matrix profile indices
        core._merge_topk_PI(profile[0], profile[i], indices[0], indices[i])

        # Update (top-1) left matrix profile and matrix profile indices
        mask = profile_L[0] > profile_L[i]
        profile_L[0][mask] = profile_L[i][mask]
        indices_L[0][mask] = indices_L[i][mask]

        # Update (top-1) right matrix profile and matrix profile indices
        mask = profile_R[0] > profile_R[i]
        profile_R[0][mask] = profile_R[i][mask]
        indices_R[0][mask] = indices_R[i][mask]

    out = np.empty((w, 2 * k + 2), dtype=object)  # last two columns are to store
    # (top-1) left/right matrix profile indices
    out[:, :k] = profile[0]
    out[:, k:] = np.column_stack((indices[0], indices_L[0], indices_R[0]))

    core._check_P(out[:, 0])

    return mparray(out, m, k, config.STUMPY_EXCL_ZONE_DENOM)
