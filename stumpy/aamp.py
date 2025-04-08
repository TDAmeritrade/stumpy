# STUMPY
# Copyright 2019 TD Ameritrade. Released under the terms of the 3-Clause BSD license.
# STUMPY is a trademark of TD Ameritrade IP Company, Inc. All rights reserved.

import numba
import numpy as np
from numba import njit, prange

from . import config, core
from .mparray import mparray


@njit(
    # "(f8[:], f8[:], i8, b1[:], b1[:], f8, i8[:], i8, i8, i8, f8[:, :, :],"
    # "f8[:, :], f8[:, :], i8[:, :, :], i8[:, :], i8[:, :], b1)",
    fastmath=config.STUMPY_FASTMATH_FLAGS,
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
    PL,
    PR,
    I,
    IL,
    IR,
    ignore_trivial,
):
    """
    Compute (Numba JIT-compiled) and update the (top-k) matrix profile P,
    PL, PR, I, IL, and IR sequentially along individual diagonals using a single
    thread and avoiding race conditions.

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
        The p-norm to apply for computing the Minkowski distance. Minkowski distance is
        typically used with `p` being 1 or 2, which correspond to the Manhattan distance
        and the Euclidean distance, respectively.

    diags : numpy.ndarray
        The diag of diagonals to process and compute

    diags_start_idx : int
        The start index for a range of diagonal diag to process and compute

    diags_stop_idx : int
        The (exclusive) stop index for a range of diagonal diag to process and compute

    thread_idx : int
        The thread index

    P : numpy.ndarray
        The (top-k) matrix profile, sorted in ascending order per row

    PL : numpy.ndarray
        The top-1 left marix profile

    PR : numpy.ndarray
        The top-1 right marix profile

    I : numpy.ndarray
        The (top-k) matrix profile indices

    IL : numpy.ndarray
        The top-1 left matrix profile indices

    IR : numpy.ndarray
        The top-1 right matrix profile indices

    ignore_trivial : bool
        Set to `True` if this is a self-join. Otherwise, for AB-join, set this to
        `False`. Default is `True`.

    Returns
    -------
    None
    """
    n_A = T_A.shape[0]
    n_B = T_B.shape[0]
    uint64_m = np.uint64(m)
    uint64_1 = np.uint64(1)

    for diag_idx in range(diags_start_idx, diags_stop_idx):
        g = diags[diag_idx]

        if g >= 0:
            iter_range = range(0, min(n_A - m + 1, n_B - m + 1 - g))
        else:
            iter_range = range(-g, min(n_A - m + 1, n_B - m + 1 - g))

        for i in iter_range:
            uint64_i = np.uint64(i)
            uint64_j = np.uint64(i + g)

            if uint64_i == 0 or uint64_j == 0:
                p_norm = (
                    np.linalg.norm(
                        T_B[uint64_j : uint64_j + uint64_m]
                        - T_A[uint64_i : uint64_i + uint64_m],
                        ord=p,
                    )
                    ** p
                )
            else:
                p_norm = np.abs(
                    p_norm
                    - np.absolute(T_B[uint64_j - uint64_1] - T_A[uint64_i - uint64_1])
                    ** p
                    + np.absolute(
                        T_B[uint64_j + uint64_m - uint64_1]
                        - T_A[uint64_i + uint64_m - uint64_1]
                    )
                    ** p
                )

            if p_norm < config.STUMPY_P_NORM_THRESHOLD:
                p_norm = 0.0

            if T_A_subseq_isfinite[uint64_i] and T_B_subseq_isfinite[uint64_j]:
                # Neither subsequence contains NaNs

                # `P[thread_idx, i, :]` is sorted in ascending order and MUST be updated
                # when the newly-calculated `p_norm` value becomes smaller than the
                # last (i.e. greatest) element in this array. Note that the goal
                # is to have top-k smallest distances for each subsequence.
                if p_norm < P[thread_idx, uint64_i, -1]:
                    idx = np.searchsorted(P[thread_idx, uint64_i], p_norm)
                    core._shift_insert_at_index(
                        P[thread_idx, uint64_i], idx, p_norm, shift="right"
                    )
                    core._shift_insert_at_index(
                        I[thread_idx, uint64_i], idx, uint64_j, shift="right"
                    )

                if ignore_trivial:  # self-joins only
                    if p_norm < P[thread_idx, uint64_j, -1]:
                        idx = np.searchsorted(P[thread_idx, uint64_j], p_norm)
                        core._shift_insert_at_index(
                            P[thread_idx, uint64_j], idx, p_norm, shift="right"
                        )
                        core._shift_insert_at_index(
                            I[thread_idx, uint64_j], idx, uint64_i, shift="right"
                        )

                    if uint64_i < uint64_j:
                        # left matrix profile and left matrix profile index
                        if p_norm < PL[thread_idx, uint64_j]:
                            PL[thread_idx, uint64_j] = p_norm
                            IL[thread_idx, uint64_j] = uint64_i

                        # right matrix profile and right matrix profile index
                        if p_norm < PR[thread_idx, uint64_i]:
                            PR[thread_idx, uint64_i] = p_norm
                            IR[thread_idx, uint64_i] = uint64_j

    return


@njit(
    # "(f8[:], f8[:], i8, b1[:], b1[:], i8[:], b1, i8)",
    parallel=True,
    fastmath=config.STUMPY_FASTMATH_FLAGS,
)
def _aamp(
    T_A,
    T_B,
    m,
    T_A_subseq_isfinite,
    T_B_subseq_isfinite,
    p,
    diags,
    ignore_trivial,
    k,
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
        The p-norm to apply for computing the Minkowski distance. Minkowski distance is
        typically used with `p` being 1 or 2, which correspond to the Manhattan distance
        and the Euclidean distance, respectively.

    diags : numpy.ndarray
        The diag of diagonals to process and compute

    ignore_trivial : bool
        Set to `True` if this is a self-join. Otherwise, for AB-join, set this to
        `False`. Default is `True`.

    k : int
        The number of top `k` smallest distances used to construct the matrix profile.
        Note that this will increase the total computational time and memory usage
        when k > 1.

    Returns
    -------
    out1 : numpy.ndarray
        The (top-k) matrix profile

    out2 : numpy.ndarray
        The (top-1) left matrix profile

    out3 : numpy.ndarray
        The (top-1) right matrix profile

    out4 : numpy.ndarray
        The (top-k) matrix profile indices

    out5 : numpy.ndarray
        The (top-1) left matrix profile indices

    out6 : numpy.ndarray
        The (top-1) right matrix profile indices

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

    P = np.full((n_threads, l, k), np.inf, dtype=np.float64)
    I = np.full((n_threads, l, k), -1, dtype=np.int64)

    PL = np.full((n_threads, l), np.inf, dtype=np.float64)
    IL = np.full((n_threads, l), -1, dtype=np.int64)

    PR = np.full((n_threads, l), np.inf, dtype=np.float64)
    IR = np.full((n_threads, l), -1, dtype=np.int64)

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
            PL,
            PR,
            I,
            IL,
            IR,
            ignore_trivial,
        )

    # Reduction of results from all threads
    for thread_idx in range(1, n_threads):
        # update top-k arrays
        core._merge_topk_PI(P[0], P[thread_idx], I[0], I[thread_idx])

        # update left matrix profile and matrix profile indices
        mask = PL[0] > PL[thread_idx]
        PL[0][mask] = PL[thread_idx][mask]
        IL[0][mask] = IL[thread_idx][mask]

        # update right matrix profile and matrix profile indices
        mask = PR[0] > PR[thread_idx]
        PR[0][mask] = PR[thread_idx][mask]
        IR[0][mask] = IR[thread_idx][mask]

    return (
        np.power(P[0], 1.0 / p),
        np.power(PL[0], 1.0 / p),
        np.power(PR[0], 1.0 / p),
        I[0],
        IL[0],
        IR[0],
    )


def aamp(T_A, m, T_B=None, ignore_trivial=True, p=2.0, k=1):
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
        The p-norm to apply for computing the Minkowski distance. Minkowski distance is
        typically used with `p` being 1 or 2, which correspond to the Manhattan distance
        and the Euclidean distance, respectively.

    k : int, default 1
        The number of top `k` smallest distances used to construct the matrix profile.
        Note that this will increase the total computational time and memory usage
        when k > 1.

    Returns
    -------
    out : numpy.ndarray
        When k = 1 (default), the first column consists of the matrix profile,
        the second column consists of the matrix profile indices, the third column
        consists of the left matrix profile indices, and the fourth column consists
        of the right matrix profile indices. However, when k > 1, the output array
        will contain exactly 2 * k + 2 columns. The first k columns (i.e., out[:, :k])
        consists of the top-k matrix profile, the next set of k columns
        (i.e., out[:, k:2k]) consists of the corresponding top-k matrix profile
        indices, and the last two columns (i.e., out[:, 2k] and out[:, 2k+1] or,
        equivalently, out[:, -2] and out[:, -1]) correspond to the top-1 left
        matrix profile indices and the top-1 right matrix profile indices, respectively.

        For convenience, the matrix profile (distances) and matrix profile indices can
        also be accessed via their corresponding named array attributes, `.P_` and
        `.I_`,respectively. Similarly, the corresponding left matrix profile indices
        and right matrix profile indices may also be accessed via the `.left_I_` and
        `.right_I_` array attributes.

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

    n_A = T_A.shape[0]
    n_B = T_B.shape[0]
    l = n_A - m + 1

    ignore_trivial = core.check_ignore_trivial(T_A, T_B, ignore_trivial)
    excl_zone = int(np.ceil(m / config.STUMPY_EXCL_ZONE_DENOM))
    if ignore_trivial:  # self-join
        core.check_window_size(m, max_size=min(n_A, n_B), n=n_A)
        diags = np.arange(excl_zone + 1, n_A - m + 1, dtype=np.int64)
    else:  # AB-join
        core.check_window_size(m, max_size=min(n_A, n_B))
        diags = np.arange(-(n_A - m + 1) + 1, n_B - m + 1, dtype=np.int64)

    P, PL, PR, I, IL, IR = _aamp(
        T_A,
        T_B,
        m,
        T_A_subseq_isfinite,
        T_B_subseq_isfinite,
        p,
        diags,
        ignore_trivial,
        k,
    )

    out = np.empty((l, 2 * k + 2), dtype=object)
    out[:, :k] = P
    out[:, k:] = np.column_stack((I, IL, IR))

    core._check_P(out[:, 0])

    return mparray(out, m, k, config.STUMPY_EXCL_ZONE_DENOM)
