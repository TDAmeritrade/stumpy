# STUMPY
# Copyright 2019 TD Ameritrade. Released under the terms of the 3-Clause BSD license.
# STUMPY is a trademark of TD Ameritrade IP Company, Inc. All rights reserved.

import numba
import numpy as np
from numba import njit, prange

from . import config, core
from .aamp import aamp
from .mparray import mparray


@njit(
    # "(f8[:], f8[:], i8, f8[:], f8[:], f8[:], f8[:], f8[:], f8[:], f8[:], f8[:],"
    # "b1[:], b1[:], b1[:], b1[:], i8[:], i8, i8, i8, f8[:, :, :], f8[:, :],"
    # "f8[:, :], i8[:, :, :], i8[:, :], i8[:, :], b1)",
    fastmath=config.STUMPY_FASTMATH_FLAGS,
)
def _compute_diagonal(
    T_A,
    T_B,
    m,
    μ_Q,
    M_T,
    σ_Q_inverse,
    Σ_T_inverse,
    cov_a,
    cov_b,
    cov_c,
    cov_d,
    T_A_subseq_isfinite,
    T_B_subseq_isfinite,
    T_A_subseq_isconstant,
    T_B_subseq_isconstant,
    diags,
    diags_start_idx,
    diags_stop_idx,
    thread_idx,
    ρ,
    ρL,
    ρR,
    I,
    IL,
    IR,
    ignore_trivial,
):
    """
    Compute (Numba JIT-compiled) and update the (top-k) Pearson correlation (ρ),
    ρL, ρR, I, IL, and IR sequentially along individual diagonals using a single
    thread and avoiding race conditions.

    Parameters
    ----------
    T_A : numpy.ndarray
        The time series or sequence for which to compute the matrix profile

    T_B : numpy.ndarray
        The time series or sequence that will be used to annotate `T_A`. For every
        subsequence in `T_A`, its nearest neighbor in `T_B` will be recorded.

    m : int
        Window size

    μ_Q : numpy.ndarray
        Mean of the query sequence, `Q`, relative to the current sliding window

    M_T : numpy.ndarray
        Sliding mean of time series, `T`

    σ_Q_inverse : numpy.ndarray
        Inverse standard deviation of the query sequence, `Q`, relative to the current
        sliding window

    Σ_T_inverse : numpy.ndarray
        Inverse sliding standard deviation of time series, `T`

    cov_a : numpy.ndarray
        The first covariance term relating T_A[i + g + m - 1] and M_T_m_1[i + g]

    cov_b : numpy.ndarray
        The second covariance term relating T_B[i + m - 1] and μ_Q_m_1[i]

    cov_c : numpy.ndarray
        The third covariance term relating T_A[i + g - 1] and M_T_m_1[i + g]

    cov_d : numpy.ndarray
        The fourth covariance term relating T_B[i - 1] and μ_Q_m_1[i]

    T_A_subseq_isfinite : numpy.ndarray
        A boolean array that indicates whether a subsequence in `T_A` contains a
        `np.nan`/`np.inf` value (False)

    T_B_subseq_isfinite : numpy.ndarray
        A boolean array that indicates whether a subsequence in `T_B` contains a
        `np.nan`/`np.inf` value (False)

    T_A_subseq_isconstant : numpy.ndarray
        A boolean array that indicates whether a subsequence in `T_A` is constant (True)

    T_B_subseq_isconstant : numpy.ndarray
        A boolean array that indicates whether a subsequence in `T_B` is constant (True)

    diags : numpy.ndarray
        The diagonal indices

    diags_start_idx : int
        The starting (inclusive) diagonal index

    diags_stop_idx : int
        The stopping (exclusive) diagonal index

    thread_idx : int
        The thread index

    ρ : numpy.ndarray
        The (top-k) Pearson correlations, sorted in ascending order per row

    ρL : numpy.ndarray
        The top-1 left Pearson correlations

    ρR : numpy.ndarray
        The top-1 right Pearson correlations

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

    Notes
    -----
    `DOI: 10.1007/s10115-017-1138-x \
    <https://www.cs.ucr.edu/~eamonn/ten_quadrillion.pdf>`__

    See Section 4.5

    The above reference outlines a general approach for traversing the distance
    matrix in a diagonal fashion rather than in a row-wise fashion.

    `DOI: 10.1145/3357223.3362721 \
    <https://www.cs.ucr.edu/~eamonn/public/GPU_Matrix_profile_VLDB_30DraftOnly.pdf>`__

    See Section 3.1 and Section 3.3

    The above reference outlines the use of the Pearson correlation via Welford's
    centered sum-of-products along each diagonal of the distance matrix in place of the
    sliding window dot product found in the original STOMP method.
    """
    n_A = T_A.shape[0]
    n_B = T_B.shape[0]
    m_inverse = 1.0 / m
    constant = (m - 1) * m_inverse * m_inverse  # (m - 1)/(m * m)
    uint64_m = np.uint64(m)

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
                cov = (
                    np.dot(
                        (T_B[uint64_j : uint64_j + uint64_m] - M_T[uint64_j]),
                        (T_A[uint64_i : uint64_i + uint64_m] - μ_Q[uint64_i]),
                    )
                    * m_inverse
                )
            else:
                # The next lines are equivalent and left for reference
                # cov = cov + constant * (
                #     (T_B[i + g + m - 1] - M_T_m_1[i + g])
                #     * (T_A[i + m - 1] - μ_Q_m_1[i])
                #     - (T_B[i + g - 1] - M_T_m_1[i + g]) * (T_A[i - 1] - μ_Q_m_1[i])
                # )
                cov = cov + constant * (
                    cov_a[uint64_j] * cov_b[uint64_i]
                    - cov_c[uint64_j] * cov_d[uint64_i]
                )

            if T_B_subseq_isfinite[uint64_j] and T_A_subseq_isfinite[uint64_i]:
                # Neither subsequence contains NaNs
                if T_B_subseq_isconstant[uint64_j] and T_A_subseq_isconstant[uint64_i]:
                    pearson = 1.0
                elif T_B_subseq_isconstant[uint64_j] or T_A_subseq_isconstant[uint64_i]:
                    pearson = 0.5
                else:
                    pearson = cov * Σ_T_inverse[uint64_j] * σ_Q_inverse[uint64_i]
                    pearson = min(1.0, pearson)

                # `ρ[thread_idx, i, :]` is sorted in ascending order and MUST be updated
                # when the newly-calculated `pearson` value becomes greater than the
                # first (i.e. smallest) element in this array. Note that a higher
                # pearson value corresponds to a lower distance.
                if pearson > ρ[thread_idx, uint64_i, 0]:
                    idx = np.searchsorted(ρ[thread_idx, uint64_i], pearson)
                    core._shift_insert_at_index(
                        ρ[thread_idx, uint64_i], idx, pearson, shift="left"
                    )
                    core._shift_insert_at_index(
                        I[thread_idx, uint64_i], idx, uint64_j, shift="left"
                    )

                if ignore_trivial:  # self-joins only
                    if pearson > ρ[thread_idx, uint64_j, 0]:
                        idx = np.searchsorted(ρ[thread_idx, uint64_j], pearson)
                        core._shift_insert_at_index(
                            ρ[thread_idx, uint64_j], idx, pearson, shift="left"
                        )
                        core._shift_insert_at_index(
                            I[thread_idx, uint64_j], idx, uint64_i, shift="left"
                        )

                    if uint64_i < uint64_j:
                        # left pearson correlation and left matrix profile index
                        if pearson > ρL[thread_idx, uint64_j]:
                            ρL[thread_idx, uint64_j] = pearson
                            IL[thread_idx, uint64_j] = uint64_i

                        # right pearson correlation and right matrix profile index
                        if pearson > ρR[thread_idx, uint64_i]:
                            ρR[thread_idx, uint64_i] = pearson
                            IR[thread_idx, uint64_i] = uint64_j

    return


@njit(
    # "(f8[:], f8[:], i8, f8[:], f8[:], f8[:], f8[:], f8[:], f8[:], b1[:], b1[:],"
    # "b1[:], b1[:], i8[:], b1, i8)",
    parallel=True,
    fastmath=config.STUMPY_FASTMATH_FLAGS,
)
def _stump(
    T_A,
    T_B,
    m,
    μ_Q,
    M_T,
    σ_Q_inverse,
    Σ_T_inverse,
    μ_Q_m_1,
    M_T_m_1,
    T_A_subseq_isfinite,
    T_B_subseq_isfinite,
    T_A_subseq_isconstant,
    T_B_subseq_isconstant,
    diags,
    ignore_trivial,
    k,
):
    """
    A Numba JIT-compiled version of STOMPopt with Pearson correlations for parallel
    computation of the (top-k) matrix profile, the (top-k) matrix profile indices,
    the top-1 left matrix profile and its matrix profile index, and the top-1 right
    matrix profile and its matrix profile index.

    Parameters
    ----------
    T_A : numpy.ndarray
        The time series or sequence for which to compute the matrix profile

    T_B : numpy.ndarray
        The time series or sequence that will be used to annotate `T_A`. For every
        subsequence in `T_A`, its nearest neighbor in `T_B` will be recorded.

    m : int
        Window size

    μ_Q : numpy.ndarray
        Mean of the query sequence, `Q`, relative to the current sliding window

    M_T : numpy.ndarray
        Sliding mean of time series, `T`

    σ_Q_inverse : numpy.ndarray
        Inverse standard deviation of the query sequence, `Q`, relative to the current
        sliding window

    Σ_T_inverse : numpy.ndarray
        Inverse sliding standard deviation of time series, `T`

    μ_Q_m_1 : numpy.ndarray
        Mean of the query sequence, `Q`, relative to the current sliding window and
        using a window size of `m-1`

    M_T_m_1 : numpy.ndarray
        Sliding mean of time series, `T`, using a window size of `m-1`

    T_A_subseq_isfinite : numpy.ndarray
        A boolean array that indicates whether a subsequence in `T_A` contains a
        `np.nan`/`np.inf` value (False)

    T_B_subseq_isfinite : numpy.ndarray
        A boolean array that indicates whether a subsequence in `T_B` contains a
        `np.nan`/`np.inf` value (False)

    T_A_subseq_isconstant : numpy.ndarray
        A boolean array that indicates whether a subsequence in `T_A` is constant (True)

    T_B_subseq_isconstant : numpy.ndarray
        A boolean array that indicates whether a subsequence in `T_B` is constant (True)

    diags : numpy.ndarray
        The diagonal indices

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
    `DOI: 10.1007/s10115-017-1138-x \
    <https://www.cs.ucr.edu/~eamonn/ten_quadrillion.pdf>`__

    See Section 4.5

    The above reference outlines a general approach for traversing the distance
    matrix in a diagonal fashion rather than in a row-wise fashion.

    `DOI: 10.1145/3357223.3362721 \
    <https://www.cs.ucr.edu/~eamonn/public/GPU_Matrix_profile_VLDB_30DraftOnly.pdf>`__

    See Section 3.1 and Section 3.3

    The above reference outlines the use of the Pearson correlation via Welford's
    centered sum-of-products along each diagonal of the distance matrix in place of the
    sliding window dot product found in the original STOMP method.

    `DOI: 10.1109/ICDM.2016.0085 \
    <https://www.cs.ucr.edu/~eamonn/STOMP_GPU_final_submission_camera_ready.pdf>`__

    See Table II

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
    """
    n_A = T_A.shape[0]
    n_B = T_B.shape[0]
    l = n_A - m + 1
    n_threads = numba.config.NUMBA_NUM_THREADS

    ρ = np.full((n_threads, l, k), -np.inf, dtype=np.float64)
    I = np.full((n_threads, l, k), -1, dtype=np.int64)

    ρL = np.full((n_threads, l), -np.inf, dtype=np.float64)
    IL = np.full((n_threads, l), -1, dtype=np.int64)

    ρR = np.full((n_threads, l), -np.inf, dtype=np.float64)
    IR = np.full((n_threads, l), -1, dtype=np.int64)

    ndist_counts = core._count_diagonal_ndist(diags, m, n_A, n_B)
    diags_ranges = core._get_array_ranges(ndist_counts, n_threads, False)

    cov_a = T_B[m - 1 :] - M_T_m_1[:-1]
    cov_b = T_A[m - 1 :] - μ_Q_m_1[:-1]
    # The next lines are equivalent and left for reference
    # cov_c = np.roll(T_A, 1)
    # cov_ = cov_c[:M_T_m_1.shape[0]] - M_T_m_1[:]
    cov_c = np.empty(M_T_m_1.shape[0], dtype=np.float64)
    cov_c[1:] = T_B[: M_T_m_1.shape[0] - 1]
    cov_c[0] = T_B[-1]
    cov_c[:] = cov_c - M_T_m_1
    # The next lines are equivalent and left for reference
    # cov_d = np.roll(T_B, 1)
    # cov_d = cov_d[:μ_Q_m_1.shape[0]] - μ_Q_m_1[:]
    cov_d = np.empty(μ_Q_m_1.shape[0], dtype=np.float64)
    cov_d[1:] = T_A[: μ_Q_m_1.shape[0] - 1]
    cov_d[0] = T_A[-1]
    cov_d[:] = cov_d - μ_Q_m_1

    for thread_idx in prange(n_threads):
        # Compute and update pearson correlations and matrix profile indices
        # within a single thread and avoiding race conditions
        _compute_diagonal(
            T_A,
            T_B,
            m,
            μ_Q,
            M_T,
            σ_Q_inverse,
            Σ_T_inverse,
            cov_a,
            cov_b,
            cov_c,
            cov_d,
            T_A_subseq_isfinite,
            T_B_subseq_isfinite,
            T_A_subseq_isconstant,
            T_B_subseq_isconstant,
            diags,
            diags_ranges[thread_idx, 0],
            diags_ranges[thread_idx, 1],
            thread_idx,
            ρ,
            ρL,
            ρR,
            I,
            IL,
            IR,
            ignore_trivial,
        )

    # Reduction of results from all threads
    for thread_idx in range(1, n_threads):
        # update top-k arrays
        core._merge_topk_ρI(ρ[0], ρ[thread_idx], I[0], I[thread_idx])

        # update left matrix profile and matrix profile indices
        mask = ρL[0] < ρL[thread_idx]
        ρL[0][mask] = ρL[thread_idx][mask]
        IL[0][mask] = IL[thread_idx][mask]

        # update right matrix profile and matrix profile indices
        mask = ρR[0] < ρR[thread_idx]
        ρR[0][mask] = ρR[thread_idx][mask]
        IR[0][mask] = IR[thread_idx][mask]

    # Reverse top-k rho (and its associated I) to be in descending order and
    # then convert from Pearson correlations to Euclidean distances (ascending order)
    p_norm = np.abs(2 * m * (1 - ρ[0, :, ::-1]))
    I = I[0, :, ::-1]

    p_norm_L = np.abs(2 * m * (1 - ρL[0, :]))
    p_norm_R = np.abs(2 * m * (1 - ρR[0, :]))

    for i in prange(p_norm.shape[0]):
        for j in range(p_norm.shape[1]):
            if p_norm[i, j] < config.STUMPY_P_NORM_THRESHOLD:
                p_norm[i, j] = 0.0

        if p_norm_L[i] < config.STUMPY_P_NORM_THRESHOLD:
            p_norm_L[i] = 0.0

        if p_norm_R[i] < config.STUMPY_P_NORM_THRESHOLD:
            p_norm_R[i] = 0.0

    return (
        np.sqrt(p_norm),
        np.sqrt(p_norm_L),
        np.sqrt(p_norm_R),
        I,
        IL[0],
        IR[0],
    )


@core.non_normalized(
    aamp,
    exclude=["normalize", "p", "T_A_subseq_isconstant", "T_B_subseq_isconstant"],
)
def stump(
    T_A,
    m,
    T_B=None,
    ignore_trivial=True,
    normalize=True,
    p=2.0,
    k=1,
    T_A_subseq_isconstant=None,
    T_B_subseq_isconstant=None,
):
    """
    Compute the z-normalized matrix profile

    This is a convenience wrapper around the Numba JIT-compiled parallelized
    ``_stump`` function which computes the (top-k) matrix profile according to
    STOMPopt with Pearson correlations.

    Parameters
    ----------
    T_A : numpy.ndarray
        The time series or sequence for which to compute the matrix profile.

    m : int
        Window size.

    T_B : numpy.ndarray, default None
        The time series or sequence that will be used to annotate ``T_A``. For every
        subsequence in ``T_A``, its nearest neighbor in ``T_B`` will be recorded.
        Default is ``None`` which corresponds to a self-join.

    ignore_trivial : bool, default True
        Set to ``True`` if this is a self-join. Otherwise, for AB-join, set this
        to ``False``.

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
        usage when ``k > 1``. If you have access to a GPU device, then you may be able
        to leverage ``gpu_stump`` for better performance and scalability.

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
        columns (i.e., ``out[:, k : 2 * k]``) consists of the corresponding top-k
        matrix profile indices, and the last two columns (i.e., ``out[:, 2 * k]`` and
        ``out[:, 2 * k + 1]`` or, equivalently, ``out[:, -2]`` and ``out[:, -1]``)
        correspond to the top-1 left matrix profile indices and the top-1 right matrix
        profile indices, respectively.

        |

        For convenience, the matrix profile (distances) and matrix profile indices can
        also be accessed via their corresponding named array attributes, ``.P_`` and
        ``.I_``,respectively. Similarly, the corresponding left matrix profile indices
        and right matrix profile indices may also be accessed via the ``.left_I_`` and
        ``.right_I_`` array attributes.  See examples below.

    See Also
    --------
    stumpy.stumped : Compute the z-normalized matrix profile with a ``dask``/ ``ray``
        cluster
    stumpy.gpu_stump : Compute the z-normalized matrix profile with one or more GPU
        devices
    stumpy.scrump : Compute an approximate z-normalized matrix profile

    Notes
    -----
    `DOI: 10.1007/s10115-017-1138-x \
    <https://www.cs.ucr.edu/~eamonn/ten_quadrillion.pdf>`__

    See Section 4.5

    The above reference outlines a general approach for traversing the distance
    matrix in a diagonal fashion rather than in a row-wise fashion.

    `DOI: 10.1145/3357223.3362721 \
    <https://www.cs.ucr.edu/~eamonn/public/GPU_Matrix_profile_VLDB_30DraftOnly.pdf>`__

    See Section 3.1 and Section 3.3

    The above reference outlines the use of the Pearson correlation via Welford's
    centered sum-of-products along each diagonal of the distance matrix in place of the
    sliding window dot product found in the original STOMP method.

    `DOI: 10.1109/ICDM.2016.0085 \
    <https://www.cs.ucr.edu/~eamonn/STOMP_GPU_final_submission_camera_ready.pdf>`__

    See Table II

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
    >>> mp = stumpy.stump(np.array([584., -11., 23., 79., 1001., 0., -19.]), m=3)
    >>> mp
    mparray([[0.11633857113691416, 4, -1, 4],
             [2.694073918063438, 3, -1, 3],
             [3.0000926340485923, 0, 0, 4],
             [2.694073918063438, 1, 1, -1],
             [0.11633857113691416, 0, 0, -1]], dtype=object)
    >>>
    >>> mp.P_
    mparray([0.11633857, 2.69407392, 3.00009263, 2.69407392, 0.11633857])
    >>> mp.I_
    mparray([4, 3, 0, 1, 0])
    """
    if T_B is None:
        ignore_trivial = True
        T_B = T_A
        T_B_subseq_isconstant = T_A_subseq_isconstant

    (
        T_A,
        μ_Q,
        σ_Q_inverse,
        μ_Q_m_1,
        T_A_subseq_isfinite,
        T_A_subseq_isconstant,
    ) = core.preprocess_diagonal(T_A, m, T_subseq_isconstant=T_A_subseq_isconstant)

    (
        T_B,
        M_T,
        Σ_T_inverse,
        M_T_m_1,
        T_B_subseq_isfinite,
        T_B_subseq_isconstant,
    ) = core.preprocess_diagonal(T_B, m, T_subseq_isconstant=T_B_subseq_isconstant)

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

    P, PL, PR, I, IL, IR = _stump(
        T_A,
        T_B,
        m,
        μ_Q,
        M_T,
        σ_Q_inverse,
        Σ_T_inverse,
        μ_Q_m_1,
        M_T_m_1,
        T_A_subseq_isfinite,
        T_B_subseq_isfinite,
        T_A_subseq_isconstant,
        T_B_subseq_isconstant,
        diags,
        ignore_trivial,
        k,
    )

    out = np.empty((l, 2 * k + 2), dtype=object)  # last two columns are to
    # store left and right matrix profile indices
    out[:, :k] = P
    out[:, k:] = np.column_stack((I, IL, IR))

    core._check_P(out[:, 0])

    return mparray(out, m, k, config.STUMPY_EXCL_ZONE_DENOM)
