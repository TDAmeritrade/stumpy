# STUMPY
# Copyright 2019 TD Ameritrade. Released under the terms of the 3-Clause BSD license.
# STUMPY is a trademark of TD Ameritrade IP Company, Inc. All rights reserved.

import numba
import numpy as np
from numba import njit, prange

from . import config, core
from .scraamp import prescraamp, scraamp
from .stump import _stump


def _preprocess_prescrump(
    T_A, m, T_B=None, s=None, T_A_subseq_isconstant=None, T_B_subseq_isconstant=None
):
    """
    Performs several preprocessings and returns outputs that are needed for the
    prescrump algorithm.

    Parameters
    ----------
    T_A : numpy.ndarray
        The time series or sequence for which to compute the matrix profile

    m : int
        Window size

    T_B : numpy.ndarray, default None
        The time series or sequence that will be used to annotate T_A. For every
        subsequence in T_A, its nearest neighbor in T_B will be recorded.

    s : int, default None
        The sampling interval that defaults to
        `int(np.ceil(m / config.STUMPY_EXCL_ZONE_DENOM))`

    T_A_subseq_isconstant : numpy.ndarray or function, default None
        A boolean array that indicates whether a subsequence in `T_A` is constant
        (True). Alternatively, a custom, user-defined function that returns a
        boolean array that indicates whether a subsequence in `T_A` is constant
        (True). The function must only take two arguments, `a`, a 1-D array,
        and `w`, the window size, while additional arguments may be specified
        by currying the user-defined function using `functools.partial`. Any
        subsequence with at least one np.nan/np.inf will automatically have its
        corresponding value set to False in this boolean array.

    T_B_subseq_isconstant : numpy.ndarray or function, default None
        A boolean array that indicates whether a subsequence in `T_B` is constant
        (True). Alternatively, a custom, user-defined function that returns a
        boolean array that indicates whether a subsequence in `T_B` is constant
        (True). The function must only take two arguments, `a`, a 1-D array,
        and `w`, the window size, while additional arguments may be specified
        by currying the user-defined function using `functools.partial`. Any
        subsequence with at least one np.nan/np.inf will automatically have its
        corresponding value set to False in this boolean array.

    Returns
    -------
    T_A : numpy.ndarray
        A copy of the time series input `T_A`, where all NaN and inf values
        are replaced with zero.

    T_B : numpy.ndarray
        A copy of the time series input `T_B`, where all NaN and inf values
        are replaced with zero. If the input `T_B` is not provided (default),
        this array is just a copy of `T_A`.

    μ_Q : numpy.ndarray
        Sliding window mean for `T_A`

    σ_Q : numpy.ndarray
        Sliding window standard deviation for `T_A`

    M_T : numpy.ndarray
        Sliding window mean for `T_B`

    Σ_T : numpy.ndarray
        Sliding window standard deviation for `T_B`

    Q_subseq_isconstant : numpy.ndarray
        A boolean array that indicates whether the subsequence in `Q` is constant (True)

    T_subseq_isconstant : numpy.ndarray
        A boolean array that indicates whether a subsequence in `T` is constant (True)

    indices : numpy.ndarray
        The subsequence indices to compute `prescrump` for

    s : int
        The sampling interval that defaults to
        `int(np.ceil(m / config.STUMPY_EXCL_ZONE_DENOM))`

    excl_zone : int
        The half width for the exclusion zone
    """
    if T_B is None:
        T_B = T_A
        T_B_subseq_isconstant = T_A_subseq_isconstant
        excl_zone = int(np.ceil(m / config.STUMPY_EXCL_ZONE_DENOM))
    else:
        excl_zone = None

    T_A, μ_Q, σ_Q, Q_subseq_isconstant = core.preprocess(
        T_A, m, T_subseq_isconstant=T_A_subseq_isconstant
    )
    T_B, M_T, Σ_T, T_subseq_isconstant = core.preprocess(
        T_B, m, T_subseq_isconstant=T_B_subseq_isconstant
    )

    n_A = T_A.shape[0]
    l = n_A - m + 1

    if s is None:  # pragma: no cover
        if excl_zone is not None:  # self-join
            s = excl_zone
        else:  # AB-join
            s = int(np.ceil(m / config.STUMPY_EXCL_ZONE_DENOM))

    indices = np.random.permutation(range(0, l, s)).astype(np.int64)

    return (
        T_A,
        T_B,
        μ_Q,
        σ_Q,
        M_T,
        Σ_T,
        Q_subseq_isconstant,
        T_subseq_isconstant,
        indices,
        s,
        excl_zone,
    )


@njit(fastmath=config.STUMPY_FASTMATH_FLAGS)
def _compute_PI(
    T_A,
    T_B,
    m,
    μ_Q,
    σ_Q,
    M_T,
    Σ_T,
    Q_subseq_isconstant,
    T_subseq_isconstant,
    indices,
    start,
    stop,
    thread_idx,
    s,
    P_squared,
    I,
    excl_zone=None,
    k=1,
):
    """
    Compute (Numba JIT-compiled) and update the squared (top-k) matrix profile
    distance and matrix profile indces according to the preSCRIMP algorithm.

    Parameters
    ----------
    T_A : numpy.ndarray
        The time series or sequence for which to compute the matrix profile

    T_B : numpy.ndarray
        The time series or sequence that will be used to annotate T_A. For every
        subsequence in T_A, its nearest neighbor in T_B will be recorded.

    m : int
        Window size

    μ_Q : numpy.ndarray
        Sliding window mean for `T_A`

    σ_Q : numpy.ndarray
        Sliding window standard deviation for `T_A`

    M_T : numpy.ndarray
        Sliding window mean for `T_B`

    Σ_T : numpy.ndarray
        Sliding window standard deviation for `T_B`

    Q_subseq_isconstant : numpy.ndarray
        A boolean array that indicates whether a subsequence in `T_A` is constant (True)

    T_subseq_isconstant : numpy.ndarray
        A boolean array that indicates whether a subsequence in `T_B` is constant (True)

    indices : numpy.ndarray
        The subsequence indices to compute `prescrump` for

    start : int
        The (inclusive) start index for `indices`

    stop : int
        The (exclusive) stop index for `indices`

    thread_idx : int
        The thread index

    s : int
        The sampling interval that defaults to
        `int(np.ceil(m / config.STUMPY_EXCL_ZONE_DENOM))`

    P_squared : numpy.ndarray
        The squared (top-k) matrix profile

    I : numpy.ndarray
        The (top-k) matrix profile indices

    excl_zone : int
        The half width for the exclusion zone relative to the `i`.

    k : int, default 1
        The number of top `k` smallest distances used to construct the matrix profile.
        Note that this will increase the total computational time and memory usage
        when k > 1.

    Returns
    -------
    None

    Notes
    -----
    `DOI: 10.1109/ICDM.2018.00099 \
    <https://www.cs.ucr.edu/~eamonn/SCRIMP_ICDM_camera_ready_updated.pdf>`__

    See Algorithm 2
    """
    l = T_A.shape[0] - m + 1  # length of matrix profile
    w = T_B.shape[0] - m + 1  # length of distance profile
    squared_distance_profile = np.empty(w)
    QT = np.empty(w, dtype=np.float64)
    for i in indices[start:stop]:
        Q = T_A[i : i + m]
        QT[:] = core._sliding_dot_product(Q, T_B)
        squared_distance_profile[:] = core._calculate_squared_distance_profile(
            m,
            QT,
            μ_Q[i],
            σ_Q[i],
            M_T,
            Σ_T,
            Q_subseq_isconstant[i],
            T_subseq_isconstant,
        )

        if excl_zone is not None:
            core._apply_exclusion_zone(squared_distance_profile, i, excl_zone, np.inf)

        nn_i = np.argmin(squared_distance_profile)
        if squared_distance_profile[nn_i] == np.inf:
            continue

        if (
            squared_distance_profile[nn_i] < P_squared[thread_idx, i, -1]
            and nn_i not in I[thread_idx, i]
        ):
            idx = np.searchsorted(
                P_squared[thread_idx, i],
                squared_distance_profile[nn_i],
                side="right",
            )
            core._shift_insert_at_index(
                P_squared[thread_idx, i], idx, squared_distance_profile[nn_i]
            )
            core._shift_insert_at_index(I[thread_idx, i], idx, nn_i)

        j = nn_i
        QT_j = QT[j]
        QT_j_prime = QT_j
        # Update top-k for both subsequences `S[i+g] = T[i+g:i+g+m]`` and
        # `S[j+g] = T[j+g:j+g+m]` (i.e., the right neighbors of `T[i : i+m]` and
        # `T[j:j+m]`) by using the distance between `S[i+g]` and `S[j+g]`
        for g in range(1, min(s, l - i, w - j)):
            QT_j = (
                QT_j
                - T_B[j + g - 1] * T_A[i + g - 1]
                + T_B[j + g + m - 1] * T_A[i + g + m - 1]
            )
            D_squared = core._calculate_squared_distance(
                m,
                QT_j,
                M_T[j + g],
                Σ_T[j + g],
                μ_Q[i + g],
                σ_Q[i + g],
                T_subseq_isconstant[j + g],
                Q_subseq_isconstant[i + g],
            )
            if (
                D_squared < P_squared[thread_idx, i + g, -1]
                and (j + g) not in I[thread_idx, i + g]
            ):
                idx = np.searchsorted(
                    P_squared[thread_idx, i + g], D_squared, side="right"
                )
                core._shift_insert_at_index(
                    P_squared[thread_idx, i + g], idx, D_squared
                )
                core._shift_insert_at_index(I[thread_idx, i + g], idx, j + g)

            if (
                excl_zone is not None
                and D_squared < P_squared[thread_idx, j + g, -1]
                and (i + g) not in I[thread_idx, j + g]
            ):
                idx = np.searchsorted(
                    P_squared[thread_idx, j + g], D_squared, side="right"
                )
                core._shift_insert_at_index(
                    P_squared[thread_idx, j + g], idx, D_squared
                )
                core._shift_insert_at_index(I[thread_idx, j + g], idx, i + g)

        QT_j = QT_j_prime
        # Update top-k for both subsequences `S[i-g] = T[i-g:i-g+m]` and
        # `S[j-g] = T[j-g:j-g+m]` (i.e., the left neighbors of `T[i : i+m]` and
        # `T[j:j+m]`) by using the distance between `S[i-g]` and `S[j-g]`
        for g in range(1, min(s, i + 1, j + 1)):
            QT_j = QT_j - T_B[j - g + m] * T_A[i - g + m] + T_B[j - g] * T_A[i - g]
            D_squared = core._calculate_squared_distance(
                m,
                QT_j,
                M_T[j - g],
                Σ_T[j - g],
                μ_Q[i - g],
                σ_Q[i - g],
                T_subseq_isconstant[j - g],
                Q_subseq_isconstant[i - g],
            )
            if (
                D_squared < P_squared[thread_idx, i - g, -1]
                and (j - g) not in I[thread_idx, i - g]
            ):
                idx = np.searchsorted(
                    P_squared[thread_idx, i - g], D_squared, side="right"
                )
                core._shift_insert_at_index(
                    P_squared[thread_idx, i - g], idx, D_squared
                )
                core._shift_insert_at_index(I[thread_idx, i - g], idx, j - g)

            if (
                excl_zone is not None
                and D_squared < P_squared[thread_idx, j - g, -1]
                and (i - g) not in I[thread_idx, j - g]
            ):
                idx = np.searchsorted(
                    P_squared[thread_idx, j - g], D_squared, side="right"
                )
                core._shift_insert_at_index(
                    P_squared[thread_idx, j - g], idx, D_squared
                )
                core._shift_insert_at_index(I[thread_idx, j - g], idx, i - g)

        # In the case of a self-join, the calculated distance profile can also be
        # used to refine the top-k for all non-trivial subsequences
        if excl_zone is not None:
            # Note that the squared distance, `squared_distance_profile[j]`,
            # between subsequences `S_i = T[i : i + m]` and `S_j = T[j : j + m]`
            # can be used to update the top-k for BOTH subsequence `i` and
            # subsequence `j`. We update the latter here.

            indices = np.flatnonzero(
                squared_distance_profile < P_squared[thread_idx, :, -1]
            )
            for j in indices:
                if i not in I[thread_idx, j]:
                    idx = np.searchsorted(
                        P_squared[thread_idx, j],
                        squared_distance_profile[j],
                        side="right",
                    )
                    core._shift_insert_at_index(
                        P_squared[thread_idx, j], idx, squared_distance_profile[j]
                    )
                    core._shift_insert_at_index(I[thread_idx, j], idx, i)


@njit(
    # "(f8[:], f8[:], i8, f8[:], f8[:], f8[:], f8[:], f8[:], i8, i8, f8[:], f8[:],"
    # "i8[:], optional(i8))",
    parallel=True,
    fastmath=config.STUMPY_FASTMATH_FLAGS,
)
def _prescrump(
    T_A,
    T_B,
    m,
    μ_Q,
    σ_Q,
    M_T,
    Σ_T,
    Q_subseq_isconstant,
    T_subseq_isconstant,
    indices,
    s,
    excl_zone=None,
    k=1,
):
    """
    A Numba JIT-compiled implementation of the preSCRIMP algorithm.

    Parameters
    ----------
    T_A : numpy.ndarray
        The time series or sequence for which to compute the matrix profile

    T_B : numpy.ndarray
        The time series or sequence that will be used to annotate T_A. For every
        subsequence in T_A, its nearest neighbor in T_B will be recorded.

    m : int
        Window size

    μ_Q : numpy.ndarray
        Sliding window mean for `T_A`

    σ_Q : numpy.ndarray
        Sliding window standard deviation for `T_A`

    M_T : numpy.ndarray
        Sliding window mean for `T_B`

    Σ_T : numpy.ndarray
        Sliding window standard deviation for `T_B`

    Q_subseq_isconstant : numpy.ndarray
        A boolean array that indicates whether a subsequence in `T_A` is constant (True)

    T_subseq_isconstant : numpy.ndarray
        A boolean array that indicates whether a subsequence in `T_B` is constant (True)

    indices : numpy.ndarray
        The subsequence indices to compute `prescrump` for

    s : int
        The sampling interval that defaults to
        `int(np.ceil(m / config.STUMPY_EXCL_ZONE_DENOM))`

    excl_zone : int
        The half width for the exclusion zone relative to the `i`.

    k : int, default 1
        The number of top `k` smallest distances used to construct the matrix profile.
        Note that this will increase the total computational time and memory usage
        when k > 1.

    Returns
    -------
    out1 : numpy.ndarray
        The (top-k) matrix profile. When k=1 (default), the first (and only) column
        in this 2D array consists of the matrix profile. When k > 1, the output
        has exactly `k` columns consisting of the top-k matrix profile.

    out2 : numpy.ndarray
        The (top-k) matrix profile indices. When k=1 (default), the first (and only)
        column in this 2D array consists of the matrix profile indices. When k > 1,
        the output has exactly `k` columns consisting of the top-k matrix profile
        indices.

    Notes
    -----
    `DOI: 10.1109/ICDM.2018.00099 \
    <https://www.cs.ucr.edu/~eamonn/SCRIMP_ICDM_camera_ready_updated.pdf>`__

    See Algorithm 2
    """
    n_threads = numba.config.NUMBA_NUM_THREADS
    l = T_A.shape[0] - m + 1
    P_squared = np.full((n_threads, l, k), np.inf, dtype=np.float64)
    I = np.full((n_threads, l, k), -1, dtype=np.int64)

    idx_ranges = core._get_ranges(len(indices), n_threads, truncate=False)
    for thread_idx in prange(n_threads):
        _compute_PI(
            T_A,
            T_B,
            m,
            μ_Q,
            σ_Q,
            M_T,
            Σ_T,
            Q_subseq_isconstant,
            T_subseq_isconstant,
            indices,
            idx_ranges[thread_idx, 0],
            idx_ranges[thread_idx, 1],
            thread_idx,
            s,
            P_squared,
            I,
            excl_zone,
            k,
        )

    for thread_idx in range(1, n_threads):
        core._merge_topk_PI(P_squared[0], P_squared[thread_idx], I[0], I[thread_idx])

    return np.sqrt(P_squared[0]), I[0]


@core.non_normalized(prescraamp)
def prescrump(
    T_A,
    m,
    T_B=None,
    s=None,
    normalize=True,
    p=2.0,
    k=1,
    T_A_subseq_isconstant=None,
    T_B_subseq_isconstant=None,
):
    """
    A convenience wrapper around the Numba JIT-compiled parallelized
    `_prescrump` function which computes the approximate (top-k) matrix
    profile according to the preSCRIMP algorithm.

    Parameters
    ----------
    T_A : numpy.ndarray
        The time series or sequence for which to compute the matrix profile

    m : int
        Window size

    T_B : numpy.ndarray, default None
        The time series or sequence that will be used to annotate T_A. For every
        subsequence in T_A, its nearest neighbor in T_B will be recorded.

    s : int, default None
        The sampling interval that defaults to
        `int(np.ceil(m / config.STUMPY_EXCL_ZONE_DENOM))`

    normalize : bool, default True
        When set to `True`, this z-normalizes subsequences prior to computing distances.
        Otherwise, this function gets re-routed to its complementary non-normalized
        equivalent set in the `@core.non_normalized` function decorator.

    p : float, default 2.0
        The p-norm to apply for computing the Minkowski distance. Minkowski distance is
        typically used with `p` being 1 or 2, which correspond to the Manhattan distance
        and the Euclidean distance, respectively. This parameter is ignored when
        `normalize == True`.

    k : int, default 1
        The number of top `k` smallest distances used to construct the matrix profile.
        Note that this will increase the total computational time and memory usage
        when k > 1.

    T_A_subseq_isconstant : numpy.ndarray or function, default None
        A boolean array that indicates whether a subsequence in `T_A` is constant
        (True). Alternatively, a custom, user-defined function that returns a
        boolean array that indicates whether a subsequence in `T_A` is constant
        (True). The function must only take two arguments, `a`, a 1-D array,
        and `w`, the window size, while additional arguments may be specified
        by currying the user-defined function using `functools.partial`. Any
        subsequence with at least one np.nan/np.inf will automatically have its
        corresponding value set to False in this boolean array.

    T_B_subseq_isconstant : numpy.ndarray or function, default None
        A boolean array that indicates whether a subsequence in `T_B` is constant
        (True). Alternatively, a custom, user-defined function that returns a
        boolean array that indicates whether a subsequence in `T_B` is constant
        (True). The function must only take two arguments, `a`, a 1-D array,
        and `w`, the window size, while additional arguments may be specified
        by currying the user-defined function using `functools.partial`. Any
        subsequence with at least one np.nan/np.inf will automatically have its
        corresponding value set to False in this boolean array.

    Returns
    -------
    P : numpy.ndarray
        The (top-k) matrix profile. When k = 1 (default), this is a 1D array
        consisting of the matrix profile. When k > 1, the output is a 2D array that
        has exactly `k` columns consisting of the top-k matrix profile.

    I : numpy.ndarray
        The (top-k) matrix profile indices. When k = 1 (default), this is a 1D array
        consisting of the matrix profile indices. When k > 1, the output is a 2D
        array that has exactly `k` columns consisting of the top-k matrix profile
        indices.

    Notes
    -----
    `DOI: 10.1109/ICDM.2018.00099 \
    <https://www.cs.ucr.edu/~eamonn/SCRIMP_ICDM_camera_ready_updated.pdf>`__

    See Algorithm 2
    """
    (
        T_A,
        T_B,
        μ_Q,
        σ_Q,
        M_T,
        Σ_T,
        Q_subseq_isconstant,
        T_subseq_isconstant,
        indices,
        s,
        excl_zone,
    ) = _preprocess_prescrump(
        T_A,
        m,
        T_B=T_B,
        s=s,
        T_A_subseq_isconstant=T_A_subseq_isconstant,
        T_B_subseq_isconstant=T_B_subseq_isconstant,
    )

    P, I = _prescrump(
        T_A,
        T_B,
        m,
        μ_Q,
        σ_Q,
        M_T,
        Σ_T,
        Q_subseq_isconstant,
        T_subseq_isconstant,
        indices,
        s,
        excl_zone,
        k,
    )

    if k == 1:
        return P.flatten().astype(np.float64), I.flatten().astype(np.int64)
    else:
        return P, I


@core.non_normalized(
    scraamp,
    exclude=[
        "normalize",
        "pre_scrump",
        "pre_scraamp",
        "p",
        "T_A_subseq_isconstant",
        "T_B_subseq_isconstant",
    ],
    replace={"pre_scrump": "pre_scraamp"},
)
class scrump:
    """
    A class to ompute an approximate z-normalized matrix profile

    This is a convenience wrapper around the Numba JIT-compiled parallelized
    ``_stump`` function which computes the matrix profile according to SCRIMP.

    Parameters
    ----------
    T_A : numpy.ndarray
        The time series or sequence for which to compute the matrix profile.

    T_B : numpy.ndarray
        The time series or sequence that will be used to annotate ``T_A``. For every
        subsequence in ``T_A``, its nearest neighbor in ``T_B`` will be recorded.

    m : int
        Window size.

    ignore_trivial : bool
        Set to ``True`` if this is a self-join. Otherwise, for AB-join, set this to
        ``False``.

    percentage : float
        Approximate percentage completed. The value is between ``0.0`` and ``1.0``.

    pre_scrump : bool
        A flag for whether or not to perform the PreSCRIMP calculation prior to
        computing SCRIMP. If set to ``True``, this is equivalent to computing
        SCRIMP++ and may lead to faster convergence

    s : int
        The size of the PreSCRIMP fixed interval. If ``pre_scrump = True`` and
        ``s = None``, then ``s`` will automatically be set to
        ``s = int(np.ceil(m / config.STUMPY_EXCL_ZONE_DENOM))``, which is the size of
        the exclusion zone.

    normalize : bool, default True
        When set to ``True``, this z-normalizes subsequences prior to computing
        distances. Otherwise, this class gets re-routed to its complementary
        non-normalized equivalent set in the ``@core.non_normalized`` class decorator.

    p : float, default 2.0
        The p-norm to apply for computing the Minkowski distance. Minkowski distance is
        typically used with ``p`` being ``1`` or ``2``, which correspond to the
        Manhattan distance and the Euclidean distance, respectively. This parameter is
        ignored when ``normalize == True``.

    k : int, default 1
        The number of top ``k`` smallest distances used to construct the matrix profile.
        Note that this will increase the total computational time and memory usage
        when ``k > 1``.

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

    Attributes
    ----------
    P_ : numpy.ndarray
        The updated (top-k) matrix profile. When ``k = 1`` (default), this output is
        a 1D array consisting of the matrix profile. When ``k > 1``, the output
        is a 2D array that has exactly ``k`` columns consisting of the top-k matrix
        profile.

    I_ : numpy.ndarray
        The updated (top-k) matrix profile indices. When ``k = 1`` (default), this
        output is a 1D array consisting of the matrix profile indices. When ``k > 1``,
        the output is a 2D array that has exactly ``k`` columns consisting of the top-k
        matrix profile indices.

    left_I_ : numpy.ndarray
        The updated left (top-1) matrix profile indices.

    right_I_ : numpy.ndarray
        The updated right (top-1) matrix profile indices.


    Methods
    -------
    update()
        Update the matrix profile and the matrix profile indices by computing
        additional new distances (limited by ``percentage``) that make up the full
        distance matrix. It updates the (top-k) matrix profile, (top-1) left
        matrix profile, (top-1) right matrix profile, (top-k) matrix profile indices,
        (top-1) left matrix profile indices, and (top-1) right matrix profile indices.

    See Also
    --------
    stumpy.stump : Compute the z-normalized matrix profile
    stumpy.stumped : Compute the z-normalized matrix profile with a ``dask``/``ray``
        cluster
    stumpy.gpu_stump : Compute the z-normalized matrix profile with one or more GPU
        devices

    Notes
    -----
    `DOI: 10.1109/ICDM.2018.00099 \
    <https://www.cs.ucr.edu/~eamonn/SCRIMP_ICDM_camera_ready_updated.pdf>`__

    See Algorithm 1 and Algorithm 2

    Examples
    --------
    >>> import stumpy
    >>> import numpy as np
    >>> approx_mp = stumpy.scrump(
    ...     np.array([584., -11., 23., 79., 1001., 0., -19.]),
    ...     m=3)
    >>> approx_mp.update()
    >>> approx_mp.P_
    array([2.982409  , 3.28412702,        inf, 2.982409  , 3.28412702])
    >>> approx_mp.I_
    array([ 3,  4, -1,  0,  1])
    """

    def __init__(
        self,
        T_A,
        m,
        T_B=None,
        ignore_trivial=True,
        percentage=0.01,
        pre_scrump=False,
        s=None,
        normalize=True,
        p=2.0,
        k=1,
        T_A_subseq_isconstant=None,
        T_B_subseq_isconstant=None,
    ):
        """
        Initialize the `scrump` object

        Parameters
        ----------
        T_A : numpy.ndarray
            The time series or sequence for which to compute the matrix profile

        m : int
            Window size

        T_B : numpy.ndarray, default None
            The time series or sequence that will be used to annotate T_A. For every
            subsequence in T_A, its nearest neighbor in T_B will be recorded.

        ignore_trivial : bool, default True
            Set to `True` if this is a self-join. Otherwise, for AB-join, set this to
            `False`. Default is `True`.

        percentage : float, default 0.01
            Approximate percentage completed. The value is between 0.0 and 1.0.

        pre_scrump : bool, default False
            A flag for whether or not to perform the PreSCRIMP calculation prior to
            computing SCRIMP. If set to `True`, this is equivalent to computing
            SCRIMP++

        s : int, default None
            The size of the PreSCRIMP fixed interval. If `pre_scrump=True` and `s=None`,
            then `s` will automatically be set to
            `s=int(np.ceil(m / config.STUMPY_EXCL_ZONE_DENOM))`, the size of the
            exclusion zone.

        normalize : bool, default True
            When set to `True`, this z-normalizes subsequences prior to computing
            distances. Otherwise, this class gets re-routed to its complementary
            non-normalized equivalent set in the `@core.non_normalized` class decorator.

        p : float, default 2.0
            The p-norm to apply for computing the Minkowski distance. Minkowski distance
            is typically used with `p` being 1 or 2, which correspond to the Manhattan
            distance and the Euclidean distance, respectively.This parameter is ignored
            when `normalize == True`.

        k : int, default 1
            The number of top `k` smallest distances used to construct the matrix
            profile. Note that this will increase the total computational time and
            memory usage when k > 1.

        T_A_subseq_isconstant : numpy.ndarray or function, default None
            A boolean array that indicates whether a subsequence in `T_A` is constant
            (True). Alternatively, a custom, user-defined function that returns a
            boolean array that indicates whether a subsequence in `T_A` is constant
            (True). The function must only take two arguments, `a`, a 1-D array,
            and `w`, the window size, while additional arguments may be specified
            by currying the user-defined function using `functools.partial`. Any
            subsequence with at least one np.nan/np.inf will automatically have its
            corresponding value set to False in this boolean array.

        T_B_subseq_isconstant : numpy.ndarray or function, default None
            A boolean array that indicates whether a subsequence in `T_B` is constant
            (True). Alternatively, a custom, user-defined function that returns a
            boolean array that indicates whether a subsequence in `T_B` is constant
            (True). The function must only take two arguments, `a`, a 1-D array,
            and `w`, the window size, while additional arguments may be specified
            by currying the user-defined function using `functools.partial`. Any
            subsequence with at least one np.nan/np.inf will automatically have its
            corresponding value set to False in this boolean array.
        """
        self._ignore_trivial = ignore_trivial

        if T_B is None:
            T_B = T_A
            self._ignore_trivial = True
            T_B_subseq_isconstant = T_A_subseq_isconstant

        self._m = m
        (
            self._T_A,
            self._μ_Q,
            self._σ_Q_inverse,
            self._μ_Q_m_1,
            self._Q_subseq_isfinite,
            self._Q_subseq_isconstant,
        ) = core.preprocess_diagonal(
            T_A, self._m, T_subseq_isconstant=T_A_subseq_isconstant
        )

        (
            self._T_B,
            self._M_T,
            self._Σ_T_inverse,
            self._M_T_m_1,
            self._T_subseq_isfinite,
            self._T_subseq_isconstant,
        ) = core.preprocess_diagonal(
            T_B, self._m, T_subseq_isconstant=T_B_subseq_isconstant
        )

        if self._T_A.ndim != 1:  # pragma: no cover
            raise ValueError(
                f"T_A is {self._T_A.ndim}-dimensional and must be 1-dimensional. "
                "For multidimensional STUMP use `stumpy.mstump` or `stumpy.mstumped`"
            )

        if self._T_B.ndim != 1:  # pragma: no cover
            raise ValueError(
                f"T_B is {self._T_B.ndim}-dimensional and must be 1-dimensional. "
                "For multidimensional STUMP use `stumpy.mstump` or `stumpy.mstumped`"
            )

        self._ignore_trivial = core.check_ignore_trivial(
            self._T_A, self._T_B, self._ignore_trivial
        )
        if self._ignore_trivial:
            core.check_window_size(
                m, max_size=min(T_A.shape[0], T_B.shape[0]), n=T_A.shape[0]
            )
        else:
            core.check_window_size(m, max_size=min(T_A.shape[0], T_B.shape[0]))

        self._n_A = self._T_A.shape[0]
        self._n_B = self._T_B.shape[0]
        self._l = self._n_A - self._m + 1
        self._k = k

        self._P = np.full((self._l, self._k), np.inf, dtype=np.float64)
        self._PL = np.full(self._l, np.inf, dtype=np.float64)
        self._PR = np.full(self._l, np.inf, dtype=np.float64)

        self._I = np.full((self._l, self._k), -1, dtype=np.int64)
        self._IL = np.full(self._l, -1, dtype=np.int64)
        self._IR = np.full(self._l, -1, dtype=np.int64)

        self._excl_zone = int(np.ceil(self._m / config.STUMPY_EXCL_ZONE_DENOM))
        if s is None:
            s = self._excl_zone

        if pre_scrump:
            if self._ignore_trivial:
                (
                    T_A,
                    T_B,
                    μ_Q,
                    σ_Q,
                    M_T,
                    Σ_T,
                    Q_subseq_isconstant,
                    T_subseq_isconstant,
                    indices,
                    s,
                    excl_zone,
                ) = _preprocess_prescrump(
                    T_A, m, s=s, T_A_subseq_isconstant=T_A_subseq_isconstant
                )
            else:
                (
                    T_A,
                    T_B,
                    μ_Q,
                    σ_Q,
                    M_T,
                    Σ_T,
                    Q_subseq_isconstant,
                    T_subseq_isconstant,
                    indices,
                    s,
                    excl_zone,
                ) = _preprocess_prescrump(
                    T_A,
                    m,
                    T_B=T_B,
                    s=s,
                    T_A_subseq_isconstant=T_A_subseq_isconstant,
                    T_B_subseq_isconstant=T_B_subseq_isconstant,
                )

            P, I = _prescrump(
                T_A,
                T_B,
                m,
                μ_Q,
                σ_Q,
                M_T,
                Σ_T,
                Q_subseq_isconstant,
                T_subseq_isconstant,
                indices,
                s,
                excl_zone,
                k,
            )
            core._merge_topk_PI(self._P, P, self._I, I)

        if self._ignore_trivial:
            self._diags = np.random.permutation(
                range(self._excl_zone + 1, self._n_A - self._m + 1)
            ).astype(np.int64)
            if self._diags.shape[0] == 0:  # pragma: no cover
                max_m = core.get_max_window_size(self._T_A.shape[0])
                raise ValueError(
                    f"The window size, `m = {self._m}`, is too long for a self join. "
                    f"Please try a value of `m <= {max_m}`"
                )
        else:
            self._diags = np.random.permutation(
                range(-(self._n_A - self._m + 1) + 1, self._n_B - self._m + 1)
            ).astype(np.int64)

        self._n_threads = numba.config.NUMBA_NUM_THREADS
        self._percentage = np.clip(percentage, 0.0, 1.0)
        self._n_chunks = int(np.ceil(1.0 / percentage))
        self._ndist_counts = core._count_diagonal_ndist(
            self._diags, self._m, self._n_A, self._n_B
        )
        self._chunk_diags_ranges = core._get_array_ranges(
            self._ndist_counts, self._n_chunks, True
        )
        self._n_chunks = self._chunk_diags_ranges.shape[0]
        self._chunk_idx = 0

    def update(self):
        """
        Update the (top-k) matrix profile and the (top-k) matrix profile indices by
        computing additional new distances (limited by `percentage`) that make up
        the full distance matrix.

        Parameters
        ----------
        None

        Returns
        -------
        None
        """
        if self._chunk_idx < self._n_chunks:
            start_idx, stop_idx = self._chunk_diags_ranges[self._chunk_idx]

            P, PL, PR, I, IL, IR = _stump(
                self._T_A,
                self._T_B,
                self._m,
                self._μ_Q,
                self._M_T,
                self._σ_Q_inverse,
                self._Σ_T_inverse,
                self._μ_Q_m_1,
                self._M_T_m_1,
                self._Q_subseq_isfinite,
                self._T_subseq_isfinite,
                self._Q_subseq_isconstant,
                self._T_subseq_isconstant,
                self._diags[start_idx:stop_idx],
                self._ignore_trivial,
                self._k,
            )

            # Update (top-k) matrix profile and indices
            core._merge_topk_PI(self._P, P, self._I, I)

            # update left matrix profile and indices
            mask = PL < self._PL
            self._PL[mask] = PL[mask]
            self._IL[mask] = IL[mask]

            # update right matrix profile and indices
            mask = PR < self._PR
            self._PR[mask] = PR[mask]
            self._IR[mask] = IR[mask]

            self._chunk_idx += 1

    @property
    def P_(self):
        """
        Get the updated (top-k) matrix profile. When `k=1` (default), this output
        is a 1D array consisting of the updated matrix profile. When `k > 1`, the
        output is a 2D array that has exactly `k` columns consisting of the updated
        top-k matrix profile.

        Parameters
        ----------
        None

        Returns
        -------
        None
        """
        if self._k == 1:
            return self._P.flatten().astype(np.float64)
        else:
            return self._P.astype(np.float64)

    @property
    def I_(self):
        """
        Get the updated (top-k) matrix profile indices. When `k=1` (default), this
        output is a 1D array consisting of the updated matrix profile indices. When
        `k > 1`, the output is a 2D array that has exactly `k` columns consisting
        of the updated top-k matrix profile indices.

        Parameters
        ----------
        None

        Returns
        -------
        None
        """
        if self._k == 1:
            return self._I.flatten().astype(np.int64)
        else:
            return self._I.astype(np.int64)

    @property
    def left_I_(self):
        """
        Get the updated left (top-1) matrix profile indices

        Parameters
        ----------
        None

        Returns
        -------
        None
        """
        return self._IL.astype(np.int64)

    @property
    def right_I_(self):
        """
        Get the updated right (top-1) matrix profile indices

        Parameters
        ----------
        None

        Returns
        -------
        None
        """
        return self._IR.astype(np.int64)
