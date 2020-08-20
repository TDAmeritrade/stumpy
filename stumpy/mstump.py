# STUMPY
# Copyright 2019 TD Ameritrade. Released under the terms of the 3-Clause BSD license.
# STUMPY is a trademark of TD Ameritrade IP Company, Inc. All rights reserved.

import logging

import numpy as np
from numba import njit, prange

from . import core

logger = logging.getLogger(__name__)


def _multi_mass(Q, T, m, M_T, Σ_T, μ_Q, σ_Q):
    """
    A multi-dimensional wrapper around "Mueen's Algorithm for Similarity Search"
    (MASS) to compute multi-dimensional distance profile.

    Parameters
    ----------
    Q : ndarray
        Query array or subsequence

    T : ndarray
        Time series array or sequence

    m : int
        Window size

    M_T : ndarray
        Sliding mean for `T_A`

    Σ_T : ndarray
        Sliding standard deviation for `T_A`

    μ_Q : ndarray
        Mean value of `Q`

    σ_Q : ndarray
        Standard deviation of `Q`

    include : ndarray
        A list of (zero-based) indices corresponding to the dimensions in `T` that
        must be included in the constrained multidimensional motif search.
        For more information, see Section IV D in:

        `DOI: 10.1109/ICDM.2017.66 \
        <https://www.cs.ucr.edu/~eamonn/Motif_Discovery_ICDM.pdf>`__

    discords : bool
        When set to `True`, this reverses the distance profile to favor discords rather
        than motifs. Note that indices in `include` are still maintained and respected.

    Returns
    -------
    D : ndarray
        Multi-dimensional distance profile
    """
    d, n = T.shape
    k = n - m + 1

    D = np.empty((d, k), dtype="float64")

    for i in range(d):
        if np.isinf(μ_Q[i]):
            D[i, :] = np.inf
        else:
            D[i, :] = core.mass(Q[i], T[i], M_T[i], Σ_T[i])

    return D


def _apply_include(
    D,
    include,
    restricted_indices=None,
    unrestricted_indices=None,
    mask=None,
    tmp_swap=None,
):
    """
    Apply a transformation to the multi-dimensional distance profile so that specific
    dimensions are always included. Essentially, it is swapping rows within the distance
    profile.

    Parameters
    ----------
    D : ndarray
        The multi-dimensional distance profile

    include : ndarray
        A list of (zero-based) indices corresponding to the dimensions in `T` that
        must be included in the constrained multidimensional motif search.
        For more information, see Section IV D in:

        `DOI: 10.1109/ICDM.2017.66 \
        <https://www.cs.ucr.edu/~eamonn/Motif_Discovery_ICDM.pdf>`__

    restricted_indices : ndarray
        A list of indices specified in `include` that reside in the first
        `include.shape[0]` rows

    unrestricted_indices : ndarray
        A list of indices specified in `include` that do not reside in the first
        `include.shape[0]` rows

    mask : ndarray
        A boolean mask to select for unrestricted indices

    tmp_swap : ndarray
        A reusable array to aid in array element swapping
    """
    if restricted_indices is None:
        restricted_indices = include[include < include.shape[0]]

    if unrestricted_indices is None:
        unrestricted_indices = include[include >= include.shape[0]]

    if mask is None:
        mask = np.ones(include.shape[0], bool)
        mask[restricted_indices] = False

    if tmp_swap is None:
        tmp_swap = D[: include.shape[0]].copy()
    else:
        tmp_swap[:] = D[: include.shape[0]]

    D[: include.shape[0]] = D[include]
    D[unrestricted_indices] = tmp_swap[mask]


def _get_first_mstump_profile(
    start, T_A, T_B, m, excl_zone, M_T, Σ_T, μ_Q, σ_Q, include=None, discords=False
):
    """
    Multi-dimensional wrapper to compute the multi-dimensional matrix profile
    and multi-dimensional matrix profile index for a given window within the
    times series or sequence that is denote by the `start` index.
    Essentially, this is a convenience wrapper around `_multi_mass`

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

    μ_Q : ndarray
        Sliding mean for `T_B`

    σ_Q : ndarray
        Sliding standard deviation for `T_B`

    include : ndarray
        A list of (zero-based) indices corresponding to the dimensions in `T` that
        must be included in the constrained multidimensional motif search.
        For more information, see Section IV D in:

        `DOI: 10.1109/ICDM.2017.66 \
        <https://www.cs.ucr.edu/~eamonn/Motif_Discovery_ICDM.pdf>`__

    discords : bool
        When set to `True`, this reverses the distance profile to favor discords rather
        than motifs. Note that indices in `include` are still maintained and respected.

    Returns
    -------
    P : ndarray
        Multi-dimensional matrix profile for the window with index equal to
        `start`

    I : ndarray
        Multi-dimensional matrix profile index for the window with index
        equal to `start`
    """
    d, n = T_A.shape
    k = n - m + 1
    start_row_idx = 0
    D = _multi_mass(
        T_B[:, start : start + m], T_A, m, M_T, Σ_T, μ_Q[:, start], σ_Q[:, start]
    )

    if include is not None:
        _apply_include(D, include)
        start_row_idx = include.shape[0]

    if discords:
        D[start_row_idx:][::-1].sort(axis=0)
    else:
        D[start_row_idx:].sort(axis=0)

    D_prime = np.zeros(k)
    for i in range(d):
        D_prime[:] = D_prime + D[i]
        D[i, :] = D_prime / (i + 1)

    core.apply_exclusion_zone(D, start, excl_zone)

    P = np.full(d, np.inf, dtype="float64")
    I = np.full(d, -1, dtype="int64")

    for i in range(d):
        min_index = np.argmin(D[i])
        I[i] = min_index
        P[i] = D[i, min_index]
        if np.isinf(P[i]):  # pragma nocover
            I[i] = -1

    return P, I


def _get_multi_QT(start, T, m):
    """
    Multi-dimensional wrapper to compute the sliding dot product between
    the query, `T[:, start:start+m])` and the time series, `T`.
    Additionally, compute QT for the first window.

    Parameters
    ----------
    start : int
        The window index for T_B from which to calculate the QT dot product

    T : ndarray
        The time series or sequence for which to compute the dot product

    m : int
        Window size

    Returns
    -------
    QT : ndarray
        Given `start`, return the corresponding multi-dimensional QT

    QT_first : ndarray
        Multi-dimensional QT for the first window
    """
    d = T.shape[0]
    k = T.shape[1] - m + 1

    QT = np.empty((d, k), dtype="float64")
    QT_first = np.empty((d, k), dtype="float64")

    for i in range(d):
        QT[i] = core.sliding_dot_product(T[i, start : start + m], T[i])
        QT_first[i] = core.sliding_dot_product(T[i, :m], T[i])

    return QT, QT_first


@njit(parallel=True, fastmath=True)
def _compute_multi_D(
    d, k, idx, D, T, m, excl_zone, M_T, Σ_T, QT_even, QT_odd, QT_first, μ_Q, σ_Q
):
    """
    A Numba JIT-compiled version of mSTOMP for parallel computation of the
    multi-dimensional distance profile

    Parameters
    ----------
    d : int
        The total number of dimensions in `T`

    k : int
        The total number of sliding windows to iterate over

    idx : int
        The row index in `T`

    D : ndarray
        The output distance profile

    T : ndarray
        The time series or sequence for which to compute the matrix profile

    m : int
        Window size

    excl_zone : int
        The half width for the exclusion zone relative to the current
        sliding window

    M_T : ndarray
        Sliding mean of time series, `T`

    Σ_T : ndarray
        Sliding standard deviation of time series, `T`

    QT_even : ndarray
        Dot product between some query sequence,`Q`, and time series, `T`

    QT_odd : ndarray
        Dot product between some query sequence,`Q`, and time series, `T`

    QT_first : ndarray
        QT for the first window relative to the current sliding window

    μ_Q : ndarray
        Mean of the query sequence, `Q`, relative to the current sliding window

    σ_Q : ndarray
        Standard deviation of the query sequence, `Q`, relative to the current
        sliding window

    Notes
    -----
    `DOI: 10.1109/ICDM.2017.66 \
    <https://www.cs.ucr.edu/~eamonn/Motif_Discovery_ICDM.pdf>`__

    See mSTAMP Algorithm
    """
    for i in range(d):
        # Numba's prange requires incrementing a range by 1 so replace
        # `for j in range(k-1,0,-1)` with its incrementing compliment
        for rev_j in prange(1, k):
            j = k - rev_j
            # GPU Stomp Parallel Implementation with Numba
            # DOI: 10.1109/ICDM.2016.0085
            # See Figure 5
            if idx % 2 == 0:
                # Even
                QT_even[i, j] = (
                    QT_odd[i, j - 1]
                    - T[i, idx - 1] * T[i, j - 1]
                    + T[i, idx + m - 1] * T[i, j + m - 1]
                )
            else:
                # Odd
                QT_odd[i, j] = (
                    QT_even[i, j - 1]
                    - T[i, idx - 1] * T[i, j - 1]
                    + T[i, idx + m - 1] * T[i, j + m - 1]
                )

        if idx % 2 == 0:
            QT_even[i, 0] = QT_first[i, idx]
            D[i] = core._calculate_squared_distance_profile(
                m, QT_even[i], μ_Q[i, idx], σ_Q[i, idx], M_T[i], Σ_T[i]
            )
        else:
            QT_odd[i, 0] = QT_first[i, idx]
            D[i] = core._calculate_squared_distance_profile(
                m, QT_odd[i], μ_Q[i, idx], σ_Q[i, idx], M_T[i], Σ_T[i]
            )

    core.apply_exclusion_zone(D, idx, excl_zone)


@njit(parallel=True, fastmath=True)
def _compute_PI(d, idx, D, D_prime, range_start, P, I):
    """
    A Numba JIT-compiled version of mSTOMP for updating the matrix profile and matrix
    profile indices

    Parameters
    ----------
    d : int
        The total number of dimensions in `T`

    idx : int
        The row index in `T`

    D : ndarray
        The distance profile

    D_prime : ndarray
        A reusable array for storing the column-wise cumulative sum of `D`

    range_start : int
        The starting index value along `T` for which to start the matrix
        profile calculation

    P : ndarray
        The matrix profile

    I : ndarray
        The matrix profile indices
    """
    D_prime[:] = 0.0
    for i in range(d):
        D_prime = D_prime + np.sqrt(D[i])

        min_index = np.argmin(D_prime)
        pos = idx - range_start
        I[i, pos] = min_index
        P[i, pos] = D_prime[min_index] / (i + 1)
        if np.isinf(P[i, pos]):  # pragma nocover
            I[i, pos] = -1


def _mstump(
    T,
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
    range_start=1,
    include=None,
    discords=False,
):
    """
    A Numba JIT-compiled version of mSTOMP, a variant of mSTAMP, for parallel
    computation of the multi-dimensional matrix profile and multi-dimensional
    matrix profile indices. Note that only self-joins are supported.

    Parameters
    ----------
    T: ndarray
        The time series or sequence for which to compute the multi-dimensional
        matrix profile

    m : int
        Window size

    range_stop : int
        The index value along T for which to stop the matrix profile
        calculation. This parameter is here for consistency with the
        distributed `mstumped` algorithm.

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

    range_start : int
        The starting index value along T_B for which to start the matrix
        profile calculation. Default is 1.

    include : ndarray
        A list of (zero-based) indices corresponding to the dimensions in `T` that
        must be included in the constrained multidimensional motif search.
        For more information, see Section IV D in:

        `DOI: 10.1109/ICDM.2017.66 \
        <https://www.cs.ucr.edu/~eamonn/Motif_Discovery_ICDM.pdf>`__

    discords : bool
        When set to `True`, this reverses the distance profile to favor discords rather
        than motifs. Note that indices in `include` are still maintained and respected.

    Returns
    -------
    P : ndarray
        The multi-dimensional matrix profile. Each row of the array corresponds
        to each matrix profile for a given dimension (i.e., the first row is the
        1-D matrix profile and the second row is the 2-D matrix profile).

    I : ndarray
        The multi-dimensional matrix profile index where each row of the array
        corresponds to each matrix profile index for a given dimension.

    Notes
    -----
    `DOI: 10.1109/ICDM.2017.66 \
    <https://www.cs.ucr.edu/~eamonn/Motif_Discovery_ICDM.pdf>`__

    See mSTAMP Algorithm
    """
    QT_odd = QT.copy()
    QT_even = QT.copy()
    d = T.shape[0]

    P = np.empty((d, range_stop - range_start), dtype=float)
    I = np.empty((d, range_stop - range_start), dtype=int)
    D = np.empty((d, k), dtype=float)
    D_prime = np.empty(k, dtype=float)
    start_row_idx = 0

    if include is not None:
        tmp_swap = np.empty((include.shape[0], k))
        restricted_indices = include[include < include.shape[0]]
        unrestricted_indices = include[include >= include.shape[0]]
        mask = np.ones(include.shape[0], bool)
        mask[restricted_indices] = False

    for idx in range(range_start, range_stop):
        _compute_multi_D(
            d,
            k,
            idx,
            D,
            T,
            m,
            excl_zone,
            M_T,
            Σ_T,
            QT_even,
            QT_odd,
            QT_first,
            μ_Q,
            σ_Q,
        )

        if include is not None:
            _apply_include(
                D, include, restricted_indices, unrestricted_indices, mask, tmp_swap,
            )
            start_row_idx = include.shape[0]

        if discords:
            D[start_row_idx:][::-1].sort(axis=0)
        else:
            D[start_row_idx:].sort(axis=0)

        _compute_PI(d, idx, D, D_prime, range_start, P, I)

    return P, I


def mstump(T, m, include=None, discords=False):
    """
    Compute the multi-dimensional z-normalized matrix profile

    This is a convenience wrapper around the Numba JIT-compiled parallelized
    `_mstump` function which computes the multi-dimensional matrix profile and
    multi-dimensional matrix profile index according to mSTOMP, a variant of
    mSTAMP. Note that only self-joins are supported.

    Parameters
    ----------
    T : ndarray
        The time series or sequence for which to compute the multi-dimensional
        matrix profile. Each row in `T` represents data from a different
        dimension while each column in `T` represents data from the same
        dimension.

    m : int
        Window size

    include : list, ndarray
        A list of (zero-based) indices corresponding to the dimensions in `T` that
        must be included in the constrained multidimensional motif search.
        For more information, see Section IV D in:

        `DOI: 10.1109/ICDM.2017.66 \
        <https://www.cs.ucr.edu/~eamonn/Motif_Discovery_ICDM.pdf>`__

    discords : bool
        When set to `True`, this reverses the distance matrix which results in a
        multi-dimensional matrix profile that favors larger matrix profile values
        (i.e., discords) rather than smaller values (i.e., motifs). Note that indices
        in `include` are still maintained and respected.

    Returns
    -------
    P : ndarray
        The multi-dimensional matrix profile. Each column of the array corresponds
        to each matrix profile for a given dimension (i.e., the first column is
        the 1-D matrix profile and the second column is the 2-D matrix profile).

    I : ndarray
        The multi-dimensional matrix profile index where each column of the array
        corresponds to each matrix profile index for a given dimension.

    Notes
    -----
    `DOI: 10.1109/ICDM.2017.66 \
    <https://www.cs.ucr.edu/~eamonn/Motif_Discovery_ICDM.pdf>`__

    See mSTAMP Algorithm
    """
    T_A = core.transpose_dataframe(T)
    T_B = T_A

    T_A, M_T, Σ_T = core.preprocess(T_A, m)
    T_B, μ_Q, σ_Q = core.preprocess(T_B, m)

    if T_A.ndim <= 1:  # pragma: no cover
        err = f"T is {T_A.ndim}-dimensional and must be at least 1-dimensional"
        raise ValueError(f"{err}")

    core.check_dtype(T_A)
    core.check_dtype(T_B)

    core.check_window_size(m)

    if include is not None:
        include = np.asarray(include)
        _, idx = np.unique(include, return_index=True)
        if include.shape[0] != idx.shape[0]:  # pragma: no cover
            logger.warning("Removed repeating indices in `include`")
            include = include[np.sort(idx)]

    d, n = T_B.shape
    k = n - m + 1
    excl_zone = int(np.ceil(m / 4))  # See Definition 3 and Figure 3

    P = np.empty((d, k), dtype="float64")
    I = np.empty((d, k), dtype="int64")

    start = 0
    stop = k

    P[:, start], I[:, start] = _get_first_mstump_profile(
        start, T_A, T_B, m, excl_zone, M_T, Σ_T, μ_Q, σ_Q, include, discords
    )

    QT, QT_first = _get_multi_QT(start, T_A, m)

    P[:, start + 1 : stop], I[:, start + 1 : stop] = _mstump(
        T_A,
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
        start + 1,
        include,
        discords,
    )

    return P.T, I.T
