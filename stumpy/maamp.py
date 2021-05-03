# STUMPY
# Copyright 2019 TD Ameritrade. Released under the terms of the 3-Clause BSD license.
# STUMPY is a trademark of TD Ameritrade IP Company, Inc. All rights reserved.

import logging

import numpy as np
from numba import njit, prange

from . import core, config, mstump

logger = logging.getLogger(__name__)


def _multi_mass_absolute(Q, T, m, Q_subseq_isfinite, T_subseq_isfinite):
    """
    A multi-dimensional wrapper around "Mueen's Algorithm for Similarity Search"
    (MASS) absolute to compute multi-dimensional non-normalized (i.e., without
    z-normalization distance profile.

    Parameters
    ----------
    Q : ndarray
        Query array or subsequence

    T : ndarray
        Time series array or sequence

    m : int
        Window size

    Q_subseq_isfinite : ndarray
        A boolean array that indicates whether a subsequence in `Q` contains a
        `np.nan`/`np.inf` value (False)

    T_subseq_isfinite : ndarray
        A boolean array that indicates whether a subsequence in `T` contains a
        `np.nan`/`np.inf` value (False)

    Returns
    -------
    D : ndarray
        Multi-dimensional non-normalized (i.e., without z-normalization) distance
        profile
    """
    d, n = T.shape
    k = n - m + 1

    D = np.empty((d, k), dtype="float64")

    for i in range(d):
        if np.any(~Q_subseq_isfinite[i]):
            D[i, :] = np.inf
        else:
            D[i, :] = core.mass_absolute(Q[i], T[i])

        D[i][~(T_subseq_isfinite[i])] = np.inf

    return D


def maamp_subspace(T, m, subseq_idx, nn_idx, k, include=None, discords=False):
    """
    Compute the k-dimensional matrix profile subspace for a given subsequence index and
    its nearest neighbor index

    Parameters
    ----------
    T : ndarray
        The time series or sequence for which the multi-dimensional matrix profile,
        multi-dimensional matrix profile indices were computed

    m : int
        Window size

    subseq_idx : int
        The subsequence index in T

    nn_idx : int
        The nearest neighbor index in T

    k : int
        The subset number of dimensions out of `D = T.shape[0]`-dimensions to return
        the subspace for

    include : ndarray, default None
        A list of (zero-based) indices corresponding to the dimensions in `T` that
        must be included in the constrained multidimensional motif search.
        For more information, see Section IV D in:

        `DOI: 10.1109/ICDM.2017.66 \
        <https://www.cs.ucr.edu/~eamonn/Motif_Discovery_ICDM.pdf>`__

    discords : bool, default False
        When set to `True`, this reverses the distance profile to favor discords rather
        than motifs. Note that indices in `include` are still maintained and respected.

    Returns
    -------
        S : ndarray
        An array of that contains the `k`th-dimensional subspace for the subsequence
        with index equal to `motif_idx`
    """
    T, _ = core.preprocess_non_normalized(T, m)
    subseqs = T[:, subseq_idx : subseq_idx + m]
    neighbors = T[:, nn_idx : nn_idx + m]

    D = np.linalg.norm(subseqs - neighbors, axis=1)

    S = mstump._subspace(D, k, include=include, discords=discords)

    return S


def _query_maamp_profile(
    query_idx, T_A, T_B, m, excl_zone, T_B_subseq_isfinite, include=None, discords=False
):
    """
    Multi-dimensional wrapper to compute the multi-dimensional non-normalized (i.e.,
    without z-normalization) matrix profile and the multi-dimensional matrix profile
    index for a given query window within the times series or sequence that is denoted
    by the `query_idx` index. Essentially, this is a convenience wrapper around
    `_multi_mass_absolute`.

    Parameters
    ----------
    query_idx : int
        The window index to calculate the first multi-dimensional matrix profile and
        multi-dimensional matrix profile indices

    T_A : ndarray
        The time series or sequence for which the multi-dimensional matrix profile and
        multi-dimensional matrix profile indices

    T_B : ndarray
        The time series or sequence that contains your query subsequences

    m : int
        Window size

    excl_zone : int
        The half width for the exclusion zone relative to the `query_idx`.

    T_B_subseq_isfinite : ndarray
        A boolean array that indicates whether a subsequence in `T_B` contains a
        `np.nan`/`np.inf` value (False)

    include : ndarray, default None
        A list of (zero-based) indices corresponding to the dimensions in `T` that
        must be included in the constrained multidimensional motif search.
        For more information, see Section IV D in:

        `DOI: 10.1109/ICDM.2017.66 \
        <https://www.cs.ucr.edu/~eamonn/Motif_Discovery_ICDM.pdf>`__

    discords : bool, default False
        When set to `True`, this reverses the distance profile to favor discords rather
        than motifs. Note that indices in `include` are still maintained and respected.

    Returns
    -------
    P : ndarray
        Multi-dimensional matrix profile for the window with index equal to
        `query_idx`

    I : ndarray
        Multi-dimensional matrix profile indices for the window with index
        equal to `query_idx`
    """
    d, n = T_A.shape
    k = n - m + 1
    start_row_idx = 0
    D = _multi_mass_absolute(
        T_B[:, query_idx : query_idx + m],
        T_A,
        m,
        T_B_subseq_isfinite[:, query_idx],
        T_B_subseq_isfinite,
    )

    if include is not None:
        mstump._apply_include(D, include)
        start_row_idx = include.shape[0]

    if discords:
        D[start_row_idx:][::-1].sort(axis=0, kind="mergesort")
    else:
        D[start_row_idx:].sort(axis=0, kind="mergesort")

    D_prime = np.zeros(k)
    for i in range(d):
        D_prime[:] = D_prime + D[i]
        D[i, :] = D_prime / (i + 1)

    core.apply_exclusion_zone(D, query_idx, excl_zone)

    P = np.full(d, np.inf, dtype="float64")
    I = np.full(d, -1, dtype="int64")

    for i in range(d):
        min_index = np.argmin(D[i])
        I[i] = min_index
        P[i] = D[i, min_index]
        if np.isinf(P[i]):  # pragma nocover
            I[i] = -1

    return P, I


def _get_first_maamp_profile(
    start, T_A, T_B, m, excl_zone, T_B_subseq_isfinite, include=None, discords=False
):
    """
    Multi-dimensional wrapper to compute the non-normalized (i.e., without
    z-normalization multi-dimensional matrix profile and multi-dimensional matrix
    profile index for a given window within the times series or sequence that is denoted
    by the `start` index. Essentially, this is a convenience wrapper around
    `_multi_mass_absolute`. This is a convenience wrapper for the `_query_maamp_profile`
    function but does not return the multi-dimensional matrix profile subspace.

    Parameters
    ----------
    start : int
        The window index to calculate the first multi-dimensional matrix profile,
        multi-dimensional matrix profile indices, and multi-dimensional subspace.

    T_A : ndarray
        The time series or sequence for which the multi-dimensional matrix profile,
        multi-dimensional matrix profile indices, and multi-dimensional subspace will be
        returned

    T_B : ndarray
        The time series or sequence that contains your query subsequences

    m : int
        Window size

    excl_zone : int
        The half width for the exclusion zone relative to the `start`.

    T_B_subseq_isfinite : ndarray
        A boolean array that indicates whether a subsequence in `Q` contains a
        `np.nan`/`np.inf` value (False)

    include : ndarray, default None
        A list of (zero-based) indices corresponding to the dimensions in `T` that
        must be included in the constrained multidimensional motif search.
        For more information, see Section IV D in:

        `DOI: 10.1109/ICDM.2017.66 \
        <https://www.cs.ucr.edu/~eamonn/Motif_Discovery_ICDM.pdf>`__

    discords : bool, default False
        When set to `True`, this reverses the distance profile to favor discords rather
        than motifs. Note that indices in `include` are still maintained and respected.

    Returns
    -------
    P : ndarray
        Multi-dimensional matrix profile for the window with index equal to
        `start`

    I : ndarray
        Multi-dimensional matrix profile indices for the window with index
        equal to `start`
    """
    P, I = _query_maamp_profile(
        start, T_A, T_B, m, excl_zone, T_B_subseq_isfinite, include, discords
    )
    return P, I


@njit(parallel=True, fastmath=True)
def _compute_multi_D(
    d,
    k,
    idx,
    D,
    T,
    m,
    excl_zone,
    T_A_subseq_isfinite,
    T_B_subseq_isfinite,
    T_A_subseq_squared,
    T_B_subseq_squared,
    QT_even,
    QT_odd,
    QT_first,
):
    """
    A Numba JIT-compiled version of non-normalized (i.e., without z-normalization)
    mSTOMP for parallel computation of the multi-dimensional distance profile

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

    T_A_subseq_isfinite : ndarray
        A boolean array that indicates whether a subsequence in `T_A` contains a
        `np.nan`/`np.inf` value (False)

    T_B_subseq_isfinite : ndarray
        A boolean array that indicates whether a subsequence in `T_B` contains a
        `np.nan`/`np.inf` value (False)

    T_A_subseq_squared : ndarray
        The rolling sum for `T_B * T_B`

    T_B_subseq_squared : ndarray
        The rolling sum for `T_B * T_B`

    QT_even : ndarray
        Dot product between some query sequence,`Q`, and time series, `T`

    QT_odd : ndarray
        Dot product between some query sequence,`Q`, and time series, `T`

    QT_first : ndarray
        QT for the first window relative to the current sliding window

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
            if not T_B_subseq_isfinite[i, idx]:
                D[i] = np.inf
            else:
                D[i] = (
                    T_B_subseq_squared[i, idx]
                    + T_A_subseq_squared[i]
                    - 2.0 * QT_even[i]
                )
        else:
            QT_odd[i, 0] = QT_first[i, idx]
            if not T_B_subseq_isfinite[i, idx]:
                D[i] = np.inf
            else:
                D[i] = (
                    T_B_subseq_squared[i, idx] + T_A_subseq_squared[i] - 2.0 * QT_odd[i]
                )

        D[i][~(T_A_subseq_isfinite[i])] = np.inf
        D[i][D[i] < config.STUMPY_D_SQUARED_THRESHOLD] = 0

    core.apply_exclusion_zone(D, idx, excl_zone)


def _maamp(
    T,
    m,
    range_stop,
    excl_zone,
    T_A_subseq_isfinite,
    T_B_subseq_isfinite,
    T_A_subseq_squared,
    T_B_subseq_squared,
    QT,
    QT_first,
    k,
    range_start=1,
    include=None,
    discords=False,
):
    """
    A Numba JIT-compiled version of non-normailzed (i.e., without z-normalization)
    mSTOMP, a variant of mSTAMP, for parallel computation of the multi-dimensional
    matrix profile and multi-dimensional matrix profile indices. Note that only
    self-joins are supported.

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

    T_A_subseq_isfinite : ndarray
        A boolean array that indicates whether a subsequence in `T_A` contains a
        `np.nan`/`np.inf` value (False)

    T_B_subseq_isfinite : ndarray
        A boolean array that indicates whether a subsequence in `T_B` contains a
        `np.nan`/`np.inf` value (False)

    T_A_subseq_squared : ndarray
        The rolling sum for `T_A * T_A`

    T_B_subseq_squared : ndarray
        The rolling sum for `T_B * T_B`

    QT : ndarray
        Dot product between some query sequence,`Q`, and time series, `T`

    QT_first : ndarray
        QT for the first window relative to the current sliding window

    k : int
        The total number of sliding windows to iterate over

    range_start : int, default 1
        The starting index value along T_B for which to start the matrix
        profile calculation. Default is 1.

    include : ndarray, default None
        A list of (zero-based) indices corresponding to the dimensions in `T` that
        must be included in the constrained multidimensional motif search.
        For more information, see Section IV D in:

        `DOI: 10.1109/ICDM.2017.66 \
        <https://www.cs.ucr.edu/~eamonn/Motif_Discovery_ICDM.pdf>`__

    discords : bool, default False
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
            T_A_subseq_isfinite,
            T_B_subseq_isfinite,
            T_A_subseq_squared,
            T_B_subseq_squared,
            QT_even,
            QT_odd,
            QT_first,
        )

        # `include` processing must occur here since we are dealing with distances
        if include is not None:
            mstump._apply_include(
                D,
                include,
                restricted_indices,
                unrestricted_indices,
                mask,
                tmp_swap,
            )
            start_row_idx = include.shape[0]

        if discords:
            D[start_row_idx:][::-1].sort(axis=0)
        else:
            D[start_row_idx:].sort(axis=0)

        mstump._compute_PI(d, idx, D, D_prime, range_start, P, I)

    return P, I


def maamp(T, m, include=None, discords=False):
    """
    Compute the multi-dimensional non-normalized (i.e., without z-normalization) matrix
    profile

    This is a convenience wrapper around the Numba JIT-compiled parallelized
    `_maamp` function which computes the multi-dimensional matrix profile and
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

    include : list, ndarray, default None
        A list of (zero-based) indices corresponding to the dimensions in `T` that
        must be included in the constrained multidimensional motif search.
        For more information, see Section IV D in:

        `DOI: 10.1109/ICDM.2017.66 \
        <https://www.cs.ucr.edu/~eamonn/Motif_Discovery_ICDM.pdf>`__

    discords : bool, default False
        When set to `True`, this reverses the distance matrix which results in a
        multi-dimensional matrix profile that favors larger matrix profile values
        (i.e., discords) rather than smaller values (i.e., motifs). Note that indices
        in `include` are still maintained and respected.

    Returns
    -------
    P : ndarray
        The multi-dimensional matrix profile. Each row of the array corresponds
        to each matrix profile for a given dimension (i.e., the first row is
        the 1-D matrix profile and the second row is the 2-D matrix profile).

    I : ndarray
        The multi-dimensional matrix profile index where each row of the array
        corresponds to each matrix profile index for a given dimension.

    Notes
    -----
    `DOI: 10.1109/ICDM.2017.66 \
    <https://www.cs.ucr.edu/~eamonn/Motif_Discovery_ICDM.pdf>`__

    See mSTAMP Algorithm
    """
    T_A = T
    T_B = T_A

    T_A, T_A_subseq_isfinite = core.preprocess_non_normalized(T_A, m)
    T_B, T_B_subseq_isfinite = core.preprocess_non_normalized(T_B, m)
    T_A_subseq_squared = np.sum(core.rolling_window(T_A * T_A, m), axis=2)
    T_B_subseq_squared = np.sum(core.rolling_window(T_B * T_B, m), axis=2)

    if T_A.ndim <= 1:  # pragma: no cover
        err = f"T is {T_A.ndim}-dimensional and must be at least 1-dimensional"
        raise ValueError(f"{err}")

    core.check_window_size(m, max_size=min(T_A.shape[1], T_B.shape[1]))

    if include is not None:
        include = mstump._preprocess_include(include)

    d, n = T_B.shape
    k = n - m + 1
    excl_zone = int(
        np.ceil(m / config.STUMPY_EXCL_ZONE_DENOM)
    )  # See Definition 3 and Figure 3

    P = np.empty((d, k), dtype="float64")
    I = np.empty((d, k), dtype="int64")

    start = 0
    stop = k

    P[:, start], I[:, start] = _get_first_maamp_profile(
        start,
        T_A,
        T_B,
        m,
        excl_zone,
        T_B_subseq_isfinite,
        include,
        discords,
    )

    QT, QT_first = mstump._get_multi_QT(start, T_A, m)

    P[:, start + 1 : stop], I[:, start + 1 : stop] = _maamp(
        T_A,
        m,
        stop,
        excl_zone,
        T_A_subseq_isfinite,
        T_B_subseq_isfinite,
        T_A_subseq_squared,
        T_B_subseq_squared,
        QT,
        QT_first,
        k,
        start + 1,
        include,
        discords,
    )

    return P, I
