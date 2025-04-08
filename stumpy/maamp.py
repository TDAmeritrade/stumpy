# STUMPY
# Copyright 2019 TD Ameritrade. Released under the terms of the 3-Clause BSD license.
# STUMPY is a trademark of TD Ameritrade IP Company, Inc. All rights reserved.

from functools import partial

import numpy as np
from numba import njit, prange

from . import config, core
from .mmparray import mparray


def _multi_mass_absolute(Q, T, m, Q_subseq_isfinite, T_subseq_isfinite, p=2.0):
    """
    A multi-dimensional wrapper around "Mueen's Algorithm for Similarity Search"
    (MASS) absolute to compute multi-dimensional non-normalized (i.e., without
    z-normalization distance profile.

    Parameters
    ----------
    Q : numpy.ndarray
        Query array or subsequence

    T : numpy.ndarray
        Time series array or sequence

    m : int
        Window size

    Q_subseq_isfinite : numpy.ndarray
        A boolean array that indicates whether the subsequence in `Q` contains a
        `np.nan`/`np.inf` value (False)

    T_subseq_isfinite : numpy.ndarray
        A boolean array that indicates whether a subsequence in `T` contains a
        `np.nan`/`np.inf` value (False)

    p : float, default 2.0
        The p-norm to apply for computing the Minkowski distance. Minkowski distance is
        typically used with `p` being 1 or 2, which correspond to the Manhattan distance
        and the Euclidean distance, respectively.

    Returns
    -------
    D : numpy.ndarray
        Multi-dimensional non-normalized (i.e., without z-normalization) distance
        profile
    """
    d, n = T.shape
    l = n - m + 1

    D = np.empty((d, l), dtype=np.float64)

    for i in range(d):
        if np.any(~Q_subseq_isfinite[i]):
            D[i, :] = np.inf
        else:
            D[i, :] = core.mass_absolute(Q[i], T[i], p=p)

        D[i][~(T_subseq_isfinite[i])] = np.inf

    return D


def maamp_subspace(
    T,
    m,
    subseq_idx,
    nn_idx,
    k,
    include=None,
    discords=False,
    discretize_func=None,
    n_bit=8,
    p=2.0,
):
    """
    Compute the k-dimensional matrix profile subspace for a given subsequence index and
    its nearest neighbor index

    Parameters
    ----------
    T : numpy.ndarray
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

    include : numpy.ndarray, default None
        A list of (zero-based) indices corresponding to the dimensions in `T` that
        must be included in the constrained multidimensional motif search.
        For more information, see Section IV D in:

        `DOI: 10.1109/ICDM.2017.66 \
        <https://www.cs.ucr.edu/~eamonn/Motif_Discovery_ICDM.pdf>`__

    discords : bool, default False
        When set to `True`, this reverses the distance profile to favor discords rather
        than motifs. Note that indices in `include` are still maintained and respected.

    discretize_func : func, default None
        A function for discretizing each input array. When this is `None`, an
        appropriate discretization function (based on the `normalize` parameter) will
        be applied.

    n_bit : int, default 8
        The number of bits used for discretization. For more information on an
        appropriate value, see Figure 4 in:

        `DOI: 10.1109/ICDM.2016.0069 \
        <https://www.cs.ucr.edu/~eamonn/PID4481999_Matrix%20Profile_III.pdf>`__

        and Figure 2 in:

        `DOI: 10.1109/ICDM.2011.54 \
        <https://www.cs.ucr.edu/~eamonn/ICDM_mdl.pdf>`__

    p : float
        The p-norm to apply for computing the Minkowski distance. Minkowski distance is
        typically used with `p` being 1 or 2, which correspond to the Manhattan distance
        and the Euclidean distance, respectively.

    Returns
    -------
    S : numpy.ndarray
        An array of that contains the (singular) `k`th-dimensional subspace for the
        subsequence with index equal to `subseq_idx`. Note that `k+1` rows will be
        returned.
    """
    T = core._preprocess(T)
    core.check_window_size(m, max_size=T.shape[1], n=T.shape[1])

    subseqs, _ = core.preprocess_non_normalized(T[:, subseq_idx : subseq_idx + m], m)
    neighbors, _ = core.preprocess_non_normalized(T[:, nn_idx : nn_idx + m], m)

    if discretize_func is None:
        T_isfinite = np.isfinite(T)
        T_min = T[T_isfinite].min()
        T_max = T[T_isfinite].max()
        discretize_func = partial(
            _maamp_discretize, a_min=T_min, a_max=T_max, n_bit=n_bit
        )

    disc_subseqs = discretize_func(subseqs)
    disc_neighbors = discretize_func(neighbors)

    D = np.linalg.norm(disc_subseqs - disc_neighbors, axis=1, ord=p)

    S = core._subspace(D, k, include=include, discords=discords)

    return S


def _maamp_discretize(a, a_min, a_max, n_bit=8):  # pragma: no cover
    """
    Discretize each row of the input array

    This distribution is best suited for non-normalized time series data

    Parameters
    ----------
    a : numpy.ndarray
        The input array

    a_min : float
        The minimum value

    a_max : float
        The maximum value

    n_bit : int, default 8
        The number of bits to use for computing the bit size

    Returns
    -------
    out : numpy.ndarray
        Discretized array
    """
    return (
        np.round(((a - a_min) / (a_max - a_min)) * ((2**n_bit) - 1.0)).astype(np.int64)
        + 1
    )


def maamp_mdl(
    T,
    m,
    subseq_idx,
    nn_idx,
    include=None,
    discords=False,
    discretize_func=None,
    n_bit=8,
    p=2.0,
):
    """
    Compute the multi-dimensional number of bits needed to compress one
    multi-dimensional subsequence with another along each of the k-dimensions
    using the minimum description length (MDL)

    Parameters
    ----------
    T : numpy.ndarray
        The time series or sequence for which the multi-dimensional matrix profile,
        multi-dimensional matrix profile indices were computed

    m : int
        Window size

    subseq_idx : numpy.ndarray
        The multi-dimensional subsequence indices in T

    nn_idx : numpy.ndarray
        The multi-dimensional nearest neighbor index in T

    include : numpy.ndarray, default None
        A list of (zero-based) indices corresponding to the dimensions in `T` that
        must be included in the constrained multidimensional motif search.
        For more information, see Section IV D in:

        `DOI: 10.1109/ICDM.2017.66 \
        <https://www.cs.ucr.edu/~eamonn/Motif_Discovery_ICDM.pdf>`__

    discords : bool, default False
        When set to `True`, this reverses the distance profile to favor discords rather
        than motifs. Note that indices in `include` are still maintained and respected.

    discretize_func : func, default None
        A function for discretizing each input array. When this is `None`, an
        appropriate discretization function (based on the `normalization` parameter)
        will be applied.

    n_bit : int, default 8
        The number of bits used for discretization and for computing the bit size. For
        more information on an appropriate value, see Figure 4 in:

        `DOI: 10.1109/ICDM.2016.0069 \
        <https://www.cs.ucr.edu/~eamonn/PID4481999_Matrix%20Profile_III.pdf>`__

        and Figure 2 in:

        `DOI: 10.1109/ICDM.2011.54 \
        <https://www.cs.ucr.edu/~eamonn/ICDM_mdl.pdf>`__

    p : float
        The p-norm to apply for computing the Minkowski distance. Minkowski distance is
        typically used with `p` being 1 or 2, which correspond to the Manhattan distance
        and the Euclidean distance, respectively.

    Returns
    -------
    bit_sizes : numpy.ndarray
        The total number of bits computed from MDL for representing each pair of
        multidimensional subsequences.

    S : list
        A list of numpy.ndarrays that contains the `k`th-dimensional subspaces
    """
    T = core._preprocess(T)
    core.check_window_size(m, max_size=T.shape[1], n=T.shape[1])

    if discretize_func is None:
        T_isfinite = np.isfinite(T)
        T_min = T[T_isfinite].min()
        T_max = T[T_isfinite].max()
        discretize_func = partial(
            _maamp_discretize, a_min=T_min, a_max=T_max, n_bit=n_bit
        )

    bit_sizes = np.empty(T.shape[0])
    S = [None] * T.shape[0]
    for k in range(T.shape[0]):
        subseqs, _ = core.preprocess_non_normalized(
            T[:, subseq_idx[k] : subseq_idx[k] + m], m
        )
        neighbors, _ = core.preprocess_non_normalized(
            T[:, nn_idx[k] : nn_idx[k] + m], m
        )

        disc_subseqs = discretize_func(subseqs)
        disc_neighbors = discretize_func(neighbors)

        D = np.linalg.norm(disc_subseqs - disc_neighbors, axis=1, ord=p)

        S[k] = core._subspace(D, k, include=include, discords=discords)
        bit_sizes[k] = core._mdl(disc_subseqs, disc_neighbors, S[k], n_bit=n_bit)

    return bit_sizes, S


def _maamp_multi_distance_profile(
    query_idx,
    T_A,
    T_B,
    m,
    T_B_subseq_isfinite,
    p=2.0,
    include=None,
    discords=False,
    excl_zone=None,
):
    """
    Multi-dimensional wrapper to compute the multi-dimensional non-normalized (i.e.,
    without z-normalization) distance profile for a given query window within the
    times series or sequence that is denoted by the `query_idx` index. Essentially,
    this is a convenience wrapper around `_multi_mass_absolute`.

    Parameters
    ----------
    query_idx : int
        The window index to calculate the multi-dimensional distance profile

    T_A : numpy.ndarray
        The time series or sequence for which the multi-dimensional distance profile
        will be returned

    T_B : numpy.ndarray
        The time series or sequence that contains your query subsequences

    m : int
        Window size

    T_B_subseq_isfinite : numpy.ndarray
        A boolean array that indicates whether a subsequence in `T_B` contains a
        `np.nan`/`np.inf` value (False)

    p : float
        The p-norm to apply for computing the Minkowski distance. Minkowski distance is
        typically used with `p` being 1 or 2, which correspond to the Manhattan distance
        and the Euclidean distance, respectively.

    include : numpy.ndarray, default None
        A list of (zero-based) indices corresponding to the dimensions in `T` that
        must be included in the constrained multidimensional motif search.
        For more information, see Section IV D in:

        `DOI: 10.1109/ICDM.2017.66 \
        <https://www.cs.ucr.edu/~eamonn/Motif_Discovery_ICDM.pdf>`__

    discords : bool, default False
        When set to `True`, this reverses the distance profile to favor discords rather
        than motifs. Note that indices in `include` are still maintained and respected.

    excl_zone : int
        The half width for the exclusion zone relative to the `query_idx`.

    Returns
    -------
    D : numpy.ndarray
        Multi-dimensional distance profile for the window with index equal to
        `query_idx`
    """
    d, n = T_A.shape
    l = n - m + 1
    start_row_idx = 0
    D = _multi_mass_absolute(
        T_B[:, query_idx : query_idx + m],
        T_A,
        m,
        T_B_subseq_isfinite[:, query_idx],
        T_B_subseq_isfinite,
        p,
    )

    if include is not None:
        core._apply_include(D, include)
        start_row_idx = include.shape[0]

    if discords:
        D[start_row_idx:][::-1].sort(axis=0, kind="mergesort")
    else:
        D[start_row_idx:].sort(axis=0, kind="mergesort")

    D_prime = np.zeros(l, dtype=np.float64)
    for i in range(d):
        D_prime[:] = D_prime + D[i]
        D[i, :] = D_prime / (i + 1)

    if excl_zone is not None:
        core.apply_exclusion_zone(D, query_idx, excl_zone, np.inf)

    return D


def maamp_multi_distance_profile(query_idx, T, m, include=None, discords=False, p=2.0):
    """
    Multi-dimensional wrapper to compute the multi-dimensional non-normalized (i.e.,
    without z-normalization) distance profile for a given query window within the
    times series or sequence that is denoted by the `query_idx` index.

    Parameters
    ----------
    query_idx : int
        The window index to calculate the multi-dimensional distance profile

    T : numpy.ndarray
        The time series or sequence for which the multi-dimensional distance profile
        will be returned

    m : int
        Window size

    include : numpy.ndarray, default None
        A list of (zero-based) indices corresponding to the dimensions in `T` that
        must be included in the constrained multidimensional motif search.
        For more information, see Section IV D in:

        `DOI: 10.1109/ICDM.2017.66 \
        <https://www.cs.ucr.edu/~eamonn/Motif_Discovery_ICDM.pdf>`__

    discords : bool, default False
        When set to `True`, this reverses the distance profile to favor discords rather
        than motifs. Note that indices in `include` are still maintained and respected.

    p : float, default 2.0
        The p-norm to apply for computing the Minkowski distance. Minkowski distance is
        typically used with `p` being 1 or 2, which correspond to the Manhattan distance
        and the Euclidean distance, respectively.

    Returns
    -------
    D : numpy.ndarray
        Multi-dimensional distance profile for the window with index equal to
        `query_idx`
    """
    T, T_subseq_isfinite = core.preprocess_non_normalized(T, m)

    if T.ndim <= 1:  # pragma: no cover
        err = f"T is {T.ndim}-dimensional and must be at least 2-dimensional"
        raise ValueError(f"{err}")

    core.check_window_size(m, max_size=T.shape[1], n=T.shape[1])

    if include is not None:  # pragma: no cover
        include = core._preprocess_include(include)

    excl_zone = int(
        np.ceil(m / config.STUMPY_EXCL_ZONE_DENOM)
    )  # See Definition 3 and Figure 3

    D = _maamp_multi_distance_profile(
        query_idx, T, T, m, T_subseq_isfinite, p, include, discords, excl_zone
    )

    return D


def _get_first_maamp_profile(
    start,
    T_A,
    T_B,
    m,
    excl_zone,
    T_B_subseq_isfinite,
    p=2.0,
    include=None,
    discords=False,
):
    """
    Multi-dimensional wrapper to compute the non-normalized (i.e., without
    z-normalization multi-dimensional matrix profile and multi-dimensional matrix
    profile index for a given window within the times series or sequence that is denoted
    by the `start` index. Essentially, this is a convenience wrapper around
    `_multi_mass_absolute`. This is a convenience wrapper for the
    `_maamp_multi_distance_profile` function but does not return the multi-dimensional
    matrix profile subspace.

    Parameters
    ----------
    start : int
        The window index to calculate the first multi-dimensional matrix profile,
        multi-dimensional matrix profile indices, and multi-dimensional subspace.

    T_A : numpy.ndarray
        The time series or sequence for which the multi-dimensional matrix profile,
        multi-dimensional matrix profile indices, and multi-dimensional subspace will be
        returned

    T_B : numpy.ndarray
        The time series or sequence that contains your query subsequences

    m : int
        Window size

    excl_zone : int
        The half width for the exclusion zone relative to the `start`.

    T_B_subseq_isfinite : numpy.ndarray
        A boolean array that indicates whether a subsequence in `T_B` contains a
        `np.nan`/`np.inf` value (False)

    p : float, default 2.0
        The p-norm to apply for computing the Minkowski distance. Minkowski distance is
        typically used with `p` being 1 or 2, which correspond to the Manhattan distance
        and the Euclidean distance, respectively.

    include : numpy.ndarray, default None
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
    P : numpy.ndarray
        Multi-dimensional matrix profile for the window with index equal to
        `start`

    I : numpy.ndarray
        Multi-dimensional matrix profile indices for the window with index
        equal to `start`
    """
    D = _maamp_multi_distance_profile(
        start, T_A, T_B, m, T_B_subseq_isfinite, p, include, discords, excl_zone
    )

    d = T_A.shape[0]
    P = np.full(d, np.inf, dtype=np.float64)
    I = np.full(d, -1, dtype=np.int64)

    for i in range(d):
        min_index = np.argmin(D[i])
        I[i] = min_index
        P[i] = D[i, min_index]
        if np.isinf(P[i]):  # pragma nocover
            I[i] = -1

    return P, I


def _get_multi_p_norm(start, T, m, p=2.0):
    """
    Multi-dimensional wrapper to compute the p-norm between the
    query, `T[:, start:start+m])` and the time series, `T`. Additionally, compute
    p-norm for the first window.

    Parameters
    ----------
    start : int
        The window index for T_B from which to calculate the QT dot product

    T : numpy.ndarray
        The time series or sequence for which to compute the dot product

    m : int
        Window size

    p : float
        The p-norm to apply for computing the Minkowski distance. Minkowski distance is
        typically used with `p` being 1 or 2, which correspond to the Manhattan distance
        and the Euclidean distance, respectively.

    Returns
    -------
    p_norm : numpy.ndarray
        Given `start`, return the corresponding multi-dimensional p-norm

    p_norm_first : numpy.ndarray
        Multi-dimensional p-norm for the first window
    """
    d = T.shape[0]
    l = T.shape[1] - m + 1

    p_norm = np.empty((d, l), dtype=np.float64)
    p_norm_first = np.empty((d, l), dtype=np.float64)
    for i in range(d):
        p_norm[i] = np.power(core.mass_absolute(T[i, start : start + m], T[i], p=p), p)
        p_norm_first[i] = np.power(core.mass_absolute(T[i, :m], T[i], p=p), p)

    return p_norm, p_norm_first


@njit(
    # "(i8, i8, i8, f8[:, :], f8[:, :], i8, i8, b1[:, :], b1[:, :], f8,"
    # "f8[:, :], f8[:, :], f8[:, :])",
    parallel=True,
    fastmath=config.STUMPY_FASTMATH_FLAGS,
)
def _compute_multi_p_norm(
    d,
    k,
    idx,
    p_norm,
    T,
    m,
    excl_zone,
    T_A_subseq_isfinite,
    T_B_subseq_isfinite,
    p,
    p_norm_even,
    p_norm_odd,
    p_norm_first,
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

    p_norm : numpy.ndarray
        The output p_norm

    T : numpy.ndarray
        The time series or sequence for which to compute the matrix profile

    m : int
        Window size

    excl_zone : int
        The half width for the exclusion zone relative to the current
        sliding window

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

    p_norm_even : numpy.ndarray
        The even input p-norm array between some query sequence,`Q`, and
        time series, `T`

    p_norm_odd : numpy.ndarray
        The odd input p-norm array between some query sequence,`Q`, and
        time series, `T`

    p_norm_first : numpy.ndarray
        The p-norm between the first query sequence,`Q`, and time series, `T`

    Returns
    -------
    None

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
                p_norm_even[i, j] = (
                    p_norm_odd[i, j - 1]
                    - abs(T[i, idx - 1] - T[i, j - 1]) ** p
                    + abs(T[i, idx + m - 1] - T[i, j + m - 1]) ** p
                )
            else:
                # Odd
                p_norm_odd[i, j] = (
                    p_norm_even[i, j - 1]
                    - abs(T[i, idx - 1] - T[i, j - 1]) ** p
                    + abs(T[i, idx + m - 1] - T[i, j + m - 1]) ** p
                )

        if idx % 2 == 0:
            p_norm_even[i, 0] = p_norm_first[i, idx]
            if not T_B_subseq_isfinite[i, idx]:
                p_norm[i] = np.inf
            else:
                p_norm[i] = p_norm_even[i]
        else:
            p_norm_odd[i, 0] = p_norm_first[i, idx]
            if not T_B_subseq_isfinite[i, idx]:
                p_norm[i] = np.inf
            else:
                p_norm[i] = p_norm_odd[i]

        p_norm[i][~(T_A_subseq_isfinite[i])] = np.inf
        p_norm[i][p_norm[i] < config.STUMPY_P_NORM_THRESHOLD] = 0

    core._apply_exclusion_zone(p_norm, idx, excl_zone, np.inf)


def _maamp(
    T,
    m,
    range_stop,
    excl_zone,
    T_A_subseq_isfinite,
    T_B_subseq_isfinite,
    p,
    p_norm,
    p_norm_first,
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
    T : numpy.ndarray
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

    p_norm : numpy.ndarray
        The input p-norm array between some query sequence,`Q`, and time series, `T`

    p_norm_first : numpy.ndarray
        The p-norm between the first query sequence,`Q`, and time series, `T`

    k : int
        The total number of sliding windows to iterate over

    range_start : int, default 1
        The starting index value along T_B for which to start the matrix
        profile calculation. Default is 1.

    include : numpy.ndarray, default None
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
    P : numpy.ndarray
        The multi-dimensional matrix profile. Each row of the array corresponds
        to each matrix profile for a given dimension (i.e., the first row is the
        1-D matrix profile and the second row is the 2-D matrix profile).

    I : numpy.ndarray
        The multi-dimensional matrix profile index where each row of the array
        corresponds to each matrix profile index for a given dimension.

    Notes
    -----
    `DOI: 10.1109/ICDM.2017.66 \
    <https://www.cs.ucr.edu/~eamonn/Motif_Discovery_ICDM.pdf>`__

    See mSTAMP Algorithm
    """
    p_norm_odd = p_norm.copy()
    p_norm_even = p_norm.copy()
    d = T.shape[0]

    P = np.empty((d, range_stop - range_start), dtype=np.float64)
    I = np.empty((d, range_stop - range_start), dtype=np.int64)
    p_norm = np.empty((d, k), dtype=np.float64)
    p_norm_prime = np.empty(k, dtype=np.float64)
    start_row_idx = 0

    if include is not None:
        tmp_swap = np.empty((include.shape[0], k), dtype=np.float64)
        restricted_indices = include[include < include.shape[0]]
        unrestricted_indices = include[include >= include.shape[0]]
        mask = np.ones(include.shape[0], dtype=bool)
        mask[restricted_indices] = False

    for idx in range(range_start, range_stop):
        _compute_multi_p_norm(
            d,
            k,
            idx,
            p_norm,
            T,
            m,
            excl_zone,
            T_A_subseq_isfinite,
            T_B_subseq_isfinite,
            p,
            p_norm_even,
            p_norm_odd,
            p_norm_first,
        )

        # `include` processing must occur here since we are dealing with distances
        if include is not None:
            core._apply_include(
                p_norm,
                include,
                restricted_indices,
                unrestricted_indices,
                mask,
                tmp_swap,
            )
            start_row_idx = include.shape[0]

        if discords:
            p_norm[start_row_idx:][::-1].sort(axis=0)
        else:
            p_norm[start_row_idx:].sort(axis=0)

        core._compute_multi_PI(d, idx, p_norm, p_norm_prime, range_start, P, I, p)

    return P, I


def maamp(T, m, include=None, discords=False, p=2.0):
    """
    Compute the multi-dimensional non-normalized (i.e., without z-normalization) matrix
    profile

    This is a convenience wrapper around the Numba JIT-compiled parallelized
    `_maamp` function which computes the multi-dimensional matrix profile and
    multi-dimensional matrix profile index according to mSTOMP, a variant of
    mSTAMP. Note that only self-joins are supported.

    Parameters
    ----------
    T : numpy.ndarray
        The time series or sequence for which to compute the multi-dimensional
        matrix profile. Each row in `T` represents data from the same
        dimension while each column in `T` represents data from a different
        dimension.

    m : int
        Window size

    include : list, numpy.ndarray, default None
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

    p : float, default 2.0
        The p-norm to apply for computing the Minkowski distance. Minkowski distance is
        typically used with `p` being 1 or 2, which correspond to the Manhattan distance
        and the Euclidean distance, respectively.

    Returns
    -------
    P : numpy.ndarray
        The multi-dimensional matrix profile. Each row of the array corresponds
        to each matrix profile for a given dimension (i.e., the first row is
        the 1-D matrix profile and the second row is the 2-D matrix profile).

    I : numpy.ndarray
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

    if T_A.ndim <= 1:  # pragma: no cover
        err = f"T is {T_A.ndim}-dimensional and must be at least 1-dimensional"
        raise ValueError(f"{err}")

    core.check_window_size(m, max_size=min(T_A.shape[1], T_B.shape[1]), n=T_A.shape[1])

    if include is not None:
        include = core._preprocess_include(include)

    d, n = T_B.shape
    l = n - m + 1
    excl_zone = int(
        np.ceil(m / config.STUMPY_EXCL_ZONE_DENOM)
    )  # See Definition 3 and Figure 3

    P = np.empty((d, l), dtype=np.float64)
    I = np.empty((d, l), dtype=np.int64)

    start = 0
    stop = l

    P[:, start], I[:, start] = _get_first_maamp_profile(
        start,
        T_A,
        T_B,
        m,
        excl_zone,
        T_B_subseq_isfinite,
        p,
        include,
        discords,
    )

    p_norm, p_norm_first = _get_multi_p_norm(start, T_A, m, p=p)

    P[:, start + 1 : stop], I[:, start + 1 : stop] = _maamp(
        T_A,
        m,
        stop,
        excl_zone,
        T_A_subseq_isfinite,
        T_B_subseq_isfinite,
        p,
        p_norm,
        p_norm_first,
        l,
        start + 1,
        include,
        discords,
    )

    return mparray(P_=P, I_=I)
