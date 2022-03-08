# STUMPY
# Copyright 2019 TD Ameritrade. Released under the terms of the 3-Clause BSD license.
# STUMPY is a trademark of TD Ameritrade IP Company, Inc. All rights reserved.

import logging

import numpy as np
from scipy.stats import norm
from numba import njit, prange
from functools import lru_cache, partial

from . import core, config
from .maamp import maamp_multi_distance_profile, maamp, maamp_subspace, maamp_mdl

logger = logging.getLogger(__name__)


def _preprocess_include(include):
    """
    A utility function for processing the `include` input

    Parameters
    ----------
    include : numpy.ndarray
        A list of (zero-based) indices corresponding to the dimensions in `T` that
        must be included in the constrained multidimensional motif search.
        For more information, see Section IV D in:

        `DOI: 10.1109/ICDM.2017.66 \
        <https://www.cs.ucr.edu/~eamonn/Motif_Discovery_ICDM.pdf>`__

    Returns
    -------
    include : numpy.ndarray
        Process `include` and remove any redundant index values
    """
    include = np.asarray(include)
    _, idx = np.unique(include, return_index=True)
    if include.shape[0] != idx.shape[0]:  # pragma: no cover
        logger.warning("Removed repeating indices in `include`")
        include = include[np.sort(idx)]

    return include


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
    D : numpy.ndarray
        The multi-dimensional distance profile

    include : numpy.ndarray
        A list of (zero-based) indices corresponding to the dimensions in `T` that
        must be included in the constrained multidimensional motif search.
        For more information, see Section IV D in:

        `DOI: 10.1109/ICDM.2017.66 \
        <https://www.cs.ucr.edu/~eamonn/Motif_Discovery_ICDM.pdf>`__

    restricted_indices : numpy.ndarray, default None
        A list of indices specified in `include` that reside in the first
        `include.shape[0]` rows

    unrestricted_indices : numpy.ndarray, default None
        A list of indices specified in `include` that do not reside in the first
        `include.shape[0]` rows

    mask : numpy.ndarray, default None
        A boolean mask to select for unrestricted indices

    tmp_swap : numpy.ndarray, default None
        A reusable array to aid in array element swapping
    """
    include = _preprocess_include(include)

    if restricted_indices is None:
        restricted_indices = include[include < include.shape[0]]

    if unrestricted_indices is None:
        unrestricted_indices = include[include >= include.shape[0]]

    if mask is None:
        mask = np.ones(include.shape[0], dtype=bool)
        mask[restricted_indices] = False

    if tmp_swap is None:
        tmp_swap = D[: include.shape[0]].copy()
    else:
        tmp_swap[:] = D[: include.shape[0]]

    D[: include.shape[0]] = D[include]
    D[unrestricted_indices] = tmp_swap[mask]


def _multi_mass(Q, T, m, M_T, Σ_T, μ_Q, σ_Q):
    """
    A multi-dimensional wrapper around "Mueen's Algorithm for Similarity Search"
    (MASS) to compute multi-dimensional distance profile.

    Parameters
    ----------
    Q : numpy.ndarray
        Query array or subsequence

    T : numpy.ndarray
        Time series array or sequence

    m : int
        Window size

    M_T : numpy.ndarray
        Sliding mean for `T_A`

    Σ_T : numpy.ndarray
        Sliding standard deviation for `T_A`

    μ_Q : numpy.ndarray
        Mean value of `Q`

    σ_Q : numpy.ndarray
        Standard deviation of `Q`

    Returns
    -------
    D : numpy.ndarray
        Multi-dimensional distance profile
    """
    d, n = T.shape
    k = n - m + 1

    D = np.empty((d, k), dtype=np.float64)

    for i in range(d):
        if np.isinf(μ_Q[i]):
            D[i, :] = np.inf
        else:
            D[i, :] = core.mass(Q[i], T[i], M_T[i], Σ_T[i])

    return D


def _subspace(D, k, include=None, discords=False):
    """
    Compute the k-dimensional matrix profile subspace for a given subsequence index and
    its nearest neighbor index

    Parameters
    ----------
    D : numpy.ndarray
        The multi-dimensional distance profile

    k : int
        The subset number of dimensions out of `D = T.shape[0]`-dimensions to return
        the subspace for. Note that zero-based indexing is used.

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
        S : numpy.ndarray
        An array that contains the `k`th-dimensional subspace for the subsequence. Note
        that `k+1` rows will be returned.
    """
    if discords:
        sorted_idx = D[::-1].argsort(axis=0, kind="mergesort")
    else:
        sorted_idx = D.argsort(axis=0, kind="mergesort")

    # `include` processing occur here since we are dealing with indices, not distances
    if include is not None:
        include = _preprocess_include(include)
        mask = np.in1d(sorted_idx, include)
        include_idx = mask.nonzero()[0]
        exclude_idx = (~mask).nonzero()[0]
        sorted_idx[: include_idx.shape[0]], sorted_idx[include_idx.shape[0] :] = (
            sorted_idx[include_idx],
            sorted_idx[exclude_idx],
        )

    S = sorted_idx[: k + 1]

    return S


@core.non_normalized(maamp_subspace)
def subspace(
    T,
    m,
    subseq_idx,
    nn_idx,
    k,
    include=None,
    discords=False,
    discretize_func=None,
    n_bit=8,
    normalize=True,
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
        the subspace for. Note that zero-based indexing is used.

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

    normalize : bool, default True
        When set to `True`, this z-normalizes subsequences prior to computing distances.
        Otherwise, this function gets re-routed to its complementary non-normalized
        equivalent set in the `@core.non_normalized` function decorator.

    p : float, default 2.0
        The p-norm to apply for computing the Minkowski distance. This parameter is
        ignored when `normalize == True`.

    Returns
    -------
    S : numpy.ndarray
        An array that contains the (singular) `k`th-dimensional subspace for the
        subsequence with index equal to `subseq_idx`. Note that `k+1` rows will be
        returned.

    See Also
    --------
    stumpy.mstump : Compute the multi-dimensional z-normalized matrix profile
    stumpy.mstumped : Compute the multi-dimensional z-normalized matrix profile with
        a distributed dask cluster
    stumpy.mdl : Compute the number of bits needed to compress one array with another
        using the minimum description length (MDL)

    Examples
    --------
    >>> mps, indices = stumpy.mstump(
    ...     np.array([[584., -11., 23., 79., 1001., 0., -19.],
    ...               [  1.,   2.,  4.,  8.,   16., 0.,  32.]]),
    ...     m=3)
    >>> motifs_idx = np.argsort(mps, axis=1)[:, :2]
    >>> stumpy.subspace(
    ...     np.array([[584., -11., 23., 79., 1001., 0., -19.],
    ...               [  1.,   2.,  4.,  8.,   16., 0.,  32.]]),
    ...     m=3,
    ...     subseq_idx=motifs_idx[k][0],
    ...     nn_idx=indices[k][motifs_idx[k][0]],
    ...     k=1)
    array([0, 1])
    """
    T = core._preprocess(T)
    core.check_window_size(m, max_size=T.shape[-1])

    if discretize_func is None:
        bins = _inverse_norm(n_bit)
        discretize_func = partial(_discretize, bins=bins)

    subseqs, _, _ = core.preprocess(T[:, subseq_idx : subseq_idx + m], m)
    subseqs = core.z_norm(subseqs, axis=1)
    neighbors, _, _ = core.preprocess(T[:, nn_idx : nn_idx + m], m)
    neighbors = core.z_norm(neighbors, axis=1)

    disc_subseqs = discretize_func(subseqs)
    disc_neighbors = discretize_func(neighbors)

    D = np.linalg.norm(disc_subseqs - disc_neighbors, axis=1)

    S = _subspace(D, k, include=include, discords=discords)

    return S


@lru_cache()
def _inverse_norm(n_bit=8):  # pragma: no cover
    """
    Generate bin edges from an inverse normal distribution

    This distribution is best suited for z-normalized time series data

    Parameters
    ----------
    n_bit : int, default 8
        The number of bits to be used in generating the inverse normal distribution

    Returns
    -------
    out : numpy.ndarray
        Array of bin edges that can be used for data discretization
    """
    return norm.ppf(np.arange(1, (2**n_bit)) / (2**n_bit))


def _discretize(a, bins, right=True):  # pragma: no cover
    """
    Discretize each row of the input array

    This is equivalent to `np.searchsorted(bins, a)`

    Parameters
    ----------
    a : numpy.ndarray
        The input array

    bins : numpy.ndarray
        The bin edges used to discretize `a`

    right : bool, default True
        Indicates whether the intervals for binning include the right or the left bin
        edge.

    Returns
    -------
    out : numpy.ndarray
        Discretized array
    """
    return np.digitize(a, bins, right=right)


def _mdl(disc_subseqs, disc_neighbors, S, n_bit=8):
    """
    Compute the number of bits needed to compress one array with another
    using the minimum description length (MDL)

    Parameters
    ----------
    disc_subseqs : numpy.ndarray
        The discretized array to be compressed

    disc_neighbors : numpy.ndarray
        The discretized array that will be used as a hypothesis for compression

    S : numpy.ndarray
        An array that contains the `k`th-dimensional subspace to be used

    n_bit : int, default 8
        The number of bits to use for computing the bit size

    Returns
    -------
    bit_size : float
        The total number of bits computed from MDL for representing both input arrays
    """
    ndim = disc_subseqs.shape[0]
    sub_dims, m = disc_subseqs[S].shape

    n_val = len(np.unique(disc_subseqs[S] - disc_neighbors[S]))
    bit_size = n_bit * (2 * ndim * m - sub_dims * m)
    bit_size = bit_size + sub_dims * m * np.log2(n_val) + n_val * n_bit

    return bit_size


@core.non_normalized(maamp_mdl)
def mdl(
    T,
    m,
    subseq_idx,
    nn_idx,
    include=None,
    discords=False,
    discretize_func=None,
    n_bit=8,
    normalize=True,
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

    normalize : bool, default True
        When set to `True`, this z-normalizes subsequences prior to computing distances.
        Otherwise, this function gets re-routed to its complementary non-normalized
        equivalent set in the `@core.non_normalized` function decorator.

    p : float, default 2.0
        The p-norm to apply for computing the Minkowski distance. This parameter is
        ignored when `normalize == True`.

    Returns
    -------
    bit_sizes : numpy.ndarray
        The total number of bits computed from MDL for representing each pair of
        multidimensional subsequences.

    S : list
        A list of numpy.ndarrays that contain the `k`th-dimensional subspaces

    See Also
    --------
    stumpy.mstump : Compute the multi-dimensional z-normalized matrix profile
    stumpy.mstumped : Compute the multi-dimensional z-normalized matrix profile with
        a distributed dask cluster
    stumpy.subspace : Compute the k-dimensional matrix profile subspace for a given
        subsequence index and its nearest neighbor index

    Examples
    --------
    >>> mps, indices = stumpy.mstump(
    ...     np.array([[584., -11., 23., 79., 1001., 0., -19.],
    ...               [  1.,   2.,  4.,  8.,   16., 0.,  32.]]),
    ...     m=3)
    >>> motifs_idx = np.argsort(mps, axis=1)[:, 0]
    >>> stumpy.mdl(
    ...     np.array([[584., -11., 23., 79., 1001., 0., -19.],
    ...               [  1.,   2.,  4.,  8.,   16., 0.,  32.]]),
    ...     m=3,
    ...     subseq_idx=motifs_idx,
    ...     nn_idx=indices[np.arange(motifs_idx.shape[0]), motifs_idx])
    (array([ 80.      , 111.509775]), [array([1]), array([0, 1])])
    """
    T = core._preprocess(T)
    core.check_window_size(m, max_size=T.shape[-1])

    if discretize_func is None:
        bins = _inverse_norm(n_bit)
        discretize_func = partial(_discretize, bins=bins)

    bit_sizes = np.empty(T.shape[0])
    S = [None] * T.shape[0]
    for k in range(T.shape[0]):
        subseqs, _, _ = core.preprocess(T[:, subseq_idx[k] : subseq_idx[k] + m], m)
        subseqs = core.z_norm(subseqs, axis=1)
        neighbors, _, _ = core.preprocess(T[:, nn_idx[k] : nn_idx[k] + m], m)
        neighbors = core.z_norm(neighbors, axis=1)

        disc_subseqs = discretize_func(subseqs)
        disc_neighbors = discretize_func(neighbors)

        D = np.linalg.norm(disc_subseqs - disc_neighbors, axis=1)

        S[k] = _subspace(D, k, include=include, discords=discords)

        bit_sizes[k] = _mdl(disc_subseqs, disc_neighbors, S[k], n_bit=n_bit)

    return bit_sizes, S


def _multi_distance_profile(
    query_idx,
    T_A,
    T_B,
    m,
    M_T,
    Σ_T,
    μ_Q,
    σ_Q,
    include=None,
    discords=False,
    excl_zone=None,
):
    """
    Multi-dimensional wrapper to compute the multi-dimensional distance profile for a
    given query window within the times series or sequence that is denoted by the
    `query_idx` index. Essentially, this is a convenience wrapper around `_multi_mass`.

    Parameters
    ----------
    query_idx : int
        The window index to calculate the multi-dimensional distance profile for

    T_A : numpy.ndarray
        The time series or sequence for which the multi-dimensional distance profile
        is computed

    T_B : numpy.ndarray
        The time series or sequence that contains your query subsequences

    m : int
        Window size

    M_T : numpy.ndarray
        Sliding mean for `T_A`

    Σ_T : numpy.ndarray
        Sliding standard deviation for `T_A`

    μ_Q : numpy.ndarray
        Sliding mean for the query subsequence `T_B`

    σ_Q : numpy.ndarray
        Sliding standard deviation for the query subsequence `T_B`

    include : numpy.ndarray, default None
        A list of (zero-based) indices corresponding to the dimensions in `T` that
        must be included in the constrained multidimensional motif search.
        For more information, see Section IV D in:

        `DOI: 10.1109/ICDM.2017.66 \
        <https://www.cs.ucr.edu/~eamonn/Motif_Discovery_ICDM.pdf>`__

    discords : bool, default False
        When set to `True`, this reverses the distance profile to favor discords rather
        than motifs. Note that indices in `include` are still maintained and respected.

    excl_zone : int, default None
        The half width for the exclusion zone relative to the `query_idx`.

    Returns
    -------
    D : numpy.ndarray
        Multi-dimensional distance profile for the window with index equal to
        `query_idx`
    """
    d, n = T_A.shape
    k = n - m + 1
    start_row_idx = 0
    D = _multi_mass(
        T_B[:, query_idx : query_idx + m],
        T_A,
        m,
        M_T,
        Σ_T,
        μ_Q[:, query_idx],
        σ_Q[:, query_idx],
    )

    if include is not None:
        _apply_include(D, include)
        start_row_idx = include.shape[0]

    if discords:
        D[start_row_idx:][::-1].sort(axis=0, kind="mergesort")
    else:
        D[start_row_idx:].sort(axis=0, kind="mergesort")

    D_prime = np.zeros(k, dtype=np.float64)
    for i in range(d):
        D_prime[:] = D_prime + D[i]
        D[i, :] = D_prime / (i + 1)

    if excl_zone is not None:
        core.apply_exclusion_zone(D, query_idx, excl_zone, np.inf)

    return D


@core.non_normalized(maamp_multi_distance_profile)
def multi_distance_profile(
    query_idx, T, m, include=None, discords=False, normalize=True, p=2.0
):
    """
    Multi-dimensional wrapper to compute the multi-dimensional distance profile for a
    given query window within the times series or sequence that is denoted by the
    `query_idx` index.

    Parameters
    ----------
    query_idx : int
        The window index to calculate the multi-dimensional distance profile for

    T : numpy.ndarray
        The multi-dimensional time series or sequence for which the multi-dimensional
        distance profile will be returned

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

    normalize : bool, default True
        When set to `True`, this z-normalizes subsequences prior to computing distances.
        Otherwise, this function gets re-routed to its complementary non-normalized
        equivalent set in the `@core.non_normalized` function decorator.

    p : float, default 2.0
        The p-norm to apply for computing the Minkowski distance. This parameter is
        ignored when `normalize == True`.

    Returns
    -------
    D : numpy.ndarray
        Multi-dimensional distance profile for the window with index equal to
        `query_idx`
    """
    T, M_T, Σ_T = core.preprocess(T, m)

    if T.ndim <= 1:  # pragma: no cover
        err = f"T is {T.ndim}-dimensional and must be at least 1-dimensional"
        raise ValueError(f"{err}")

    core.check_window_size(m, max_size=T.shape[1])

    if include is not None:  # pragma: no cover
        include = _preprocess_include(include)

    excl_zone = int(
        np.ceil(m / config.STUMPY_EXCL_ZONE_DENOM)
    )  # See Definition 3 and Figure 3

    D = _multi_distance_profile(
        query_idx, T, T, m, M_T, Σ_T, M_T, Σ_T, include, discords, excl_zone
    )

    return D


def _get_first_mstump_profile(
    start, T_A, T_B, m, excl_zone, M_T, Σ_T, μ_Q, σ_Q, include=None, discords=False
):
    """
    Multi-dimensional wrapper to compute the multi-dimensional matrix profile
    and multi-dimensional matrix profile index for a given window within the
    times series or sequence that is denoted by the `start` index.
    Essentially, this is a convenience wrapper around `_multi_mass`. This is a
    convenience wrapper for the `_multi_distance_profile` function but does not
    return the multi-dimensional matrix profile subspace.

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

    M_T : numpy.ndarray
        Sliding mean for `T_A`

    Σ_T : numpy.ndarray
        Sliding standard deviation for `T_A`

    μ_Q : numpy.ndarray
        Sliding mean for `T_B`

    σ_Q : numpy.ndarray
        Sliding standard deviation for `T_B`

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
    D = _multi_distance_profile(
        start, T_A, T_B, m, M_T, Σ_T, μ_Q, σ_Q, include, discords, excl_zone
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


def _get_multi_QT(start, T, m):
    """
    Multi-dimensional wrapper to compute the sliding dot product between
    the query, `T[:, start:start+m])` and the time series, `T`.
    Additionally, compute QT for the first window.

    Parameters
    ----------
    start : int
        The window index for T_B from which to calculate the QT dot product

    T : numpy.ndarray
        The time series or sequence for which to compute the dot product

    m : int
        Window size

    Returns
    -------
    QT : numpy.ndarray
        Given `start`, return the corresponding multi-dimensional QT

    QT_first : numpy.ndarray
        Multi-dimensional QT for the first window
    """
    d = T.shape[0]
    k = T.shape[1] - m + 1

    QT = np.empty((d, k), dtype=np.float64)
    QT_first = np.empty((d, k), dtype=np.float64)

    for i in range(d):
        QT[i] = core.sliding_dot_product(T[i, start : start + m], T[i])
        QT_first[i] = core.sliding_dot_product(T[i, :m], T[i])

    return QT, QT_first


@njit(
    # "(i8, i8, i8, f8[:, :], f8[:, :], i8, i8, f8[:, :], f8[:, :], f8[:, :],"
    # "f8[:, :], f8[:, :], f8[:, :], f8[:, :])",
    parallel=True,
    fastmath=True,
)
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

    D : numpy.ndarray
        The output distance profile

    T : numpy.ndarray
        The time series or sequence for which to compute the matrix profile

    m : int
        Window size

    excl_zone : int
        The half width for the exclusion zone relative to the current
        sliding window

    M_T : numpy.ndarray
        Sliding mean of time series, `T`

    Σ_T : numpy.ndarray
        Sliding standard deviation of time series, `T`

    QT_even : numpy.ndarray
        Dot product between some query sequence,`Q`, and time series, `T`

    QT_odd : numpy.ndarray
        Dot product between some query sequence,`Q`, and time series, `T`

    QT_first : numpy.ndarray
        QT for the first window relative to the current sliding window

    μ_Q : numpy.ndarray
        Mean of the query sequence, `Q`, relative to the current sliding window

    σ_Q : numpy.ndarray
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

    core._apply_exclusion_zone(D, idx, excl_zone, np.inf)


@njit(
    # "(i8, i8, f8[:, :], f8[:], i8, f8[:, :], i8[:, :], f8)",
    parallel=True,
    fastmath=True,
)
def _compute_PI(d, idx, D, D_prime, range_start, P, I, p=2.0):
    """
    A Numba JIT-compiled version of mSTOMP for updating the matrix profile and matrix
    profile indices

    Parameters
    ----------
    d : int
        The total number of dimensions in `T`

    idx : int
        The row index in `T`

    D : numpy.ndarray
        The distance profile

    D_prime : numpy.ndarray
        A reusable array for storing the column-wise cumulative sum of `D`

    range_start : int
        The starting index value along `T` for which to start the matrix
        profile calculation

    P : numpy.ndarray
        The matrix profile

    I : numpy.ndarray
        The matrix profile indices

    p : float, default 2.0
        The p-norm to apply for computing the Minkowski distance.
    """
    D_prime[:] = 0.0
    for i in range(d):
        D_prime = D_prime + np.power(D[i], 1.0 / p)

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
    T: numpy.ndarray
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

    M_T : numpy.ndarray
        Sliding mean of time series, `T`

    Σ_T : numpy.ndarray
        Sliding standard deviation of time series, `T`

    QT : numpy.ndarray
        Dot product between some query sequence,`Q`, and time series, `T`

    QT_first : numpy.ndarray
        QT for the first window relative to the current sliding window

    μ_Q : numpy.ndarray
        Mean of the query sequence, `Q`, relative to the current sliding window

    σ_Q : numpy.ndarray
        Standard deviation of the query sequence, `Q`, relative to the current
        sliding window

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
    QT_odd = QT.copy()
    QT_even = QT.copy()
    d = T.shape[0]

    P = np.empty((d, range_stop - range_start), dtype=np.float64)
    I = np.empty((d, range_stop - range_start), dtype=np.int64)
    D = np.empty((d, k), dtype=np.float64)
    D_prime = np.empty(k, dtype=np.float64)
    start_row_idx = 0

    if include is not None:
        tmp_swap = np.empty((include.shape[0], k), dtype=np.float64)
        restricted_indices = include[include < include.shape[0]]
        unrestricted_indices = include[include >= include.shape[0]]
        mask = np.ones(include.shape[0], dtype=bool)
        mask[restricted_indices] = False

    for idx in range(range_start, range_stop):
        _compute_multi_D(
            d, k, idx, D, T, m, excl_zone, M_T, Σ_T, QT_even, QT_odd, QT_first, μ_Q, σ_Q
        )

        # `include` processing must occur here since we are dealing with distances
        if include is not None:
            _apply_include(
                D, include, restricted_indices, unrestricted_indices, mask, tmp_swap
            )
            start_row_idx = include.shape[0]

        if discords:
            D[start_row_idx:][::-1].sort(axis=0)
        else:
            D[start_row_idx:].sort(axis=0)

        _compute_PI(d, idx, D, D_prime, range_start, P, I)

    return P, I


@core.non_normalized(maamp)
def mstump(T, m, include=None, discords=False, normalize=True, p=2.0):
    """
    Compute the multi-dimensional z-normalized matrix profile

    This is a convenience wrapper around the Numba JIT-compiled parallelized
    `_mstump` function which computes the multi-dimensional matrix profile and
    multi-dimensional matrix profile index according to mSTOMP, a variant of
    mSTAMP. Note that only self-joins are supported.

    Parameters
    ----------
    T : numpy.ndarray
        The time series or sequence for which to compute the multi-dimensional
        matrix profile. Each row in `T` represents data from a different
        dimension while each column in `T` represents data from the same
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

    normalize : bool, default True
        When set to `True`, this z-normalizes subsequences prior to computing distances.
        Otherwise, this function gets re-routed to its complementary non-normalized
        equivalent set in the `@core.non_normalized` function decorator.

    p : float, default 2.0
        The p-norm to apply for computing the Minkowski distance. This parameter is
        ignored when `normalize == True`.

    Returns
    -------
    P : numpy.ndarray
        The multi-dimensional matrix profile. Each row of the array corresponds
        to each matrix profile for a given dimension (i.e., the first row is
        the 1-D matrix profile and the second row is the 2-D matrix profile).

    I : numpy.ndarray
        The multi-dimensional matrix profile index where each row of the array
        corresponds to each matrix profile index for a given dimension.

    See Also
    --------
    stumpy.mstumped : Compute the multi-dimensional z-normalized matrix profile with
        a distributed dask cluster
    stumpy.subspace : Compute the k-dimensional matrix profile subspace for a given
        subsequence index and its nearest neighbor index
    stumpy.mdl : Compute the number of bits needed to compress one array with another
        using the minimum description length (MDL)

    Notes
    -----
    `DOI: 10.1109/ICDM.2017.66 \
    <https://www.cs.ucr.edu/~eamonn/Motif_Discovery_ICDM.pdf>`__

    See mSTAMP Algorithm

    Examples
    --------
    >>> stumpy.mstump(
    ...     np.array([[584., -11., 23., 79., 1001., 0., -19.],
    ...               [  1.,   2.,  4.,  8.,   16., 0.,  32.]]),
    ...     m=3)
    (array([[0.        , 1.43947142, 0.        , 2.69407392, 0.11633857],
            [0.777905  , 2.36179922, 1.50004632, 2.92246722, 0.777905  ]]),
     array([[2, 4, 0, 1, 0],
            [4, 4, 0, 1, 0]]))
    """
    T_A = T
    T_B = T_A

    T_A, M_T, Σ_T = core.preprocess(T_A, m)
    T_B, μ_Q, σ_Q = core.preprocess(T_B, m)

    if T_A.ndim <= 1:  # pragma: no cover
        err = f"T is {T_A.ndim}-dimensional and must be at least 1-dimensional"
        raise ValueError(f"{err}")

    core.check_window_size(m, max_size=min(T_A.shape[1], T_B.shape[1]))

    if include is not None:
        include = _preprocess_include(include)

    d, n = T_B.shape
    k = n - m + 1
    excl_zone = int(
        np.ceil(m / config.STUMPY_EXCL_ZONE_DENOM)
    )  # See Definition 3 and Figure 3

    P = np.empty((d, k), dtype=np.float64)
    I = np.empty((d, k), dtype=np.int64)

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

    return P, I
