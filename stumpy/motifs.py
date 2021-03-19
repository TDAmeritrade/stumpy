# STUMPY
# Copyright 2019 TD Ameritrade. Released under the terms of the 3-Clause BSD license.
# STUMPY is a trademark of TD Ameritrade IP Company, Inc. All rights reserved.

import logging

import numpy as np

from . import core

logger = logging.getLogger(__name__)


def _jagged_list_to_array(a, fill_value, dtype):
    """Fits a 2d jagged list into a 2d numpy array of the specified dtype.
    The resulting array will have a shape of (len(a), l), where l is the length
    of the longest list in a. All other lists will be padded with `fill_value`.

    Example:
    [[2, 1, 1], [0]] with a fill value of -1 will become
    np.array([[2, 1, 1], [0, -1, -1]])

    Parameters
    ----------
    a : list
        Jagged list (list-of-lists) to be converted into a ndarray.

    fill_value : int or float
        Missing entries will be filled with this value.

    dtype : dtype
        The desired data-type for the array.

    Return
    ------
    out : ndarray
        The resuling ndarray of dtype `dtype`.
    """
    if not a:
        return np.array([[]])

    max_length = max([len(row) for row in a])

    out = np.full((len(a), max_length), fill_value, dtype=dtype)
    for i, row in enumerate(a):
        out[i, : row.size] = row

    return out


def _motifs(
    T,
    P,
    M_T,
    Σ_T,
    excl_zone,
    min_neighbors,
    max_distance,
    max_matches,
    max_motifs,
    normalize,
):
    """Find the top motifs of the time series `T`.

    A subsequence `Q` is called a motif representative there are at least
    `min_neighbor` other matches in `T` (outside the exclusion zone) with a
    distance less or equal to `max_distance`.

    Parameters
    ----------
    T : ndarray
        The time series of interest

    P : ndarray
        Matrix Profile of `T` (result of a self-join)

    M_T : ndarray
        Sliding mean for `T`

    Σ_T : ndarray
        Sliding standard deviation for `T`

    excl_zone : int
        Size of the exclusion zone.
        If `None`, defaults to `m/4`, where `m` is the length of `Q`.

    min_neighbors : int
        The minimum amount of similar matches a subsequence needs to have
        to be considered a motif.
        Defaults to `1`. This means, that a subsequence has to have
        at least one similar match to be considered a motif.

    max_distance : float or function, default None
        Maximum distance between a motif representative `Q` and a subsequence `S` for
        `S` to be considered a match of `Q`.
        If `max_distance` is a function, then it has to be a function of one argument
        `D`, which will be the distance profile of `Q` with `T` (a 1D numpy array of
        size `n-m+1`).
        If None, defaults to `max(np.mean(D) - 2 * np.std(D), np.min(D))`.

    max_matches : int
        The maximum amount of similar matches to be returned. The resulting
        matches are sorted by distance, so a value of `10` means that the
        indices of the most similar `10` subsequences is returned. If `None`,
        all matches in the given tolerance range are returned.

    max_motifs : int
        Maximum number of different motifs to return.

    normalize : bool
        When set to `True`, this z-normalizes subsequences prior to computing distances.
        Otherwise, this function gets re-routed to its complementary non-normalized
        equivalent set in the `@core.non_normalized` function decorator.

    Return
    ------
    motif_distances : list


    motif_indices : list

    """
    n = T.shape[1]
    l = P.shape[1]
    m = n - l + 1

    motif_indices = []
    motif_distances = []

    candidate_idx = np.argmin(P[-1])
    while len(motif_indices) < max_motifs:
        profile_value = P[-1, candidate_idx]
        if np.isinf(profile_value):  # pragma: no cover
            break

        # If max_distance is a constant (independent of the distance profile D of Q
        # and T), then we can stop the iteration if the matrix profile value of Q is
        # larger than the maximum distance.
        if isinstance(max_distance, float) and profile_value > max_distance:
            break

        Q = T[:, candidate_idx : candidate_idx + m]

        query_matches = match(
            Q,
            T,
            M_T=M_T,
            Σ_T=Σ_T,
            excl_zone=excl_zone,
            max_matches=None,
            max_distance=max_distance,
            normalize=normalize,
        )

        if len(query_matches) > min_neighbors:
            motif_distances.append(query_matches[:max_matches, 0])
            motif_indices.append(query_matches[:max_matches, 1])

        for idx in query_matches[:, 1]:
            core.apply_exclusion_zone(P, int(idx), excl_zone)

        candidate_idx = np.argmin(P[-1])

    return motif_distances, motif_indices


def motifs(
    T,
    P,
    excl_zone=None,
    min_neighbors=1,
    max_distance=None,
    cutoff=None,
    max_matches=10,
    max_motifs=1,
    normalize=True,
):
    """Find the top motifs of the time series `T`.

    A subsequence `Q` is called a motif representative there are at least
    `min_neighbor` other matches in `T` (outside the exclusion zone) with a
    distance less or equal to `max_distance`.

    Parameters
    ----------
    T : ndarray
        The time series of interest

    P : ndarray
        Matrix Profile of `T` (result of a self-join)

    excl_zone : int, default None
        Size of the exclusion zone.
        If `None`, defaults to `m/4`, where `m` is the length of `Q`.

    min_neighbors : int, default 1
        The minimum amount of similar matches a subsequence needs to have
        to be considered a motif representative.
        Defaults to `1`. This means, that a subsequence has to have
        at least one similar match to be considered a motif.

    max_distance : float or function, default None
        Maximum distance between a motif representative `Q` and a subsequence `S` for
        `S` to be considered a match of `Q`.
        If `max_distance` is a function, then it has to be a function of one argument
        `D`, which will be the distance profile of `Q` with `T` (a 1D numpy array of
        size `n-m+1`).
        If None, defaults to `max(np.mean(D) - 2 * np.std(D), np.min(D))`.

    cutoff : float, default None
        The largest matrix profile value a motif representative is allowed to have.
        If `None`, defaults to `max(np.mean(P) - 2 * np.std(P), np.min(P))`

    max_matches : int, default 10
        The maximum amount of similar matches of a motif representative to be returned.
        The resulting matches are sorted by distance, so a value of `10` means that the
        indices of the most similar `10` subsequences is returned.
        If `None`, all matches within `max_distance` of the motif representative
        will be returned.

    max_motifs : int, default 1
        Maxmimum number of motifs to return.

    normalize : bool, default True
        When set to `True`, this z-normalizes subsequences prior to computing distances.
        Otherwise, this function gets re-routed to its complementary non-normalized
        equivalent set in the `@core.non_normalized` function decorator.

    Return
    ------

    """
    if max_motifs < 1:  # pragma: no cover
        logger.warn(
            "The maximum number of motifs, `max_motifs`, "
            "must be greater than or equal to 1"
        )
        logger.warn("`max_motifs` has been set to `1`")
        max_motifs = 1

    if T.ndim != 1:  # pragma: no cover
        raise ValueError(
            f"T is {T.ndim}-dimensional and must be 1-dimensional. "
            "Multidimensional motif discovery is not yet supported."
        )

    if P.ndim != 1:  # pragma: no cover
        raise ValueError(
            f"T is {P.ndim}-dimensional and must be 1-dimensional. "
            "Multidimensional motif discovery is not yet supported."
        )

    m = T.shape[-1] - P.shape[-1] + 1
    if excl_zone is None:  # pragma: no cover
        excl_zone = int(np.ceil(m / 4))
    if max_matches is None:  # pragma: no cover
        max_matches = np.inf
    if cutoff is None:  # pragma: no cover
        cutoff = max(np.mean(P) - 2 * np.std(P), np.min(P))

    # Maybe it makes sense to set max_distance to some constant default value if None
    # is passed.

    # core.preprocess performs a copy, so modifiying this T will not
    # change the function input time series
    T, M_T, Σ_T = core.preprocess(T[np.newaxis, :], m)
    P = P[np.newaxis, :].astype("float64")

    result = _motifs(
        T=T,
        P=P,
        M_T=M_T,
        Σ_T=Σ_T,
        excl_zone=excl_zone,
        min_neighbors=min_neighbors,
        max_distance=max_distance,
        max_matches=max_matches,
        max_motifs=max_motifs,
        normalize=normalize,
    )
    result = (
        _jagged_list_to_array(result[0], fill_value=np.nan, dtype="float64"),
        _jagged_list_to_array(result[1], fill_value=-1, dtype="int64"),
    )

    return result


def match(
    Q,
    T,
    M_T=None,
    Σ_T=None,
    excl_zone=None,
    max_distance=None,
    max_matches=None,
    normalize=True,
):
    """
    Find all matches of a query `Q` in a time series `T`, i.e. the indices
    of subsequences whose distances to `Q` are less or equal to
    `max_distance`, sorted by distance (lowest to highest).

    Around each occurrence an exclusion zone is applied before searching for the next.

    Parameters
    ----------
    Q : ndarray
        The query sequence. It doesn't have to be a subsequence of `T`

    T : ndarray
        The time series of interest

    M_T : ndarray, default None
        Sliding mean of time series, `T`

    Σ_T : ndarray, default None
        Sliding standard deviation of time series, `T`

    excl_zone : int, default None
        Size of the exclusion zone.
        If `None`, defaults to `m/4`, where `m` is the length of `Q`.

    max_distance : float or function, default None
        Maximum distance between `Q` and a subsequence `S` for `S` to be considered a
        match.
        If a function, then it has to be a function of one argument `D`, which will be
        the distance profile of `Q` with `T` (a 1D numpy array of size `n-m+1`).
        If None, defaults to `max(np.mean(D) - 2 * np.std(D), np.min(D))`, i.e. at
        least the closest match will be returned.

    max_matches : int, default None
        The maximum amount of similar occurrences to be returned. The resulting
        occurrences are sorted by distance, so a value of `10` means that the
        indices of the most similar `10` subsequences is returned. If `None`, then all
        occurrences are returned.

    normalize : bool, default True
        When set to `True`, this z-normalizes subsequences prior to computing distances.
        Otherwise, this function gets re-routed to its complementary non-normalized
        equivalent set in the `@core.non_normalized` function decorator.

    Returns
    -------
    out : ndarray
        The first column consists of distances of subsequences of `T` whose distances
        to `Q` are smaller than `max_distance`, sorted by distance (lowest to highest).
        The second column consist of the corresponding indices in `T`.
    """
    if len(Q.shape) == 1:
        Q = Q[np.newaxis, :]
    if len(T.shape) == 1:
        T = T[np.newaxis, :]

    d, n = T.shape
    m = Q.shape[1]

    if excl_zone is None:  # pragma: no cover
        excl_zone = int(np.ceil(m / 4))
    if max_matches is None:  # pragma: no cover
        max_matches = np.inf

    if np.any(np.isnan(Q)) or np.any(np.isinf(Q)):  # pragma: no cover
        raise ValueError("Q contains illegal values (NaN or inf)")

    if excl_zone is None:  # pragma: no cover
        excl_zone = int(np.ceil(m / 4))

    if max_distance is None:  # pragma: no cover

        def max_distance(D):
            return max(np.mean(D) - 2 * np.std(D), np.min(D))

    if normalize:
        if M_T is None or Σ_T is None:  # pragma: no cover
            T, M_T, Σ_T = core.preprocess(T, m)
        D = [core.mass(Q[i], T[i], M_T[i], Σ_T[i]) for i in range(d)]
    else:
        D = [core.mass_absolute(Q[i], T[i]) for i in range(d)]

    D = np.sum(D, axis=0) / d
    if not isinstance(max_distance, float):
        max_distance = max_distance(D)

    matches = []

    candidate_idx = np.argmin(D)
    while D[candidate_idx] <= max_distance and len(matches) < max_matches:
        matches.append([D[candidate_idx], candidate_idx])
        core.apply_exclusion_zone(D, candidate_idx, excl_zone)
        candidate_idx = np.argmin(D)

    return np.array(matches, dtype=object)
