# STUMPY
# Copyright 2019 TD Ameritrade. Released under the terms of the 3-Clause BSD license.
# STUMPY is a trademark of TD Ameritrade IP Company, Inc. All rights reserved.

import logging

import numpy as np

from . import core, config

logger = logging.getLogger(__name__)


def _aamp_motifs(
    T,
    P,
    T_subseq_isfinite,
    T_squared,
    excl_zone,
    min_neighbors,
    max_distance,
    cutoff,
    max_matches,
    max_motifs,
):
    """
    Find the top non-normalized motifs (i.e., without z-normalization) for time series
    `T`.

    A subsequence, `Q`, becomes a candidate motif if there are at least `min_neighbor`
    number of other subsequence matches in `T` (outside the exclusion zone) with a
    distance less or equal to `max_distance`.

    Parameters
    ----------
    T : numpy.ndarray
        The time series or sequence

    P : numpy.ndarray
        Matrix Profile of `T`

    T_subseq_isfinite : numpy.ndarray
        A boolean array that indicates whether a subsequence in `T` contains a
        `np.nan`/`np.inf` value (False)

    T_squared : numpy.ndarray
        Squared time series or sequence

    excl_zone : int
        Size of the exclusion zone

    min_neighbors : int
        The minimum number of similar matches a subsequence needs to have in order
        to be considered a motif.

    max_distance : float or function
        For a candidate motif, `Q`, and a non-trivial subsequence, `S`, `max_distance`
        is the maximum distance allowed between `Q` and `S` so that `S` is considered
        a match of `Q`. If `max_distance` is a function, then it must be a function
        that accepts a single parameter, `D`, in its function signature, which is the
        distance profile between `Q` and `T`.

    cutoff : float
        The largest matrix profile value (distance) that a candidate motif is allowed
        to have.

    max_matches : int
        The maximum number of similar matches to be returned. The resulting
        matches are sorted by distance (starting with the most similar). Note that the
        first match is always the self-match/trivial-match for each motif.

    max_motifs : int
        The maximum number of motifs to return.

    Return
    ------
    motif_distances : numpy.ndarray
        The distances corresponding to a set of subsequence matches for each motif.
        Note that the first column always corresponds to the distance for the
        self-match/trivial-match for each motif.

    motif_indices : numpy.ndarray
        The indices corresponding to a set of subsequences matches for each motif
        Note that the first column always corresponds to the index for the
        self-match/trivial-match for each motif
    """
    n = T.shape[1]
    l = P.shape[1]
    m = n - l + 1

    motif_indices = []
    motif_distances = []

    candidate_idx = np.argmin(P[-1])
    while len(motif_indices) < max_motifs:
        profile_value = P[-1, candidate_idx]
        if profile_value > cutoff:  # pragma: no cover
            break

        # If max_distance is a constant (independent of the distance profile D of Q
        # and T), then we can stop the iteration if the matrix profile value of Q is
        # larger than the maximum distance.
        if (
            isinstance(max_distance, float) and profile_value > max_distance
        ):  # pragma: no cover
            break

        Q = T[:, candidate_idx : candidate_idx + m]

        query_matches = aamp_match(
            Q,
            T,
            max_matches=None,
            max_distance=max_distance,
        )

        if len(query_matches) > min_neighbors:
            motif_distances.append(query_matches[:max_matches, 0])
            motif_indices.append(query_matches[:max_matches, 1])

        for idx in query_matches[:, 1]:
            core.apply_exclusion_zone(P, int(idx), excl_zone)

        candidate_idx = np.argmin(P[-1])

    motif_distances = core._jagged_list_to_array(
        motif_distances, fill_value=np.nan, dtype=np.float64
    )
    motif_indices = core._jagged_list_to_array(
        motif_indices, fill_value=-1, dtype=np.int64
    )

    return motif_distances, motif_indices


def aamp_motifs(
    T,
    P,
    min_neighbors=1,
    max_distance=None,
    cutoff=None,
    max_matches=10,
    max_motifs=1,
):
    """
    Discover the top non-normalized motifs (i.e., without z-normalization) for time
    series `T`.

    A subsequence, `Q`, becomes a candidate motif if there are at least `min_neighbor`
    number of other subsequence matches in `T` (outside the exclusion zone) with a
    distance less or equal to `max_distance`.

    Parameters
    ----------
    T : numpy.ndarray
            The time series or sequence

    P : numpy.ndarray
        Matrix Profile of `T`

    min_neighbors : int, default 1
        The minimum number of similar matches a subsequence needs to have in order
        to be considered a motif. This defaults to `1`, which means that a
        subsequence must have at least one similar match in order to be considered
        a motif.

    max_distance : float or function, default None
        For a candidate motif, `Q`, and a non-trivial subsequence, `S`,
        `max_distance` is the maximum distance allowed between `Q` and `S` so that
        `S` is considered a match of `Q`. If `max_distance` is a function, then it
        must be a function that accepts a single parameter, `D`, in its function
        signature, which is the distance profile between `Q` and `T`. If None, this
        defaults to `np.nanmax([np.nanmean(D) - 2.0 * np.nanstd(D), np.nanmin(D)])`.

    cutoff : float, default None
        The largest matrix profile value (distance) that a candidate motif is
        allowed to have. If `None`, this defaults to
        `np.nanmax([np.nanmean(P) - 2.0 * np.nanstd(P), np.nanmin(P)])`

    max_matches : int, default 10
        The maximum amount of similar matches of a motif representative to be
        returned. The resulting matches are sorted by distance, so a value of `10`
        means that the indices of the most similar `10` subsequences is returned.
        If `None`, all matches within `max_distance` of the motif representative
        will be returned. Note that the first match is always the
        self-match/trivial-match for each motif.

    max_motifs : int, default 1
        The maximum number of motifs to return.

    Return
    ------
    motif_distances : numpy.ndarray
        The distances corresponding to a set of subsequence matches for each motif.
        Note that the first column always corresponds to the distance for the
        self-match/trivial-match for each motif.

    motif_indices : numpy.ndarray
        The indices corresponding to a set of subsequences matches for each motif
        Note that the first column always corresponds to the index for the
        self-match/trivial-match for each motif.

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
    excl_zone = int(np.ceil(m / config.STUMPY_EXCL_ZONE_DENOM))
    if max_matches is None:  # pragma: no cover
        max_matches = np.inf
    if cutoff is None:  # pragma: no cover
        P_copy = P.copy().astype(np.float64)
        P_copy[np.isinf(P_copy)] = np.nan
        cutoff = np.nanmax(
            [np.nanmean(P_copy) - 2.0 * np.nanstd(P_copy), np.nanmin(P_copy)]
        )

    T, T_subseq_isfinite = core.preprocess_non_normalized(T[np.newaxis, :], m)
    T_squared = np.sum(core.rolling_window(T * T, m), axis=-1)
    P = P[np.newaxis, :].astype(np.float64)

    motif_distances, motif_indices = _aamp_motifs(
        T,
        P,
        T_subseq_isfinite,
        T_squared,
        excl_zone,
        min_neighbors,
        max_distance,
        cutoff,
        max_matches,
        max_motifs,
    )

    return motif_distances, motif_indices


def aamp_match(
    Q,
    T,
    T_subseq_isfinite=None,
    T_squared=None,
    max_distance=None,
    max_matches=None,
):
    """
    Find all matches of a query `Q` in a time series `T`, i.e. the indices
    of subsequences whose distances to `Q` are less or equal to
    `max_distance`, sorted by distance (lowest to highest).

    Around each occurrence an exclusion zone is applied before searching for the next.

    Parameters
    ----------
    Q : numpy.ndarray
        The query sequence. It doesn't have to be a subsequence of `T`

    T : numpy.ndarray
        The time series of interest

    max_distance : float or function, default None
        Maximum distance between `Q` and a subsequence `S` for `S` to be considered a
        match. If a function, then it has to be a function of one argument `D`, which
        will be the distance profile of `Q` with `T` (a 1D numpy array of size `n-m+1`).
        If None, defaults to
        `np.nanmax([np.nanmean(D) - 2 * np.nanstd(D), np.nanmin(D)])` (i.e. at
        least the closest match will be returned).

    max_matches : int, default None
        The maximum amount of similar occurrences to be returned. The resulting
        occurrences are sorted by distance, so a value of `10` means that the
        indices of the most similar `10` subsequences is returned. If `None`, then all
        occurrences are returned.

    Returns
    -------
    out : numpy.ndarray
        The first column consists of distances of subsequences of `T` whose distances
        to `Q` are smaller than `max_distance`, sorted by distance (lowest to highest).
        The second column consists of the corresponding indices in `T`.
    """
    if len(Q.shape) == 1:
        Q = Q[np.newaxis, :]
    if len(T.shape) == 1:
        T = T[np.newaxis, :]

    d, n = T.shape
    m = Q.shape[1]

    excl_zone = int(np.ceil(m / config.STUMPY_EXCL_ZONE_DENOM))
    if max_matches is None:  # pragma: no cover
        max_matches = np.inf

    if np.any(np.isnan(Q)) or np.any(np.isinf(Q)):  # pragma: no cover
        raise ValueError("Q contains illegal values (NaN or inf)")

    if max_distance is None:  # pragma: no cover

        def max_distance(D):
            D_copy = D.copy().astype(np.float64)
            D_copy[np.isinf(D_copy)] = np.nan
            return np.nanmax(
                [np.nanmean(D_copy) - 2.0 * np.nanstd(D_copy), np.nanmin(D_copy)]
            )

    if T_subseq_isfinite is None or T_squared is None:
        T, T_subseq_isfinite = core.preprocess_non_normalized(T, m)
        T_squared = np.sum(core.rolling_window(T * T, m), axis=-1)

    D = [
        core.mass_absolute(Q[i], T[i], T_subseq_isfinite[i], T_squared[i])
        for i in range(d)
    ]

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
