# STUMPY
# Copyright 2019 TD Ameritrade. Released under the terms of the 3-Clause BSD license.
# STUMPY is a trademark of TD Ameritrade IP Company, Inc. All rights reserved.

import warnings

import numpy as np

from . import config, core


def _aamp_motifs(
    T,
    P,
    T_subseq_isfinite,
    p,
    excl_zone,
    min_neighbors,
    max_distance,
    cutoff,
    max_matches,
    max_motifs,
    atol=1e-8,
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

    p : float
        The p-norm to apply for computing the Minkowski distance. Minkowski distance is
        typically used with `p` being 1 or 2, which correspond to the Manhattan distance
        and the Euclidean distance, respectively.

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

    atol : float, default 1e-8
        The absolute tolerance parameter. This value will be added to `max_distance`
        when comparing distances between subsequences.

    Returns
    -------
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
    for _ in range(l):
        if len(motif_indices) >= max_motifs:
            break

        profile_value = P[-1, candidate_idx]
        if profile_value > cutoff or not np.isfinite(profile_value):  # pragma: no cover
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
            atol=atol,
            query_idx=candidate_idx,
            p=p,
        )

        if len(query_matches) > min_neighbors:
            motif_distances.append(query_matches[:max_matches, 0])
            motif_indices.append(query_matches[:max_matches, 1])

        if len(query_matches) == 0:  # pragma: no cover
            query_matches = np.array([[np.nan, candidate_idx]])

        for idx in query_matches[:, 1]:
            # Since the query motif is also included as the first item in the list of
            # `query_matches`, the exclusion zone is also applied to the query motif!
            core.apply_exclusion_zone(P, int(idx), excl_zone, np.inf)

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
    atol=1e-8,
    p=2.0,
):
    """
    Discover the top non-normalized motifs (i.e., without z-normalization) for time
    series `T`.

    A subsequence, `Q`, becomes a candidate motif if there are at least `min_neighbor`
    number of other subsequence matches in `T` (outside the exclusion zone) with a
    distance less or equal to `max_distance`.

    Note that, in the best case scenario, the returned arrays would have shape
    `(max_motifs, max_matches)` and contain all finite values. However, in reality,
    many conditions (see below) need to be satisfied in order for this to be true. Any
    truncation in the number of rows (i.e., motifs)  may be the result of insufficient
    candidate motifs with matches greater than or equal to `min_neighbors` or that the
    matrix profile value for the candidate motif was larger than `cutoff`. Similarly,
    any truncation in the number of columns (i.e., matches) may be the result of
    insufficient matches being found with distances (to their corresponding candidate
    motif) that are equal to or less than `max_distance`. Only motifs and matches that
    satisfy all of these constraints will be returned.

    If you must return a shape of `(max_motifs, max_matches)`, then you may consider
    specifying a smaller `min_neighbors`, a larger `max_distance`, and/or a larger
    `cutoff`. For example, while it is ill advised, setting `min_neighbors=1`,
    `max_distance=np.inf`, and `cutoff=np.inf` will ensure that the shape of the output
    arrays will be `(max_motifs, max_matches)`. However, given the lack of constraints,
    the quality of each motif and the quality of each match may be drastically
    different. Setting appropriate conditions will help ensure appropriately
    constrained results that may be easier to interpret.

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

    atol : float, default 1e-8
        The absolute tolerance parameter. This value will be added to `max_distance`
        when comparing distances between subsequences.

    p : float, default 2.0
        The p-norm to apply for computing the Minkowski distance. Minkowski distance is
        typically used with `p` being 1 or 2, which correspond to the Manhattan distance
        and the Euclidean distance, respectively.

    Returns
    -------
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
        msg = "The maximum number of motifs, `max_motifs`, "
        msg += "must be greater than or equal to 1.\n"
        msg += "`max_motifs` has been set to `1`"
        warnings.warn(msg)
        max_motifs = 1

    if T.ndim != 1:  # pragma: no cover
        raise ValueError(
            f"T is {T.ndim}-dimensional and must be 1-dimensional. "
            "Multidimensional motif discovery is not yet supported."
        )

    if P.ndim != 1:  # pragma: no cover
        raise ValueError(
            f"P is {P.ndim}-dimensional and must be 1-dimensional. "
            "Multidimensional motif discovery is not yet supported."
        )

    m = T.shape[-1] - P.shape[-1] + 1
    excl_zone = int(np.ceil(m / config.STUMPY_EXCL_ZONE_DENOM))
    if max_matches is None:  # pragma: no cover
        max_matches = P.shape[-1]
    if cutoff is None:  # pragma: no cover
        P_copy = P.copy().astype(np.float64)
        P_copy[np.isinf(P_copy)] = np.nan
        cutoff = np.nanmax(
            [np.nanmean(P_copy) - 2.0 * np.nanstd(P_copy), np.nanmin(P_copy)]
        )

    if cutoff == 0.0:  # pragma: no cover
        suggested_cutoff = np.partition(P, 1)[1]
        msg = "The `cutoff` has been set to 0.0 and may result in little/no candidate "
        msg += "motifs being identified.\n"
        msg += "You may consider relaxing the constraint by increasing the `cutoff` "
        msg += f"(e.g., cutoff={suggested_cutoff})."
        warnings.warn(msg)

    T, T_subseq_isfinite = core.preprocess_non_normalized(np.expand_dims(T, 0), m)
    P = np.expand_dims(P, 0).astype(np.float64)

    motif_distances, motif_indices = _aamp_motifs(
        T,
        P,
        T_subseq_isfinite,
        p,
        excl_zone,
        min_neighbors,
        max_distance,
        cutoff,
        max_matches,
        max_motifs,
        atol=atol,
    )

    if motif_distances.shape[1] == 0:  # pragma: no cover
        msg = "No motifs were found. You may consider increasing the `cutoff` "
        msg += f"(e.g., cutoff={2. * cutoff}) and/or increasing the `max_distance `"
        msg += "(e.g., max_distance=np.inf)."
        warnings.warn(msg)

    return motif_distances, motif_indices


def aamp_match(
    Q,
    T,
    T_subseq_isfinite=None,
    max_distance=None,
    max_matches=None,
    atol=1e-8,
    query_idx=None,
    p=2.0,
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

    T_subseq_isfinite : numpy.ndarray, default None
        A boolean array that indicates whether a subsequence in `T` contains a
        `np.nan`/`np.inf` value (False)

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

    atol : float, default 1e-8
        The absolute tolerance parameter. This value will be added to `max_distance`
        when comparing distances between subsequences.

    query_idx : int, default None
        This is the index position along the time series, `T`, where the query
        subsequence, `Q`, is located.
        `query_idx` should only be used when the matrix profile is a self-join and
        should be set to `None` for matrix profiles computed from AB-joins.
        If `query_idx` is set to a specific integer value, then this will help ensure
        that the self-match will be returned first.

    p : float, default 2.0
        The p-norm to apply for computing the Minkowski distance. Minkowski distance is
        typically used with `p` being 1 or 2, which correspond to the Manhattan distance
        and the Euclidean distance, respectively.

    Returns
    -------
    out : numpy.ndarray
        The first column consists of distances of subsequences of `T` whose distances
        to `Q` are less than or equal to`max_distance`, sorted by distance (lowest to
        highest). The second column consists of the corresponding indices in `T`.
    """
    if np.any(np.isnan(Q)) or np.any(np.isinf(Q)):  # pragma: no cover
        raise ValueError("Q contains illegal values (NaN or inf)")

    if len(Q.shape) == 1:
        Q = np.expand_dims(Q, 0)
    if len(T.shape) == 1:
        T = np.expand_dims(T, 0)

    d, n = T.shape
    m = Q.shape[1]
    excl_zone = int(np.ceil(m / config.STUMPY_EXCL_ZONE_DENOM))

    if T_subseq_isfinite is None:
        T, T_subseq_isfinite = core.preprocess_non_normalized(T, m)
    if len(T_subseq_isfinite.shape) == 1:
        T_subseq_isfinite = np.expand_dims(T_subseq_isfinite, 0)

    D = np.empty((d, n - m + 1))
    for i in range(d):
        D[i, :] = core.mass_absolute(Q[i], T[i], T_subseq_isfinite[i], p=p)
    D = np.mean(D, axis=0)

    return core._find_matches(
        D,
        excl_zone,
        max_distance=max_distance,
        max_matches=max_matches,
        query_idx=query_idx,
        atol=atol,
    )
