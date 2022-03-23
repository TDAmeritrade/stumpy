# STUMPY
# Copyright 2019 TD Ameritrade. Released under the terms of the 3-Clause BSD license.
# STUMPY is a trademark of TD Ameritrade IP Company, Inc. All rights reserved.

import logging

import numpy as np

from .aamp_motifs import aamp_motifs, aamp_match
from . import core, config

logger = logging.getLogger(__name__)


def _motifs(
    T,
    P,
    M_T,
    Σ_T,
    excl_zone,
    min_neighbors,
    max_distance,
    cutoff,
    max_matches,
    max_motifs,
    atol=1e-8,
):
    """
    Find the top motifs for time series `T`.

    A subsequence, `Q`, becomes a candidate motif if there are at least `min_neighbor`
    number of other subsequence matches in `T` (outside the exclusion zone) with a
    distance less or equal to `max_distance`.

    Parameters
    ----------
    T : numpy.ndarray
        The time series or sequence

    P : numpy.ndarray
        Matrix Profile of time series, `T`

    M_T : numpy.ndarray
        Sliding mean of time series, `T`

    Σ_T : numpy.ndarray
        Sliding standard deviation of time series, `T`

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
        matches are sorted by distance (starting with the most similar). Note that
        the first match is always the self-match/trivial-match for each motif.

    max_motifs : int
        The maximum number of motifs to return.

    atol : float, default 1e-8
        The absolute tolerance parameter. This value will be added to `max_distance`
        when comparing distances between subsequences.

    Return
    ------
    motif_distances : numpy.ndarray
        The distances corresponding to a set of subsequence matches for each motif.
        Note that the first column always corresponds to the distance for the
        self-match/trivial-match for each motif.

    motif_indices : numpy.ndarray
        The indices corresponding to a set of subsequences matches for each motif.
        Note that the first column always corresponds to the index for the
        self-match/trivial-match for each motif.
    """
    n = T.shape[1]
    l = P.shape[1]
    m = n - l + 1

    motif_indices = []
    motif_distances = []

    candidate_idx = np.argmin(P[-1])
    while len(motif_indices) < max_motifs:
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

        query_matches = match(
            Q,
            T,
            M_T=M_T,
            Σ_T=Σ_T,
            max_matches=None,
            max_distance=max_distance,
            atol=atol,
            query_idx=candidate_idx,
        )

        if len(query_matches) > min_neighbors:
            motif_distances.append(query_matches[:max_matches, 0])
            motif_indices.append(query_matches[:max_matches, 1])

        for idx in query_matches[:, 1]:
            core.apply_exclusion_zone(P, int(idx), excl_zone, np.inf)

        candidate_idx = np.argmin(P[-1])

    motif_distances = core._jagged_list_to_array(
        motif_distances, fill_value=np.nan, dtype=np.float64
    )
    motif_indices = core._jagged_list_to_array(
        motif_indices, fill_value=-1, dtype=np.int64
    )

    return motif_distances, motif_indices


@core.non_normalized(aamp_motifs)
def motifs(
    T,
    P,
    min_neighbors=1,
    max_distance=None,
    cutoff=None,
    max_matches=10,
    max_motifs=1,
    atol=1e-8,
    normalize=True,
    p=2.0,
):
    """
    Discover the top motifs for time series `T`

    A subsequence, `Q`, becomes a candidate motif if there are at least `min_neighbor`
    number of other subsequence matches in `T` (outside the exclusion zone) with a
    distance less or equal to `max_distance`.

    Note that, in the best case scenario, the returned arrays would have shape
    `(max_motifs, max_matches)` and contain all finite values. However, in reality,
    many conditions (see below) need to be satisfied in order for this to be true. Any
    truncation in the number of rows (i.e., motifs)  may be the result of insufficient
    candidate motifs with matches greater than or equal to `min_neighbors` or that the
    matrix profile value for the candidate motif was larger than `cutoff`. Similarly,
    any truncationin in the number of columns (i.e., matches) may be the result of
    insufficient matches being found with distances (to their corresponding candidate
    motif) that are equal to or less than `max_distance`. Only motifs and matches that
    satisfy all of these constraints will be returned.

    If you must return a shape of `(max_motifs, max_matches)`, then you may consider
    specifying a smaller `min_neighors`, a larger `max_distance`, and/or a larger
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
        to be considered a motif. This defaults to `1`, which means that a subsequence
        must have at least one similar match in order to be considered a motif.

    max_distance : float or function, default None
        For a candidate motif, `Q`, and a non-trivial subsequence, `S`, `max_distance`
        is the maximum distance allowed between `Q` and `S` so that `S` is considered
        a match of `Q`. If `max_distance` is a function, then it must be a function
        that accepts a single parameter, `D`, in its function signature, which is the
        distance profile between `Q` and `T`. If None, this defaults to
        `np.nanmax([np.nanmean(D) - 2.0 * np.nanstd(D), np.nanmin(D)])`.

    cutoff : float, default None
        The largest matrix profile value (distance) that a candidate motif is allowed
        to have. If `None`, this defaults to
        `np.nanmax([np.nanmean(P) - 2.0 * np.nanstd(P), np.nanmin(P)])`

    max_matches : int, default 10
        The maximum amount of similar matches of a motif representative to be returned.
        The resulting matches are sorted by distance, so a value of `10` means that the
        indices of the most similar `10` subsequences is returned.
        If `None`, all matches within `max_distance` of the motif representative
        will be returned. Note that the first match is always the
        self-match/trivial-match for each motif.

    max_motifs : int, default 1
        The maximum number of motifs to return

    atol : float, default 1e-8
        The absolute tolerance parameter. This value will be added to `max_distance`
        when comparing distances between subsequences.

    normalize : bool, default True
        When set to `True`, this z-normalizes subsequences prior to computing distances.
        Otherwise, this function gets re-routed to its complementary non-normalized
        equivalent set in the `@core.non_normalized` function decorator.

    p : float, default 2.0
        The p-norm to apply for computing the Minkowski distance. This parameter is
        ignored when `normalize == True`.

    Return
    ------
    motif_distances : numpy.ndarray
        The distances corresponding to a set of subsequence matches for each motif.
        Note that the first column always corresponds to the distance for the
        self-match/trivial-match for each motif.

    motif_indices : numpy.ndarray
        The indices corresponding to a set of subsequences matches for each motif.
        Note that the first column always corresponds to the index for the
        self-match/trivial-match for each motif.

    See Also
    --------
    stumpy.match : Find all matches of a query `Q` in a time series `T`
    stumpy.mmotifs : Discover the top motifs for the multi-dimensional time series `T`
    stumpy.stump : Compute the z-normalized matrix profile
    stumpy.stumped : Compute the z-normalized matrix profile with a distributed dask
        cluster
    stumpy.gpu_stump : Compute the z-normalized matrix profile with one or more GPU
        devices
    stumpy.scrump : Compute an approximate z-normalized matrix profile

    Examples
    --------
    >>> mp = stumpy.stump(np.array([584., -11., 23., 79., 1001., 0., -19.]), m=3)
    >>> stumpy.motifs(
    ...     np.array([584., -11., 23., 79., 1001., 0., -19.]),
    ...     mp[:, 0],
    ...     max_distance=2.0)
    (array([[0.        , 0.11633857]]), array([[0, 4]]))
    """
    T = core._preprocess(T)

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
            f"P is {P.ndim}-dimensional and must be 1-dimensional. "
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

    T, M_T, Σ_T = core.preprocess(T[np.newaxis, :], m)
    P = P[np.newaxis, :].astype(np.float64)

    motif_distances, motif_indices = _motifs(
        T,
        P,
        M_T,
        Σ_T,
        excl_zone,
        min_neighbors,
        max_distance,
        cutoff,
        max_matches,
        max_motifs,
        atol=atol,
    )

    return motif_distances, motif_indices


@core.non_normalized(
    aamp_match,
    exclude=["normalize", "M_T", "Σ_T", "T_subseq_isfinite", "p"],
    replace={"M_T": "T_subseq_isfinite", "Σ_T": None},
)
def match(
    Q,
    T,
    M_T=None,
    Σ_T=None,
    max_distance=None,
    max_matches=None,
    atol=1e-8,
    query_idx=None,
    normalize=True,
    p=2.0,
):
    """
    Find all matches of a query `Q` in a time series `T`

    The indices of subsequences whose distances to `Q` are less than or equal to
    `max_distance`, sorted by distance (lowest to highest). Around each occurrence an
    exclusion zone is applied before searching for the next.

    Parameters
    ----------
    Q : numpy.ndarray
        The query sequence. It doesn't have to be a subsequence of `T`

    T : numpy.ndarray
        The time series of interest

    M_T : numpy.ndarray, default None
        Sliding mean of time series, `T`

    Σ_T : numpy.ndarray, default None
        Sliding standard deviation of time series, `T`

    max_distance : float or function, default None
        Maximum distance between `Q` and a subsequence `S` for `S` to be considered a
        match.
        If a function, then it has to be a function of one argument `D`, which will be
        the distance profile of `Q` with `T` (a 1D numpy array of size `n-m+1`).
        If None, this defaults to
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

    normalize : bool, default True
        When set to `True`, this z-normalizes subsequences prior to computing distances.
        Otherwise, this function gets re-routed to its complementary non-normalized
        equivalent set in the `@core.non_normalized` function decorator.

    p : float, default 2.0
        The p-norm to apply for computing the Minkowski distance. This parameter is
        ignored when `normalize == True`.

    Returns
    -------
    out : numpy.ndarray
        The first column consists of distances of subsequences of `T` whose distances
        to `Q` are less than or equal to `max_distance`, sorted by distance (lowest to
        highest). The second column consists of the corresponding indices in `T`.

    See Also
    --------
    stumpy.motifs : Discover the top motifs for time series `T`
    stumpy.mmotifs : Discover the top motifs for the multi-dimensional time series `T`
    stumpy.stump : Compute the z-normalized matrix profile
    stumpy.stumped : Compute the z-normalized matrix profile with a distributed dask
        cluster
    stumpy.gpu_stump : Compute the z-normalized matrix profile with one or more GPU
        devices
    stumpy.scrump : Compute an approximate z-normalized matrix profile

    Examples
    --------
    >>> stumpy.match(
    ...     np.array([-11.1, 23.4, 79.5, 1001.0]),
    ...     np.array([584., -11., 23., 79., 1001., 0., -19.])
    ...     )
    array([[0.0011129739290248121, 1]], dtype=object)
    """
    Q = core._preprocess(Q)
    T = core._preprocess(T)

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

    if M_T is None or Σ_T is None:  # pragma: no cover
        T, M_T, Σ_T = core.preprocess(T, m)

    D = np.empty((d, n - m + 1))
    for i in range(d):
        D[i, :] = core.mass(Q[i], T[i], M_T[i], Σ_T[i])

    D = np.mean(D, axis=0)
    if not isinstance(max_distance, float):
        max_distance = max_distance(D)

    matches = []

    if query_idx is not None:
        candidate_idx = query_idx
    else:
        candidate_idx = np.argmin(D)

    while (
        D[candidate_idx] <= atol + max_distance
        and np.isfinite(D[candidate_idx])
        and len(matches) < max_matches
    ):
        matches.append([D[candidate_idx], candidate_idx])
        core.apply_exclusion_zone(D, candidate_idx, excl_zone, np.inf)
        candidate_idx = np.argmin(D)

    return np.array(matches, dtype=object)
