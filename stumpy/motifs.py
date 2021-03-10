# STUMPY
# Copyright 2019 TD Ameritrade. Released under the terms of the 3-Clause BSD license.
# STUMPY is a trademark of TD Ameritrade IP Company, Inc. All rights reserved.

import logging

import numpy as np

from . import core

logger = logging.getLogger(__name__)


def _fill_result_with_nans(top_Q_indices, top_Q_values):
    max_number_of_occurrences = 0
    for motif_occurrences in top_Q_indices:
        max_number_of_occurrences = max(
            max_number_of_occurrences, motif_occurrences.size
        )

    result_top_Q_indices = np.full(
        (len(top_Q_indices), max_number_of_occurrences), -1, dtype=int
    )
    result_top_Q_values = np.full(
        (len(top_Q_indices), max_number_of_occurrences), np.nan, dtype=float
    )
    for i, (motif_occurrences, motif_values) in enumerate(
        zip(top_Q_indices, top_Q_values)
    ):
        result_top_Q_indices[i, : motif_occurrences.size] = motif_occurrences
        result_top_Q_values[i, : motif_values.size] = motif_values

    return result_top_Q_indices, result_top_Q_values


def _motifs(
    T, P, k, M_T, Σ_T, excl_zone, min_neighbors, max_occurrences, atol, rtol, normalize,
):
    n = T.shape[1]
    l = P.shape[1]
    m = n - l + 1

    top_Q_indices = []
    top_Q_values = []

    candidate_idx = np.argmin(P[-1])
    while len(top_Q_indices) < k:
        profile_value = P[-1, candidate_idx]
        if np.isinf(profile_value):  # pragma: no cover
            break

        Q = T[:, candidate_idx : candidate_idx + m]

        query_occurrences = occurrences(
            Q,
            T,
            M_T=M_T,
            Σ_T=Σ_T,
            excl_zone=excl_zone,
            max_occurrences=max_occurrences,
            profile_value=profile_value,
            atol=atol,
            rtol=rtol,
            normalize=normalize,
        )

        if len(query_occurrences) >= min_neighbors + 1:
            top_Q_indices.append(query_occurrences[:, 0])
            top_Q_values.append(query_occurrences[:, 1])

        for idx in query_occurrences[:, 0]:
            core.apply_exclusion_zone(P, idx, excl_zone)

        candidate_idx = np.argmin(P[-1])

    return top_Q_indices, top_Q_values


def motifs(
    T,
    P,
    k=1,
    excl_zone=None,
    min_neighbors=1,
    max_occurrences=None,
    atol=None,
    rtol=1.0,
    normalize=True,
):
    """Find the top k motifs of the time series `T`.

    A subsequence `Q` is considered a motif if there is at least `min_neighbor` other
    occurrences in `T` (outside the exclusion zone) with distance smaller than
    `atol + rtol * profile_value`, where `profile_value` is the matrix profile
    value of `Q`.

    Parameters
    ----------
    T : ndarray
        The time series of interest

    P : ndarray
        Matrix Profile of `T` (result of a self-join)

    k : int
        Number of motifs to search. Defaults to `1`.

    excl_zone : int or None, default None
        Size of the exclusion zone.
        If `None`, defaults to `m/4`, where `m` is the length of `Q`.

    min_neighbors : int, default 1
        The minimum amount of similar occurrences a subsequence needs to have
        to be considered a motif.
        Defaults to `1`. This means, that a subsequence has to have
        at least one similar occurrence to be considered a motif.

    max_occurrences : int or None, default None
        The maximum amount of similar occurrences to be returned. The resulting
        occurrences are sorted by distance, so a value of `10` means that the
        indices of the most similar `10` subsequences is returned. If `None`,
        all occurrences in the given tolerance range are returned.

    atol : float or None, default None
        Absolute tolerance (see equation in description).
        If `None`, defaults to `0.25 * np.std(P)`

    rtol : float, default 1.0
        Relative tolerance (see equation in description).

    normalize : bool, default True
        When set to `True`, this z-normalizes subsequences prior to computing distances.
        Otherwise, this function gets re-routed to its complementary non-normalized
        equivalent set in the `@core.non_normalized` function decorator.

    Return
    ------
    top_motif_indices : ndarray
        Array of top motif indices. There are at most `k` rows, each of which represents
        one motif. All found occurrences within a distance of `atol + MP * rtol` (where
        MP is the matrix profile value of the motif representative) are returned. The
        array has either `max_neighbors` columns if specified, or as many columns as
        number of occurrences for the motif with most found occurrences. Everything else
        is filled up with -1.

    top_motif_values : ndarray
        For every occurrence its distance to the motif representative.
    """
    if k < 1:  # pragma: no cover
        logger.warn("The number of motifs, `k`, must be greater than or equal to 1")
        logger.warn("`k` has been set to `1`")
        k = 1

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

    T = T.copy()
    T = T[np.newaxis, :]

    P = P.copy().astype(float)
    P = P[np.newaxis, :]

    m = T.shape[1] - P.shape[1] + 1

    if excl_zone is None:  # pragma: no cover
        excl_zone = int(np.ceil(m / 4))
    if atol is None:  # pragma: no cover
        atol = 0.25 * np.std(P)
    if max_occurrences is None:  # pragma: no cover
        max_occurrences = np.inf

    T, M_T, Σ_T = core.preprocess(T, m)

    result = _motifs(
        T=T,
        P=P,
        k=k,
        M_T=M_T,
        Σ_T=Σ_T,
        excl_zone=excl_zone,
        min_neighbors=min_neighbors,
        max_occurrences=max_occurrences,
        atol=atol,
        rtol=rtol,
        normalize=normalize,
    )
    result = _fill_result_with_nans(result[0], result[1])

    return result


def occurrences(
    Q,
    T,
    M_T=None,
    Σ_T=None,
    excl_zone=None,
    max_occurrences=None,
    profile_value=0.0,
    atol=None,
    rtol=1.0,
    normalize=True,
):
    """
    Find all occurrences of a query `Q` in a time series `T`, i.e. the indices
    of subsequences whose distance to `Q` is smaller than `atol + rtol * profile_value`,
    sorted by distance (lowest to highest).

    Around each occurrence an exclusion zone is applied before searching for the next.

    Parameters
    ----------
    Q : ndarray
        The query sequence. It doesn't have to be a subsequence of `T`

    T : ndarray
        The time series of interest

    M_T : ndarray or None
        Sliding mean of time series, `T`

    Σ_T : ndarray or None
        Sliding standard deviation of time series, `T`

    excl_zone : int or None, default None
        Size of the exclusion zone.
        If `None`, defaults to `m/4`, where `m` is the length of `Q`.

    max_occurrences : int or None, default None
        The maximum amount of similar occurrences to be returned. The resulting
        occurrences are sorted by distance, so a value of `10` means that the
        indices of the most similar `10` subsequences is returned. If `None`, then all
        occurrences are returned.

    profile_value : float, default `0.0`
        Reference value for relative tolerance (if `Q` is a subsequence of `T`,
        this will typically be Qs matrix profile value).

    atol : float or None, default None
        Absolute tolerance (see equation in description).
        If `None`, defaults to `0.25 * np.std(P)`

    rtol : float, default `1.0`
        Relative tolerance (see equation in description

    normalize : bool, default True
        When set to `True`, this z-normalizes subsequences prior to computing distances.
        Otherwise, this function gets re-routed to its complementary non-normalized
        equivalent set in the `@core.non_normalized` function decorator.

    Returns
    -------
    out : ndarray
        The first column consists of indices of subsequences of `T` whose distances
        to `Q` are smaller than `atol + rtol * profile_value`, sorted by distance
        (lowest to highest). The second column consist of the corresponding distances.
    """
    if len(Q.shape) == 1:
        Q = Q[np.newaxis, :]
    if len(T.shape) == 1:
        T = T[np.newaxis, :]

    d, n = T.shape
    m = Q.shape[1]

    if excl_zone is None:  # pragma: no cover
        excl_zone = int(np.ceil(m / 4))
    if max_occurrences is None:  # pragma: no cover
        max_occurrences = np.inf

    if np.any(np.isnan(Q)) or np.any(np.isinf(Q)):  # pragma: no cover
        raise ValueError("Q contains illegal values (NaN or inf)")

    if excl_zone is None:  # pragma: no cover
        excl_zone = int(np.ceil(m / 4))

    if M_T is None or Σ_T is None:  # pragma: no cover
        T, M_T, Σ_T = core.preprocess(T, m)

    D = [core.mass(Q[i], T[i], M_T[i], Σ_T[i]) for i in range(d)]
    D = np.sum(D, axis=0) / d

    occurrences = []

    candidate_idx = np.argmin(D)
    while (
        D[candidate_idx] < atol + rtol * profile_value
        and len(occurrences) < max_occurrences
    ):
        occurrences.append([candidate_idx, D[candidate_idx]])
        core.apply_exclusion_zone(D, candidate_idx, excl_zone)
        candidate_idx = np.argmin(D)

    return np.array(occurrences, dtype=object)
