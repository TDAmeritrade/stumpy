import numpy as np

from . import core


def motifs(
    T,
    P,
    M_T=None,
    Σ_T=None,
    k=1,
    excl_zone=None,
    min_neighbors=1,
    max_occurrences=None,
    atol=None,
    rtol=1.0,
    aamp=False,
):
    """Find the top k motifs of the time series `T`.

    A subsequence `Q` is considered a motif if there is at least one other
    occurrence in `T` (outside the exclusion zone) with distance smaller than
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

    excl_zone : int or None
        Size of the exclusion zone.
        If `None`, defaults to `m/4`, where `m` is the length of `Q`.

    min_neighbors : int
        The minimum amount of similar occurrences a subsequence needs to have
        to be considered a motif.
        Defaults to `1`. This means, that a subsequence has to have
        at least one similar occurence to be considered a motif.

    max_occurrences : int
        The maximum amount of similar occurrences to be returned. The resulting
        occurrences are sorted by distance, so a value of `10` means that the
        indices of the most similar `10` subsequences is returned. If `None`,
        all occurrences are returned.
        Defaults to `None`

    atol : float or None
        Absolute tolerance (see equation in description).
        If `None`, defaults to `0.25 * np.std(P)`

    rtol : float
        Relative tolerance (see equation in description).
        Defaults to `1.0`

    aamp : boolean
        `True`, if no z-normalization should applied.
        Defaults to `False`

    Return
    ------
    top_motif_indices : list
        Each item is another list, with each item representing an occurrence of the
        motif in `T`, sorted by distance from the "original occurrence", i.e. the
        subsequence of `T` starting at the index in the first place of the list

    top_motif_values : list
        The matrix profile values of the "original occurrences"
    """
    if k < 1:  # pragma: no cover
        raise ValueError("k (the number of motifs) has to be at least 1")

    T = T.copy()
    if T.ndim == 1:
        T = T[np.newaxis, :]

    P = P.copy().astype(float).T
    if P.ndim == 1:
        P = P[np.newaxis, :]

    m = T.shape[1] - P.shape[1] + 1

    if excl_zone is None:  # pragma: no cover
        excl_zone = int(np.ceil(m / 4))
    if atol is None:  # pragma: no cover
        atol = 0.25 * np.std(P)
    if max_occurrences is None:  # pragma: no cover
        max_occurrences = np.inf

    if M_T is None or Σ_T is None:  # pragma: no cover
        if not aamp:
            T, M_T, Σ_T = core.preprocess(T, m)

    result = _search(
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
        aamp=aamp,
    )
    return result


def _search(
    T, P, k, M_T, Σ_T, excl_zone, min_neighbors, max_occurrences, atol, rtol, aamp
):
    n = T.shape[1]
    l = P.shape[1]
    m = n - l + 1

    top_motif_indices = []
    top_motif_values = []

    # search in the highest dimensional matrix profile
    dim = P.shape[0] - 1

    candidate_idx = np.argmin(P[dim])
    while len(top_motif_indices) < k:
        profile_value = P[dim, candidate_idx]
        if np.isinf(profile_value):  # pragma: no cover
            break

        Q = T[:, candidate_idx : candidate_idx + m]

        occurrences = pattern(
            Q,
            T,
            M_T=M_T,
            Σ_T=Σ_T,
            excl_zone=excl_zone,
            max_occurrences=max_occurrences,
            profile_value=profile_value,
            atol=atol,
            rtol=rtol,
            aamp=aamp,
        )

        if len(occurrences) >= min_neighbors + 1:
            top_motif_indices.append(occurrences)
            top_motif_values.append(profile_value)

        for idx in occurrences:
            core.apply_exclusion_zone(P, idx, excl_zone)

        candidate_idx = np.argmin(P[dim])

    return top_motif_indices, top_motif_values


def pattern(
    Q,
    T,
    M_T=None,
    Σ_T=None,
    excl_zone=None,
    max_occurrences=None,
    profile_value=0.0,
    atol=None,
    rtol=1.0,
    aamp=False,
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

    excl_zone : int or None
        Size of the exclusion zone.
        If `None`, defaults to `m/4`, where `m` is the length of `Q`.

    max_occurrences : int or None
        The maximum amount of similar occurrences to be returned. The resulting
        occurrences are sorted by distance, so a value of `10` means that the
        indices of the most similar `10` subsequences is returned. If `None`, then all
        occurrences are returned.
        Defaults to `None`

    profile_value : float
        Reference value for relative tolerance (if `Q` is a subsequence of `T`,
        this will typically be Qs matrix profile value).
        Defaults to `0.0`

    atol : float or None
        Absolute tolerance (see equation in description).
        If `None`, defaults to `0.0`

    rtol : float
        Relative tolerance (see equation in description).
        Defaults to `1.0`

    Returns
    -------
    occurrences : list
        The indices of subsequences of `T` whose distance to `Q` is smaller than
        `atol + rtol * profile_value`, sorted by distance (lowest to highest).
    """
    if len(Q.shape) == 1:
        Q = Q[np.newaxis, :]
    if len(T.shape) == 1:
        T = T[np.newaxis, :]

    d, n = T.shape
    m = Q.shape[1]
    k = n - m + 1

    if excl_zone is None:  # pragma: no cover
        excl_zone = int(np.ceil(m / 4))
    if atol is None:  # pragma: no cover
        atol = 0.0
    if max_occurrences is None:  # pragma: no cover
        max_occurrences = np.inf

    if np.any(np.isnan(Q)) or np.any(np.isinf(Q)):  # pragma: no cover
        raise ValueError("Q contains illegal values (NaN or inf)")

    if M_T is None or Σ_T is None:  # pragma: no cover
        if aamp:
            T, T_subseq_isfinite = core.preprocess_non_normalized(T, m)
        else:
            T, M_T, Σ_T = core.preprocess(T, m)

    D = np.empty((d, k), dtype=float)

    # Calculate the d-dimensional distance profile
    for i in range(d):
        if aamp:
            D[i] = core.mass_absolute(Q[i], T[i])
        else:
            D[i] = core.mass(Q[i], T[i], M_T[i], Σ_T[i])
    D = np.sum(D, axis=0) / d

    occurrences = []

    candidate_idx = np.argmin(D)
    while (
        D[candidate_idx] < atol + rtol * profile_value
        and len(occurrences) < max_occurrences
    ):
        occurrences.append(candidate_idx)
        core.apply_exclusion_zone(D, candidate_idx, excl_zone)
        candidate_idx = np.argmin(D)

    return occurrences
