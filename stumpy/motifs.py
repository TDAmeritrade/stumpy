import numpy as np

from . import core


def k_motifs(T, P, k=1, excl_zone=None, atol=1.0, rtol=0.0):
    """
    Find the top k motifs of the time series `T`.

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
        Number of motifs to search

    excl_zone : int (optional)
        Size of the exclusion zone (defaults to `m/4`, where `m` is the length of `Q`)

    atol : float (optional)
        Absolute tolerance (see equation in description)

    rtol : float (optional)
        Relative tolerance (see equation in description)

    Return
    ------
    top_motif_indices : list
        Each item is another list, with each item representing an occurrence of the
        motif in `T`, sorted by distance from the "original occurrence", i.e. the
        subsequence of `T` starting at the index in the first place of the list

    top_motif_values : list
        The matrix profile values of the "original occurrences"
    """

    if k < 1:
        raise ValueError("k (the number of motifs) has to be at least 1")

    if len(P.shape) == 1:
        P = P.copy()
    elif len(P.shape) == 2:
        P = P[:, 0].copy()
    else:
        raise ValueError("P has too many dimensions (max. 2)")

    P = P.astype(float)

    n = T.shape[0]
    l = P.shape[0]
    m = n - l + 1

    if excl_zone is None:
        excl_zone = int(np.ceil(m / 4))

    T, M_T, Σ_T = core.preprocess(T, m)

    top_motif_indices = []
    top_motif_values = []

    candidate_idx = np.argmin(P)
    while len(top_motif_indices) < k:
        profile_value = P[candidate_idx]
        if np.isinf(profile_value):
            break

        Q = T[candidate_idx : candidate_idx + m]

        occurrences = find_occurrences(
            Q,
            T,
            M_T=M_T,
            Σ_T=Σ_T,
            excl_zone=excl_zone,
            profile_value=profile_value,
            atol=atol,
            rtol=rtol,
        )
        for idx in occurrences:
            core.apply_exclusion_zone(P, idx, excl_zone)

        if len(occurrences) > 1:
            top_motif_indices.append(occurrences)
            top_motif_values.append(profile_value)

    return top_motif_indices, top_motif_values


def find_occurrences(
    Q, T, M_T=None, Σ_T=None, excl_zone=None, profile_value=0.0, atol=1.0, rtol=0.0
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

    M_T : ndarray (optional)
        Sliding mean of time series, `T`

    Σ_T : ndarray (optional)
        Sliding standard deviation of time series, `T`

    excl_zone : int (optional)
        Size of the exclusion zone (defaults to `m/4`, where `m` is the length of `Q`)

    profile_value : float (optional)
        Reference value for relative tolerance (if `Q` is a subsequence of `T`,
        this will typically be Qs matrix profile value)

    atol : float (optional)
        Absolute tolerance (see equation in description)

    rtol : float (optional)
        Relative tolerance (see equation in description)

    Returns
    -------
    occurrences : list
        The indices of subsequences of `T` whose distance to `Q` is smaller than
        `atol + rtol * profile_value`, sorted by distance (lowest to highest).
    """

    m = Q.shape[0]
    if np.any(np.isnan(Q)) or np.any(np.isinf(Q)):
        raise ValueError("Q contains illigal values (NaN or inf)")

    if excl_zone is None:
        excl_zone = int(np.ceil(m / 4))

    if M_T is None or Σ_T is None:
        T, M_T, Σ_T = core.preprocess(T, m)

    D = core.mass(Q, T, M_T, Σ_T)
    occurrences = []

    candidate_idx = np.argmin(D)
    while D[candidate_idx] < atol + rtol * profile_value:
        occurrences.append(candidate_idx)
        core.apply_exclusion_zone(D, candidate_idx, excl_zone)
        candidate_idx = np.argmin(D)

    return occurrences
