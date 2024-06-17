import math

import numpy as np
from scipy.spatial.distance import cdist
from scipy.stats import norm

from stumpy import config, core


def is_ptp_zero_1d(a, w):  # `a` is 1-D
    n = len(a) - w + 1
    out = np.empty(n)
    for i in range(n):
        out[i] = np.max(a[i : i + w]) - np.min(a[i : i + w])
    return out == 0


def rolling_isconstant(a, w, a_subseq_isconstant=None):
    # a_subseq_isconstant can be numpy.ndarray or function
    if a_subseq_isconstant is None:
        a_subseq_isconstant = is_ptp_zero_1d

    custom_func = None
    if callable(a_subseq_isconstant):
        custom_func = a_subseq_isconstant

    if custom_func is not None:
        a_subseq_isconstant = np.logical_and(
            core.rolling_isfinite(a, w),
            np.apply_along_axis(
                lambda a_row, w: custom_func(a_row, w), axis=-1, arr=a, w=w
            ),
        )

    return a_subseq_isconstant


def rolling_nanstd(a, w):
    # a can be 1D, 2D, or more. The rolling occurs on last axis.
    return np.nanstd(core.rolling_window(a, w), axis=a.ndim)


def z_norm(a, axis=0):
    std = np.std(a, axis, keepdims=True)
    std = np.where(std > 0, std, 1.0)

    return (a - np.mean(a, axis, keepdims=True)) / std


def distance(a, b, axis=0, p=2.0):
    return np.linalg.norm(a - b, axis=axis, ord=p)


def compute_mean_std(T, m):
    n = T.shape[0]

    M_T = np.zeros(n - m + 1, dtype=float)
    Σ_T = np.zeros(n - m + 1, dtype=float)

    for i in range(n - m + 1):
        Q = T[i : i + m].copy()
        Q[np.isinf(Q)] = np.nan

        M_T[i] = np.mean(Q)
        Σ_T[i] = np.nanstd(Q)

    M_T[np.isnan(M_T)] = np.inf
    Σ_T[np.isnan(Σ_T)] = 0
    return M_T, Σ_T


def apply_exclusion_zone(a, trivial_idx, excl_zone, val):
    start = max(0, trivial_idx - excl_zone)
    stop = min(a.shape[-1], trivial_idx + excl_zone + 1)
    for i in range(start, stop):
        a[..., i] = val


def distance_profile(Q, T, m):
    T_inf = np.isinf(T)
    if np.any(T_inf):
        T = T.copy()
        T[T_inf] = np.nan

    Q_inf = np.isinf(Q)
    if np.any(Q_inf):
        Q = Q.copy()
        Q[Q_inf] = np.nan

    D = np.linalg.norm(z_norm(core.rolling_window(T, m), 1) - z_norm(Q), axis=1)

    return D


def aamp_distance_profile(Q, T, m, p=2.0):
    T_inf = np.isinf(T)
    if np.any(T_inf):
        T = T.copy()
        T[T_inf] = np.nan

    Q_inf = np.isinf(Q)
    if np.any(Q_inf):
        Q = Q.copy()
        Q[Q_inf] = np.nan

    D = np.linalg.norm(core.rolling_window(T, m) - Q, axis=1, ord=p)

    return D


def distance_matrix(T_A, T_B, m):
    distance_matrix = np.array(
        [distance_profile(Q, T_B, m) for Q in core.rolling_window(T_A, m)]
    )

    return distance_matrix


def aamp_distance_matrix(T_A, T_B, m, p):
    T_A[np.isinf(T_A)] = np.nan
    T_B[np.isinf(T_B)] = np.nan

    rolling_T_A = core.rolling_window(T_A, m)
    rolling_T_B = core.rolling_window(T_B, m)

    distance_matrix = cdist(rolling_T_A, rolling_T_B, metric="minkowski", p=p)

    return distance_matrix


def mass_PI(
    Q,
    T,
    m,
    trivial_idx=None,
    excl_zone=0,
    ignore_trivial=False,
    T_subseq_isconstant=None,
    Q_subseq_isconstant=None,
):
    Q = np.asarray(Q)
    T = np.asarray(T)

    Q_subseq_isconstant = rolling_isconstant(Q, m, Q_subseq_isconstant)[0]
    T_subseq_isconstant = rolling_isconstant(T, m, T_subseq_isconstant)

    D = distance_profile(Q, T, m)
    D[np.isnan(D)] = np.inf
    for i in range(len(T) - m + 1):
        if np.isfinite(D[i]):
            if Q_subseq_isconstant and T_subseq_isconstant[i]:
                D[i] = 0
            elif Q_subseq_isconstant or T_subseq_isconstant[i]:
                D[i] = np.sqrt(m)
            else:  # pragma: no cover
                pass

    if ignore_trivial:
        apply_exclusion_zone(D, trivial_idx, excl_zone, np.inf)
        start = max(0, trivial_idx - excl_zone)
        stop = min(D.shape[0], trivial_idx + excl_zone + 1)

    I = np.argmin(D)
    P = D[I]

    if P == np.inf:
        I = -1

    # Get left and right matrix profiles for self-joins
    if ignore_trivial and trivial_idx > 0:
        PL = np.inf
        IL = -1
        for i in range(trivial_idx):
            if D[i] < PL:  # pragma: no cover
                IL = i
                PL = D[i]
        if start <= IL < stop:  # pragma: no cover
            IL = -1
    else:  # pragma: no cover
        IL = -1

    if ignore_trivial and trivial_idx + 1 < D.shape[0]:
        PR = np.inf
        IR = -1
        for i in range(trivial_idx + 1, D.shape[0]):
            if D[i] < PR:
                IR = i
                PR = D[i]
        if start <= IR < stop:  # pragma: no cover
            IR = -1
    else:  # pragma: no cover
        IR = -1

    return P, I, IL, IR


def stamp(T_A, m, T_B=None, exclusion_zone=None):  # pragma: no cover
    if T_B is None:  # self-join
        result = np.array(
            [
                mass_PI(Q, T_A, m, i, exclusion_zone, True)
                for i, Q in enumerate(core.rolling_window(T_A, m))
            ],
            dtype=object,
        )
    else:
        result = np.array(
            [mass_PI(Q, T_B, m) for Q in core.rolling_window(T_A, m)],
            dtype=object,
        )
    return result


def searchsorted_right(a, v):
    """
    Naive version of numpy.searchsorted(..., side='right')
    """
    indices = np.flatnonzero(v < a)
    if len(indices):
        return indices.min()
    else:  # pragma: no cover
        return len(a)


def stump(
    T_A,
    m,
    T_B=None,
    exclusion_zone=None,
    row_wise=False,
    k=1,
    T_A_subseq_isconstant=None,
    T_B_subseq_isconstant=None,
):
    """
    Traverse distance matrix diagonally and update the top-k matrix profile and
    matrix profile indices if the parameter `row_wise` is set to `False`. If the
    parameter `row_wise` is set to `True`, it is a row-wise traversal.
    """
    if T_B is None:  # self-join:
        ignore_trivial = True
        distance_matrix = np.array(
            [distance_profile(Q, T_A, m) for Q in core.rolling_window(T_A, m)]
        )
        T_B = T_A.copy()
        T_B_subseq_isconstant = T_A_subseq_isconstant
    else:
        ignore_trivial = False
        distance_matrix = np.array(
            [distance_profile(Q, T_B, m) for Q in core.rolling_window(T_A, m)]
        )

    T_A_subseq_isconstant = rolling_isconstant(T_A, m, T_A_subseq_isconstant)
    T_B_subseq_isconstant = rolling_isconstant(T_B, m, T_B_subseq_isconstant)

    distance_matrix[np.isnan(distance_matrix)] = np.inf
    for i in range(distance_matrix.shape[0]):
        for j in range(distance_matrix.shape[1]):
            if np.isfinite(distance_matrix[i, j]):
                if T_A_subseq_isconstant[i] and T_B_subseq_isconstant[j]:
                    distance_matrix[i, j] = 0.0
                elif T_A_subseq_isconstant[i] or T_B_subseq_isconstant[j]:
                    distance_matrix[i, j] = np.sqrt(m)
                else:  # pragma: no cover
                    pass

    n_A = T_A.shape[0]
    n_B = T_B.shape[0]
    l = n_A - m + 1
    if exclusion_zone is None:
        exclusion_zone = int(np.ceil(m / config.STUMPY_EXCL_ZONE_DENOM))

    P = np.full((l, k + 2), np.inf, dtype=np.float64)
    I = np.full((l, k + 2), -1, dtype=np.int64)  # two more columns are to store
    # ... left and right top-1 matrix profile indices

    if row_wise:  # row-wise traversal in distance matrix
        if ignore_trivial:  # self-join
            for i in range(l):
                apply_exclusion_zone(distance_matrix[i], i, exclusion_zone, np.inf)

        for i, D in enumerate(distance_matrix):  # D: distance profile
            # self-join / AB-join: matrix profile and indices
            indices = np.argsort(D, kind="mergesort")[:k]
            P[i, :k] = D[indices]
            indices[P[i, :k] == np.inf] = -1
            I[i, :k] = indices

            # self-join: left matrix profile index (top-1)
            if ignore_trivial and i > 0:
                IL = np.argmin(D[:i])
                if D[IL] == np.inf:
                    IL = -1
                I[i, k] = IL

            # self-join: right matrix profile index (top-1)
            if ignore_trivial and i < D.shape[0]:
                IR = i + np.argmin(D[i:])  # offset by `i` to get true index
                if D[IR] == np.inf:
                    IR = -1
                I[i, k + 1] = IR

    else:  # diagonal traversal
        if ignore_trivial:
            diags = np.arange(exclusion_zone + 1, n_A - m + 1)
        else:
            diags = np.arange(-(n_A - m + 1) + 1, n_B - m + 1)

        for g in diags:
            if g >= 0:
                iter_range = range(0, min(n_A - m + 1, n_B - m + 1 - g))
            else:
                iter_range = range(-g, min(n_A - m + 1, n_B - m + 1 - g))

            for i in iter_range:
                d = distance_matrix[i, i + g]
                if d < P[i, k - 1]:
                    idx = searchsorted_right(P[i], d)
                    # to keep the top-k, we must discard the last element.
                    P[i, :k] = np.insert(P[i, :k], idx, d)[:-1]
                    I[i, :k] = np.insert(I[i, :k], idx, i + g)[:-1]

                if ignore_trivial:  # Self-joins only
                    if d < P[i + g, k - 1]:
                        idx = searchsorted_right(P[i + g], d)
                        P[i + g, :k] = np.insert(P[i + g, :k], idx, d)[:-1]
                        I[i + g, :k] = np.insert(I[i + g, :k], idx, i)[:-1]

                    if i < i + g:
                        # Left matrix profile and left matrix profile index
                        if d < P[i + g, k]:
                            P[i + g, k] = d
                            I[i + g, k] = i

                        if d < P[i, k + 1]:
                            # right matrix profile and right matrix profile index
                            P[i, k + 1] = d
                            I[i, k + 1] = i + g

    result = np.empty((l, 2 * k + 2), dtype=object)
    result[:, :k] = P[:, :k]
    result[:, k:] = I[:, :]

    return result


def aamp(T_A, m, T_B=None, exclusion_zone=None, p=2.0, row_wise=False, k=1):
    """
    Traverse distance matrix diagonally and update the top-k matrix profile and
    matrix profile indices if the parameter `row_wise` is set to `False`. If the
    parameter `row_wise` is set to `True`, it is a row-wise traversal.
    """
    T_A = np.asarray(T_A)
    T_A = T_A.copy()

    if T_B is None:
        ignore_trivial = True
        T_B = T_A.copy()
    else:
        ignore_trivial = False
        T_B = np.asarray(T_B)
        T_B = T_B.copy()

    T_A[np.isinf(T_A)] = np.nan
    T_B[np.isinf(T_B)] = np.nan

    rolling_T_A = core.rolling_window(T_A, m)
    rolling_T_B = core.rolling_window(T_B, m)

    distance_matrix = cdist(rolling_T_A, rolling_T_B, metric="minkowski", p=p)

    n_A = T_A.shape[0]
    n_B = T_B.shape[0]
    l = n_A - m + 1
    if exclusion_zone is None:
        exclusion_zone = int(np.ceil(m / config.STUMPY_EXCL_ZONE_DENOM))

    P = np.full((l, k + 2), np.inf, dtype=np.float64)
    I = np.full((l, k + 2), -1, dtype=np.int64)  # two more columns are to store
    # ... left and right top-1 matrix profile indices

    if row_wise:
        if ignore_trivial:  # self-join
            for i in range(l):
                apply_exclusion_zone(distance_matrix[i], i, exclusion_zone, np.inf)

        for i, D in enumerate(distance_matrix):  # D: distance profile
            # self-join / AB-join: matrix profile and indices
            indices = np.argsort(D)[:k]
            P[i, :k] = D[indices]
            indices[P[i, :k] == np.inf] = -1
            I[i, :k] = indices

            # self-join: left matrix profile index (top-1)
            if ignore_trivial and i > 0:
                IL = np.argmin(D[:i])
                if D[IL] == np.inf:
                    IL = -1
                I[i, k] = IL

            # self-join: right matrix profile index (top-1)
            if ignore_trivial and i < D.shape[0]:
                IR = i + np.argmin(D[i:])  # offset by `i` to get true index
                if D[IR] == np.inf:
                    IR = -1
                I[i, k + 1] = IR

    else:
        if ignore_trivial:
            diags = np.arange(exclusion_zone + 1, n_A - m + 1)
        else:
            diags = np.arange(-(n_A - m + 1) + 1, n_B - m + 1)

        for g in diags:
            if g >= 0:
                iter_range = range(0, min(n_A - m + 1, n_B - m + 1 - g))
            else:
                iter_range = range(-g, min(n_A - m + 1, n_B - m + 1 - g))

            for i in iter_range:
                d = distance_matrix[i, i + g]
                if d < P[i, k - 1]:
                    idx = searchsorted_right(P[i], d)
                    # to keep the top-k, we must discard the last element.
                    P[i, :k] = np.insert(P[i, :k], idx, d)[:-1]
                    I[i, :k] = np.insert(I[i, :k], idx, i + g)[:-1]

                if ignore_trivial:  # Self-joins only
                    if d < P[i + g, k - 1]:
                        idx = searchsorted_right(P[i + g], d)
                        P[i + g, :k] = np.insert(P[i + g, :k], idx, d)[:-1]
                        I[i + g, :k] = np.insert(I[i + g, :k], idx, i)[:-1]

                    if i < i + g:
                        # Left matrix profile and left matrix profile index
                        if d < P[i + g, k]:
                            P[i + g, k] = d
                            I[i + g, k] = i

                        if d < P[i, k + 1]:
                            # right matrix profile and right matrix profile index
                            P[i, k + 1] = d
                            I[i, k + 1] = i + g

    result = np.empty((l, 2 * k + 2), dtype=object)
    result[:, :k] = P[:, :k]
    result[:, k:] = I[:, :]

    return result


def replace_inf(x, value=0):
    x[x == np.inf] = value
    x[x == -np.inf] = value
    return


def multi_mass(
    Q,
    T,
    m,
    include=None,
    discords=False,
    T_subseq_isconstant=None,
    Q_subseq_isconstant=None,
):
    T_inf = np.isinf(T)
    if np.any(T_inf):
        T = T.copy()
        T[T_inf] = np.nan

    Q_inf = np.isinf(Q)
    if np.any(Q_inf):
        Q = Q.copy()
        Q[Q_inf] = np.nan

    T_subseq_isconstant = rolling_isconstant(T, m, T_subseq_isconstant)
    Q_subseq_isconstant = rolling_isconstant(Q, m, Q_subseq_isconstant)

    d, n = T.shape
    D = np.empty((d, n - m + 1))
    for i in range(d):
        D[i] = distance_profile(Q[i], T[i], m)
        for j in range(len(D[i])):
            if np.isfinite(D[i, j]):
                if Q_subseq_isconstant[i] and T_subseq_isconstant[i, j]:
                    D[i, j] = 0
                elif Q_subseq_isconstant[i] or T_subseq_isconstant[i, j]:
                    D[i, j] = np.sqrt(m)
                else:  # pragma: no cover
                    pass

    D[np.isnan(D)] = np.inf

    return D


def multi_mass_absolute(Q, T, m, include=None, discords=False, p=2.0):
    T_inf = np.isinf(T)
    if np.any(T_inf):
        T = T.copy()
        T[T_inf] = np.nan

    Q_inf = np.isinf(Q)
    if np.any(Q_inf):
        Q = Q.copy()
        Q[Q_inf] = np.nan

    d, n = T.shape

    D = np.empty((d, n - m + 1))
    for i in range(d):
        D[i] = aamp_distance_profile(Q[i], T[i], m, p=p)

    D[np.isnan(D)] = np.inf

    return D


def PI(D, trivial_idx, excl_zone):
    d, k = D.shape

    P = np.full((d, k), np.inf)
    I = np.ones((d, k), dtype="int64") * -1

    for i in range(d):
        col_mask = P[i] > D[i]
        P[i, col_mask] = D[i, col_mask]
        I[i, col_mask] = trivial_idx

    return P, I


def apply_include(D, include):
    restricted_indices = []
    unrestricted_indices = []
    mask = np.ones(include.shape[0], bool)

    for i in range(include.shape[0]):
        if include[i] < include.shape[0]:
            restricted_indices.append(include[i])
        if include[i] >= include.shape[0]:
            unrestricted_indices.append(include[i])

    restricted_indices = np.array(restricted_indices, dtype=np.int64)
    unrestricted_indices = np.array(unrestricted_indices, dtype=np.int64)
    mask[restricted_indices] = False
    tmp_swap = D[: include.shape[0]].copy()

    D[: include.shape[0]] = D[include]
    D[unrestricted_indices] = tmp_swap[mask]


def multi_distance_profile(
    query_idx, T, m, include=None, discords=False, T_subseq_isconstant=None
):
    excl_zone = int(np.ceil(m / config.STUMPY_EXCL_ZONE_DENOM))

    d = T.shape[0]
    if T_subseq_isconstant is None or callable(T_subseq_isconstant):
        T_subseq_isconstant = [T_subseq_isconstant] * d

    T_subseq_isconstant = np.array(
        [rolling_isconstant(T[i], m, T_subseq_isconstant[i]) for i in range(d)]
    )

    Q = T[:, query_idx : query_idx + m]
    Q_subseq_isconstant = np.expand_dims(T_subseq_isconstant[:, query_idx], axis=1)
    D = multi_mass(
        Q,
        T,
        m,
        include,
        discords,
        T_subseq_isconstant=T_subseq_isconstant,
        Q_subseq_isconstant=Q_subseq_isconstant,
    )

    start_row_idx = 0
    if include is not None:
        apply_include(D, include)
        start_row_idx = include.shape[0]

    if discords:
        D[start_row_idx:][::-1].sort(axis=0)
    else:
        D[start_row_idx:].sort(axis=0)

    d, n = T.shape
    D_prime = np.zeros(n - m + 1)
    D_prime_prime = np.zeros((d, n - m + 1))
    for j in range(d):
        D_prime[:] = D_prime + D[j]
        D_prime_prime[j, :] = D_prime / (j + 1)

    apply_exclusion_zone(D_prime_prime, query_idx, excl_zone, np.inf)

    return D_prime_prime


def mstump(T, m, excl_zone, include=None, discords=False, T_subseq_isconstant=None):
    T = T.copy()

    d, n = T.shape
    k = n - m + 1

    if T_subseq_isconstant is None or callable(T_subseq_isconstant):
        T_subseq_isconstant = [T_subseq_isconstant] * d
    # else means T_subseq_isconstant is list or a numpy 2D array

    T_subseq_isconstant = np.array(
        [rolling_isconstant(T[i], m, T_subseq_isconstant[i]) for i in range(d)]
    )

    P = np.full((d, k), np.inf)
    I = np.ones((d, k), dtype="int64") * -1

    for i in range(k):
        D = multi_distance_profile(
            i, T, m, include, discords, T_subseq_isconstant=T_subseq_isconstant
        )
        P_i, I_i = PI(D, i, excl_zone)

        for dim in range(T.shape[0]):
            col_mask = P[dim] > P_i[dim]
            P[dim, col_mask] = P_i[dim, col_mask]
            I[dim, col_mask] = I_i[dim, col_mask]

    return P, I


def maamp_multi_distance_profile(query_idx, T, m, include=None, discords=False, p=2.0):
    excl_zone = int(np.ceil(m / config.STUMPY_EXCL_ZONE_DENOM))
    d, n = T.shape
    Q = T[:, query_idx : query_idx + m]
    D = multi_mass_absolute(Q, T, m, include, discords, p=p)

    start_row_idx = 0
    if include is not None:
        apply_include(D, include)
        start_row_idx = include.shape[0]

    if discords:
        D[start_row_idx:][::-1].sort(axis=0)
    else:
        D[start_row_idx:].sort(axis=0)

    D_prime = np.zeros(n - m + 1)
    D_prime_prime = np.zeros((d, n - m + 1))
    for j in range(d):
        D_prime[:] = D_prime + D[j]
        D_prime_prime[j, :] = D_prime / (j + 1)

    apply_exclusion_zone(D_prime_prime, query_idx, excl_zone, np.inf)

    return D_prime_prime


def maamp(T, m, excl_zone, include=None, discords=False, p=2.0):
    T = T.copy()

    d, n = T.shape
    k = n - m + 1

    P = np.full((d, k), np.inf)
    I = np.ones((d, k), dtype="int64") * -1

    for i in range(k):
        D = maamp_multi_distance_profile(i, T, m, include, discords, p=p)
        P_i, I_i = PI(D, i, excl_zone)

        for dim in range(T.shape[0]):
            col_mask = P[dim] > P_i[dim]
            P[dim, col_mask] = P_i[dim, col_mask]
            I[dim, col_mask] = I_i[dim, col_mask]

    return P, I


def subspace(T, m, subseq_idx, nn_idx, k, include=None, discords=False):
    n_bit = 8
    bins = norm.ppf(np.arange(1, (2**n_bit)) / (2**n_bit))

    subseqs = core.z_norm(T[:, subseq_idx : subseq_idx + m], axis=1)
    neighbors = core.z_norm(T[:, nn_idx : nn_idx + m], axis=1)

    disc_subseqs = np.searchsorted(bins, subseqs)
    disc_neighbors = np.searchsorted(bins, neighbors)

    D = distance(
        disc_subseqs,
        disc_neighbors,
        axis=1,
    )

    if discords:
        sorted_idx = D[::-1].argsort(axis=0, kind="mergesort")
    else:
        sorted_idx = D.argsort(axis=0, kind="mergesort")

    # `include` processing can occur since we are dealing with indices, not distances
    if include is not None:
        include_idx = []
        for i in range(include.shape[0]):
            include_idx.append(np.isin(sorted_idx, include[i]).nonzero()[0])
        include_idx = np.array(include_idx).flatten()
        include_idx.sort()
        exclude_idx = np.ones(T.shape[0], dtype=bool)
        exclude_idx[include_idx] = False
        exclude_idx = exclude_idx.nonzero()[0]
        sorted_idx[: include_idx.shape[0]], sorted_idx[include_idx.shape[0] :] = (
            sorted_idx[include_idx],
            sorted_idx[exclude_idx],
        )

    S = sorted_idx[: k + 1]

    return S


def maamp_subspace(T, m, subseq_idx, nn_idx, k, include=None, discords=False, p=2.0):
    n_bit = 8
    T_isfinite = np.isfinite(T)
    T_min = T[T_isfinite].min()
    T_max = T[T_isfinite].max()

    subseqs = T[:, subseq_idx : subseq_idx + m]
    neighbors = T[:, nn_idx : nn_idx + m]

    disc_subseqs = (
        np.round(((subseqs - T_min) / (T_max - T_min)) * ((2**n_bit) - 1.0)).astype(
            np.int64
        )
        + 1
    )
    disc_neighbors = (
        np.round(((neighbors - T_min) / (T_max - T_min)) * ((2**n_bit) - 1.0)).astype(
            np.int64
        )
        + 1
    )

    D = distance(
        disc_subseqs,
        disc_neighbors,
        axis=1,
        p=p,
    )

    if discords:
        sorted_idx = D[::-1].argsort(axis=0, kind="mergesort")
    else:
        sorted_idx = D.argsort(axis=0, kind="mergesort")

    # `include` processing can occur since we are dealing with indices, not distances
    if include is not None:
        include_idx = []
        for i in range(include.shape[0]):
            include_idx.append(np.isin(sorted_idx, include[i]).nonzero()[0])
        include_idx = np.array(include_idx).flatten()
        include_idx.sort()
        exclude_idx = np.ones(T.shape[0], dtype=bool)
        exclude_idx[include_idx] = False
        exclude_idx = exclude_idx.nonzero()[0]
        sorted_idx[: include_idx.shape[0]], sorted_idx[include_idx.shape[0] :] = (
            sorted_idx[include_idx],
            sorted_idx[exclude_idx],
        )

    S = sorted_idx[: k + 1]

    return S


def mdl(
    T,
    m,
    subseq_idx,
    nn_idx,
    include=None,
    discords=False,
    discretize_func=None,
    n_bit=8,
):
    ndim = T.shape[0]
    bins = norm.ppf(np.arange(1, (2**n_bit)) / (2**n_bit))
    bit_sizes = np.empty(T.shape[0])
    S = [None] * T.shape[0]
    for k in range(T.shape[0]):
        subseqs = core.z_norm(T[:, subseq_idx[k] : subseq_idx[k] + m], axis=1)
        neighbors = core.z_norm(T[:, nn_idx[k] : nn_idx[k] + m], axis=1)

        disc_subseqs = np.searchsorted(bins, subseqs)
        disc_neighbors = np.searchsorted(bins, neighbors)

        S[k] = subspace(T, m, subseq_idx[k], nn_idx[k], k, include, discords)

        n_val = len(set((disc_subseqs[S[k]] - disc_neighbors[S[k]]).flatten()))
        sub_dims = len(S[k])
        bit_sizes[k] = n_bit * (2 * ndim * m - sub_dims * m)
        bit_sizes[k] = bit_sizes[k] + sub_dims * m * np.log2(n_val) + n_val * n_bit

    return bit_sizes, S


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
    T_isfinite = np.isfinite(T)
    T_min = T[T_isfinite].min()
    T_max = T[T_isfinite].max()
    ndim = T.shape[0]

    bit_sizes = np.empty(T.shape[0])
    S = [None] * T.shape[0]
    for k in range(T.shape[0]):
        subseqs = T[:, subseq_idx[k] : subseq_idx[k] + m]
        neighbors = T[:, nn_idx[k] : nn_idx[k] + m]
        disc_subseqs = (
            np.round(
                ((subseqs - T_min) / (T_max - T_min)) * ((2**n_bit) - 1.0)
            ).astype(np.int64)
            + 1
        )
        disc_neighbors = (
            np.round(
                ((neighbors - T_min) / (T_max - T_min)) * ((2**n_bit) - 1.0)
            ).astype(np.int64)
            + 1
        )

        S[k] = maamp_subspace(T, m, subseq_idx[k], nn_idx[k], k, include, discords, p=p)
        sub_dims = len(S[k])
        n_val = len(set((disc_subseqs[S[k]] - disc_neighbors[S[k]]).flatten()))
        bit_sizes[k] = n_bit * (2 * ndim * m - sub_dims * m)
        bit_sizes[k] = bit_sizes[k] + sub_dims * m * np.log2(n_val) + n_val * n_bit

    return bit_sizes, S


def get_array_ranges(a, n_chunks, truncate):
    out = np.zeros((n_chunks, 2), np.int64)
    ranges_idx = 0
    range_start_idx = 0

    sum = 0
    for i in range(a.shape[0]):
        sum += a[i]
        if sum > a.sum() / n_chunks:
            out[ranges_idx, 0] = range_start_idx
            out[ranges_idx, 1] = min(i + 1, a.shape[0])  # Exclusive stop index
            # Reset and Update
            range_start_idx = i + 1
            ranges_idx += 1
            sum = 0
    # Handle final range outside of for loop
    out[ranges_idx, 0] = range_start_idx
    out[ranges_idx, 1] = a.shape[0]
    if ranges_idx < n_chunks - 1:
        out[ranges_idx:] = a.shape[0]

    if truncate:
        out = out[:ranges_idx]

    return out


class aampi_egress(object):
    def __init__(self, T, m, excl_zone=None, p=2.0, k=1, mp=None):
        self._T = np.asarray(T)
        self._T = self._T.copy()
        self._T_isfinite = np.isfinite(self._T)
        self._m = m
        self._p = p
        self._k = k

        self._excl_zone = excl_zone
        if self._excl_zone is None:
            self._excl_zone = int(np.ceil(self._m / config.STUMPY_EXCL_ZONE_DENOM))

        self._l = self._T.shape[0] - m + 1
        if mp is None:
            mp = aamp(self._T, self._m, exclusion_zone=self._excl_zone, p=p, k=self._k)
        else:
            mp = mp.copy()
        self._P = mp[:, :k].astype(np.float64)
        self._I = mp[:, k : 2 * k].astype(np.int64)

        self._left_I = mp[:, 2 * k].astype(np.int64)
        self._left_P = np.full_like(self._left_I, np.inf, dtype=np.float64)
        for idx, nn_idx in enumerate(self._left_I):
            if nn_idx >= 0:
                self._left_P[idx] = np.linalg.norm(
                    self._T[idx : idx + self._m] - self._T[nn_idx : nn_idx + self._m],
                    ord=self._p,
                )

        self._n_appended = 0

    def update(self, t):
        self._T[:] = np.roll(self._T, -1)
        self._T_isfinite[:] = np.roll(self._T_isfinite, -1)
        if np.isfinite(t):
            self._T_isfinite[-1] = True
            self._T[-1] = t
        else:
            self._T_isfinite[-1] = False
            self._T[-1] = 0
        self._n_appended += 1

        self._P = np.roll(self._P, -1, axis=0)
        self._I = np.roll(self._I, -1, axis=0)
        self._left_P[:] = np.roll(self._left_P, -1)
        self._left_I[:] = np.roll(self._left_I, -1)

        D = cdist(
            core.rolling_window(self._T[-self._m :], self._m),
            core.rolling_window(self._T, self._m),
            metric="minkowski",
            p=self._p,
        )[0]
        T_subseq_isfinite = np.all(
            core.rolling_window(self._T_isfinite, self._m), axis=1
        )
        D[~T_subseq_isfinite] = np.inf
        if np.any(~self._T_isfinite[-self._m :]):
            D[:] = np.inf

        apply_exclusion_zone(D, D.shape[0] - 1, self._excl_zone, np.inf)
        for j in range(D.shape[0]):
            if D[j] < self._P[j, -1]:
                pos = np.searchsorted(self._P[j], D[j], side="right")
                self._P[j] = np.insert(self._P[j], pos, D[j])[:-1]
                self._I[j] = np.insert(
                    self._I[j], pos, D.shape[0] - 1 + self._n_appended
                )[:-1]

        # update top-k for the last, newly-updated index
        I_last_topk = np.argsort(D, kind="mergesort")[: self._k]
        self._P[-1] = D[I_last_topk]
        self._I[-1] = I_last_topk + self._n_appended
        self._I[-1][self._P[-1] == np.inf] = -1

        # for the last index, the left matrix profile value is self.P_[-1, 0]
        # and the same goes for the left matrix profile index
        self._left_P[-1] = self._P[-1, 0]
        self._left_I[-1] = self._I[-1, 0]

    @property
    def P_(self):
        if self._k == 1:
            return self._P.flatten().astype(np.float64)
        else:
            return self._P.astype(np.float64)

    @property
    def I_(self):
        if self._k == 1:
            return self._I.flatten().astype(np.int64)
        else:
            return self._I.astype(np.int64)

    @property
    def left_P_(self):
        return self._left_P.astype(np.float64)

    @property
    def left_I_(self):
        return self._left_I.astype(np.int64)


class stumpi_egress(object):
    def __init__(
        self, T, m, excl_zone=None, k=1, mp=None, T_subseq_isconstant_func=None
    ):
        self._T = np.asarray(T)
        self._T = self._T.copy()
        self._T_isfinite = np.isfinite(self._T)
        self._m = m
        self._k = k
        if T_subseq_isconstant_func is None:
            T_subseq_isconstant_func = core._rolling_isconstant
        self._T_subseq_isconstant_func = T_subseq_isconstant_func
        self._T_subseq_isconstant = rolling_isconstant(
            self._T, self._m, self._T_subseq_isconstant_func
        )

        self._excl_zone = excl_zone
        if self._excl_zone is None:
            self._excl_zone = int(np.ceil(self._m / config.STUMPY_EXCL_ZONE_DENOM))

        self._l = self._T.shape[0] - m + 1

        if mp is None:
            mp = stump(
                self._T,
                self._m,
                exclusion_zone=self._excl_zone,
                k=self._k,
                T_A_subseq_isconstant=self._T_subseq_isconstant,
            )
        else:
            mp = mp.copy()

        self._P = mp[:, :k].astype(np.float64)
        self._I = mp[:, k : 2 * k].astype(np.int64)

        self._left_I = mp[:, 2 * k].astype(np.int64)
        self._left_P = np.full_like(self._left_I, np.inf, dtype=np.float64)

        for idx, nn_idx in enumerate(self._left_I):
            if nn_idx >= 0:
                if self._T_subseq_isconstant[idx] and self._T_subseq_isconstant[nn_idx]:
                    self._left_P[idx] = 0
                elif (
                    self._T_subseq_isconstant[idx] or self._T_subseq_isconstant[nn_idx]
                ):
                    self._left_P[idx] = np.sqrt(self._m)
                else:
                    self._left_P[idx] = distance_profile(
                        self._T[idx : idx + self._m],
                        self._T[nn_idx : nn_idx + self._m],
                        m,
                    )[0]

        self._n_appended = 0

    def update(self, t):
        self._T[:] = np.roll(self._T, -1)
        self._T_isfinite[:] = np.roll(self._T_isfinite, -1)
        if np.isfinite(t):
            self._T_isfinite[-1] = True
            self._T[-1] = t
        else:
            self._T_isfinite[-1] = False
            self._T[-1] = 0

        self._T_subseq_isconstant[:] = np.roll(self._T_subseq_isconstant, -1)
        self._T_subseq_isconstant[-1] = rolling_isconstant(
            self._T[-self._m :], self._m, self._T_subseq_isconstant_func
        ) & np.all(self._T_isfinite[-self._m :])

        self._n_appended += 1

        self._P = np.roll(self._P, -1, axis=0)
        self._I = np.roll(self._I, -1, axis=0)
        self._left_P[:] = np.roll(self._left_P, -1)
        self._left_I[:] = np.roll(self._left_I, -1)

        D = core.mass(
            self._T[-self._m :],
            self._T,
            T_subseq_isconstant=self._T_subseq_isconstant,
            Q_subseq_isconstant=self._T_subseq_isconstant[[-1]],
        )
        T_subseq_isfinite = np.all(
            core.rolling_window(self._T_isfinite, self._m), axis=1
        )
        D[~T_subseq_isfinite] = np.inf
        if np.any(~self._T_isfinite[-self._m :]):
            D[:] = np.inf

        apply_exclusion_zone(D, D.shape[0] - 1, self._excl_zone, np.inf)
        # update top-k matrix profile using newly calculated distance profile `D`
        for j in range(D.shape[0]):
            if D[j] < self._P[j, -1]:
                pos = np.searchsorted(self._P[j], D[j], side="right")
                self._P[j] = np.insert(self._P[j], pos, D[j])[:-1]
                self._I[j] = np.insert(
                    self._I[j], pos, D.shape[0] - 1 + self._n_appended
                )[:-1]

        # update top-k for the last, newly-updated index
        I_last_topk = np.argsort(D, kind="mergesort")[: self._k]
        self._P[-1] = D[I_last_topk]
        self._I[-1] = I_last_topk + self._n_appended
        self._I[-1][self._P[-1] == np.inf] = -1

        # for the last index, the left matrix profile value is self.P_[-1, 0]
        # and the same goes for the left matrix profile index
        self._left_P[-1] = self._P[-1, 0]
        self._left_I[-1] = self._I[-1, 0]

    @property
    def P_(self):
        if self._k == 1:
            return self._P.flatten().astype(np.float64)
        else:
            return self._P.astype(np.float64)

    @property
    def I_(self):
        if self._k == 1:
            return self._I.flatten().astype(np.int64)
        else:
            return self._I.astype(np.int64)

    @property
    def left_P_(self):
        return self._left_P.astype(np.float64)

    @property
    def left_I_(self):
        return self._left_I.astype(np.int64)


def across_series_nearest_neighbors(Ts, Ts_idx, subseq_idx, m, Ts_subseq_isconstant):
    """
    For multiple time series find, per individual time series, the subsequences closest
    to a query.

    Parameters
    ----------
    Ts : list
        A list of time series for which to find the nearest neighbor subsequences that
        are closest to the query subsequence `Ts[Ts_idx][subseq_idx : subseq_idx + m]`

    Ts_idx : int
        The index of time series in `Ts` which contains the query subsequence
        `Ts[Ts_idx][subseq_idx : subseq_idx + m]`

    subseq_idx : int
        The subsequence index in the time series `Ts[Ts_idx]` that contains the query
        subsequence `Ts[Ts_idx][subseq_idx : subseq_idx + m]`

    m : int
        Subsequence window size

    Ts_subseq_isconstant : list
        A list of `T_subseq_isconstant`, where the i-th item corresponds to `Ts[i]`

    Returns
    -------
    nns_radii : ndarray
        Nearest neighbor radii to subsequences in `Ts` that are closest to the query
        `Ts[Ts_idx][subseq_idx : subseq_idx + m]`

    nns_subseq_idx : ndarray
        Nearest neighbor indices to subsequences in `Ts` that are closest to the query
        `Ts[Ts_idx][subseq_idx : subseq_idx + m]`
    """
    k = len(Ts)

    Q = Ts[Ts_idx][subseq_idx : subseq_idx + m]
    Q_subseq_isconstant = Ts_subseq_isconstant[Ts_idx][subseq_idx]

    nns_radii = np.zeros(k, dtype=np.float64)
    nns_subseq_idx = np.zeros(k, dtype=np.int64)

    for i in range(k):
        dist_profile = distance_profile(Q, Ts[i], len(Q))
        for j in range(len(dist_profile)):
            if np.isfinite(dist_profile[j]):
                if Q_subseq_isconstant and Ts_subseq_isconstant[i][j]:
                    dist_profile[j] = 0
                elif Q_subseq_isconstant or Ts_subseq_isconstant[i][j]:
                    dist_profile[j] = np.sqrt(m)
                else:  # pragma: no cover
                    pass

        nns_subseq_idx[i] = np.argmin(dist_profile)
        nns_radii[i] = dist_profile[nns_subseq_idx[i]]

    return nns_radii, nns_subseq_idx


def get_central_motif(
    Ts, bsf_radius, bsf_Ts_idx, bsf_subseq_idx, m, Ts_subseq_isconstant
):
    """
    Compare subsequences with the same radius and return the most central motif

    Parameters
    ----------
    Ts : list
        List of time series for which to find the most central motif

    bsf_radius : float
        Best radius found by a consensus search algorithm

    bsf_Ts_idx : int
        Index of time series in which `radius` was first found

    bsf_subseq_idx : int
        Start index of the subsequence in `Ts[Ts_idx]` that has radius `radius`

    m : int
        Window size

    Ts_subseq_isconstant : list
        A list of boolean arrays, each corresponds to a time series in `Ts`

    Returns
    -------
    bsf_radius : float
        The updated radius of the most central consensus motif

    bsf_Ts_idx : int
        The updated index of time series which contains the most central consensus motif

    bsf_subseq_idx : int
        The update subsequence index of most central consensus motif within the time
        series `bsf_Ts_idx` that contains it
    """
    bsf_nns_radii, bsf_nns_subseq_idx = across_series_nearest_neighbors(
        Ts, bsf_Ts_idx, bsf_subseq_idx, m, Ts_subseq_isconstant
    )
    bsf_nns_mean_radii = bsf_nns_radii.mean()

    candidate_nns_Ts_idx = np.flatnonzero(np.isclose(bsf_nns_radii, bsf_radius))
    candidate_nns_subseq_idx = bsf_nns_subseq_idx[candidate_nns_Ts_idx]

    for Ts_idx, subseq_idx in zip(candidate_nns_Ts_idx, candidate_nns_subseq_idx):
        candidate_nns_radii, _ = across_series_nearest_neighbors(
            Ts, Ts_idx, subseq_idx, m, Ts_subseq_isconstant
        )
        if (
            np.isclose(candidate_nns_radii.max(), bsf_radius)
            and candidate_nns_radii.mean() < bsf_nns_mean_radii
        ):
            bsf_Ts_idx = Ts_idx
            bsf_subseq_idx = subseq_idx
            bsf_nns_mean_radii = candidate_nns_radii.mean()

    return bsf_radius, bsf_Ts_idx, bsf_subseq_idx


def consensus_search(Ts, m, Ts_subseq_isconstant):
    """
    Brute force consensus motif from
    <https://www.cs.ucr.edu/~eamonn/consensus_Motif_ICDM_Long_version.pdf>

    See Table 1

    Note that there is a bug in the pseudocode at line 8 where `i` should be `j`.
    This implementation fixes it.
    """
    k = len(Ts)

    bsf_radius = np.inf
    bsf_Ts_idx = 0
    bsf_subseq_idx = 0

    for j in range(k):
        radii = np.zeros(len(Ts[j]) - m + 1)
        for i in range(k):
            if i != j:
                mp = stump(
                    Ts[j],
                    m,
                    Ts[i],
                    T_A_subseq_isconstant=Ts_subseq_isconstant[j],
                    T_B_subseq_isconstant=Ts_subseq_isconstant[i],
                )
                radii = np.maximum(radii, mp[:, 0])
        min_radius_idx = np.argmin(radii)
        min_radius = radii[min_radius_idx]
        if min_radius < bsf_radius:
            bsf_radius = min_radius
            bsf_Ts_idx = j
            bsf_subseq_idx = min_radius_idx

    return bsf_radius, bsf_Ts_idx, bsf_subseq_idx


def ostinato(Ts, m, Ts_subseq_isconstant=None):
    if Ts_subseq_isconstant is None:
        Ts_subseq_isconstant = [None] * len(Ts)

    Ts_subseq_isconstant = [
        rolling_isconstant(Ts[i], m, Ts_subseq_isconstant[i]) for i in range(len(Ts))
    ]

    bsf_radius, bsf_Ts_idx, bsf_subseq_idx = consensus_search(
        Ts, m, Ts_subseq_isconstant
    )
    radius, Ts_idx, subseq_idx = get_central_motif(
        Ts,
        bsf_radius,
        bsf_Ts_idx,
        bsf_subseq_idx,
        m,
        Ts_subseq_isconstant,
    )
    return radius, Ts_idx, subseq_idx


def aamp_across_series_nearest_neighbors(Ts, Ts_idx, subseq_idx, m, p=2.0):
    """
    For multiple time series find, per individual time series, the subsequences closest
    to a query.

    Parameters
    ----------
    Ts : list
        A list of time series for which to find the nearest neighbor subsequences that
        are closest to the query subsequence `Ts[Ts_idx][subseq_idx : subseq_idx + m]`

    Ts_idx : int
        The index of time series in `Ts` which contains the query subsequence
        `Ts[Ts_idx][subseq_idx : subseq_idx + m]`

    subseq_idx : int
        The subsequence index in the time series `Ts[Ts_idx]` that contains the query
        subsequence `Ts[Ts_idx][subseq_idx : subseq_idx + m]`

    m : int
        Subsequence window size

    p : float, default 2.0
        The p-norm to apply for computing the Minkowski distance.

    Returns
    -------
    nns_radii : ndarray
        Nearest neighbor radii to subsequences in `Ts` that are closest to the query
        `Ts[Ts_idx][subseq_idx : subseq_idx + m]`

    nns_subseq_idx : ndarray
        Nearest neighbor indices to subsequences in `Ts` that are closest to the query
        `Ts[Ts_idx][subseq_idx : subseq_idx + m]`
    """
    k = len(Ts)
    Q = Ts[Ts_idx][subseq_idx : subseq_idx + m]
    nns_radii = np.zeros(k, dtype=np.float64)
    nns_subseq_idx = np.zeros(k, dtype=np.int64)

    for i in range(k):
        dist_profile = aamp_distance_profile(Q, Ts[i], len(Q), p=p)
        nns_subseq_idx[i] = np.argmin(dist_profile)
        nns_radii[i] = dist_profile[nns_subseq_idx[i]]

    return nns_radii, nns_subseq_idx


def get_aamp_central_motif(Ts, bsf_radius, bsf_Ts_idx, bsf_subseq_idx, m, p=2.0):
    bsf_nns_radii, bsf_nns_subseq_idx = aamp_across_series_nearest_neighbors(
        Ts, bsf_Ts_idx, bsf_subseq_idx, m, p=p
    )
    bsf_nns_mean_radii = bsf_nns_radii.mean()

    candidate_nns_Ts_idx = np.flatnonzero(np.isclose(bsf_nns_radii, bsf_radius))
    candidate_nns_subseq_idx = bsf_nns_subseq_idx[candidate_nns_Ts_idx]

    for Ts_idx, subseq_idx in zip(candidate_nns_Ts_idx, candidate_nns_subseq_idx):
        candidate_nns_radii, _ = aamp_across_series_nearest_neighbors(
            Ts, Ts_idx, subseq_idx, m, p=p
        )
        if (
            np.isclose(candidate_nns_radii.max(), bsf_radius)
            and candidate_nns_radii.mean() < bsf_nns_mean_radii
        ):
            bsf_Ts_idx = Ts_idx
            bsf_subseq_idx = subseq_idx
            bsf_nns_mean_radii = candidate_nns_radii.mean()

    return bsf_radius, bsf_Ts_idx, bsf_subseq_idx


def aamp_consensus_search(Ts, m, p=2.0):
    k = len(Ts)

    bsf_radius = np.inf
    bsf_Ts_idx = 0
    bsf_subseq_idx = 0

    for j in range(k):
        radii = np.zeros(len(Ts[j]) - m + 1)
        for i in range(k):
            if i != j:
                mp = aamp(Ts[j], m, Ts[i], p=p)
                radii = np.maximum(radii, mp[:, 0])
        min_radius_idx = np.argmin(radii)
        min_radius = radii[min_radius_idx]
        if min_radius < bsf_radius:
            bsf_radius = min_radius
            bsf_Ts_idx = j
            bsf_subseq_idx = min_radius_idx

    return bsf_radius, bsf_Ts_idx, bsf_subseq_idx


def aamp_ostinato(Ts, m, p=2.0):
    bsf_radius, bsf_Ts_idx, bsf_subseq_idx = aamp_consensus_search(Ts, m, p=p)
    radius, Ts_idx, subseq_idx = get_aamp_central_motif(
        Ts, bsf_radius, bsf_Ts_idx, bsf_subseq_idx, m, p=p
    )
    return radius, Ts_idx, subseq_idx


def mpdist_vect(
    T_A,
    T_B,
    m,
    percentage=0.05,
    k=None,
    T_A_subseq_isconstant=None,
    T_B_subseq_isconstant=None,
):
    n_A = T_A.shape[0]
    n_B = T_B.shape[0]
    j = n_A - m + 1  # `k` is reserved for `P_ABBA` selection
    P_ABBA = np.empty(2 * j, dtype=np.float64)
    MPdist_vect = np.empty(n_B - n_A + 1)

    if k is None:
        percentage = min(percentage, 1.0)
        percentage = max(percentage, 0.0)
        k = min(math.ceil(percentage * (2 * n_A)), 2 * j - 1)

    k = min(int(k), P_ABBA.shape[0] - 1)

    T_A_subseq_isconstant = rolling_isconstant(T_A, m, T_A_subseq_isconstant)
    T_B_subseq_isconstant = rolling_isconstant(T_B, m, T_B_subseq_isconstant)
    for i in range(n_B - n_A + 1):
        P_ABBA[:j] = stump(
            T_A,
            m,
            T_B[i : i + n_A],
            T_A_subseq_isconstant=T_A_subseq_isconstant,
            T_B_subseq_isconstant=T_B_subseq_isconstant[i : i + n_A - m + 1],
        )[:, 0]
        P_ABBA[j:] = stump(
            T_B[i : i + n_A],
            m,
            T_A,
            T_A_subseq_isconstant=T_B_subseq_isconstant[i : i + n_A - m + 1],
            T_B_subseq_isconstant=T_A_subseq_isconstant,
        )[:, 0]
        P_ABBA.sort()
        MPdist_vect[i] = P_ABBA[min(k, P_ABBA.shape[0] - 1)]

    return MPdist_vect


def aampdist_vect(T_A, T_B, m, percentage=0.05, k=None, p=2.0):
    n_A = T_A.shape[0]
    n_B = T_B.shape[0]
    j = n_A - m + 1  # `k` is reserved for `P_ABBA` selection
    P_ABBA = np.empty(2 * j, dtype=np.float64)
    aaMPdist_vect = np.empty(n_B - n_A + 1)

    if k is None:
        percentage = min(percentage, 1.0)
        percentage = max(percentage, 0.0)
        k = min(math.ceil(percentage * (2 * n_A)), 2 * j - 1)

    k = min(int(k), P_ABBA.shape[0] - 1)

    for i in range(n_B - n_A + 1):
        P_ABBA[:j] = aamp(T_A, m, T_B[i : i + n_A], p=p)[:, 0]
        P_ABBA[j:] = aamp(T_B[i : i + n_A], m, T_A, p=p)[:, 0]
        P_ABBA.sort()
        aaMPdist_vect[i] = P_ABBA[k]

    return aaMPdist_vect


def mpdist(
    T_A,
    T_B,
    m,
    percentage=0.05,
    k=None,
    T_A_subseq_isconstant=None,
    T_B_subseq_isconstant=None,
):
    percentage = min(percentage, 1.0)
    percentage = max(percentage, 0.0)
    n_A = T_A.shape[0]
    n_B = T_B.shape[0]
    P_ABBA = np.empty(n_A - m + 1 + n_B - m + 1, dtype=np.float64)
    if k is not None:
        k = int(k)
    else:
        k = min(math.ceil(percentage * (n_A + n_B)), n_A - m + 1 + n_B - m + 1 - 1)

    P_ABBA[: n_A - m + 1] = stump(
        T_A,
        m,
        T_B,
        T_A_subseq_isconstant=T_A_subseq_isconstant,
        T_B_subseq_isconstant=T_B_subseq_isconstant,
    )[:, 0]
    P_ABBA[n_A - m + 1 :] = stump(
        T_B,
        m,
        T_A,
        T_A_subseq_isconstant=T_B_subseq_isconstant,
        T_B_subseq_isconstant=T_A_subseq_isconstant,
    )[:, 0]

    P_ABBA.sort()
    MPdist = P_ABBA[k]
    if ~np.isfinite(MPdist):  # pragma: no cover
        k = np.isfinite(P_ABBA[:k]).sum() - 1
        MPdist = P_ABBA[k]

    return MPdist


def aampdist(T_A, T_B, m, percentage=0.05, k=None, p=2.0):
    percentage = min(percentage, 1.0)
    percentage = max(percentage, 0.0)
    n_A = T_A.shape[0]
    n_B = T_B.shape[0]
    P_ABBA = np.empty(n_A - m + 1 + n_B - m + 1, dtype=np.float64)
    if k is not None:
        k = int(k)
    else:
        k = min(math.ceil(percentage * (n_A + n_B)), n_A - m + 1 + n_B - m + 1 - 1)

    P_ABBA[: n_A - m + 1] = aamp(T_A, m, T_B, p=p)[:, 0]
    P_ABBA[n_A - m + 1 :] = aamp(T_B, m, T_A, p=p)[:, 0]

    P_ABBA.sort()
    MPdist = P_ABBA[k]
    if ~np.isfinite(MPdist):  # pragma: no cover
        k = np.isfinite(P_ABBA[:k]).sum() - 1
        MPdist = P_ABBA[k]

    return MPdist


def get_all_mpdist_profiles(
    T,
    m,
    percentage=1.0,
    s=None,
    mpdist_percentage=0.05,
    mpdist_k=None,
    mpdist_T_subseq_isconstant=None,
):
    if s is not None:
        s = min(int(s), m)
    else:
        percentage = min(percentage, 1.0)
        percentage = max(percentage, 0.0)
        s = min(math.ceil(percentage * m), m)

    T_subseq_isconstant = rolling_isconstant(T, s, mpdist_T_subseq_isconstant)
    right_pad = 0
    n_contiguous_windows = int(T.shape[0] // m)
    if T.shape[0] % m != 0:
        right_pad = int(m * np.ceil(T.shape[0] / m) - T.shape[0])
        pad_width = (0, right_pad)
        T = np.pad(T, pad_width, mode="constant", constant_values=np.nan)
        T_subseq_isconstant = np.pad(
            T_subseq_isconstant, pad_width, mode="constant", constant_values=False
        )

    n_padded = T.shape[0]
    D = np.empty((n_contiguous_windows, n_padded - m + 1))

    # Iterate over non-overlapping subsequences, see Definition 3
    for i in range(n_contiguous_windows):
        start = i * m
        stop = (i + 1) * m
        S_i = T[start:stop]
        S_i_subseq_isconstant = T_subseq_isconstant[start : stop - s + 1]
        D[i, :] = mpdist_vect(
            S_i,
            T,
            s,
            percentage=mpdist_percentage,
            k=mpdist_k,
            T_A_subseq_isconstant=S_i_subseq_isconstant,
            T_B_subseq_isconstant=T_subseq_isconstant,
        )

    stop_idx = n_padded - m + 1 - right_pad
    D = D[:, :stop_idx]

    return D


def mpdist_snippets(
    T,
    m,
    k,
    percentage=1.0,
    s=None,
    mpdist_percentage=0.05,
    mpdist_k=None,
    mpdist_T_subseq_isconstant=None,
):
    D = get_all_mpdist_profiles(
        T,
        m,
        percentage,
        s,
        mpdist_percentage,
        mpdist_k,
        mpdist_T_subseq_isconstant=mpdist_T_subseq_isconstant,
    )

    snippets = np.empty((k, m))
    snippets_indices = np.empty(k, dtype=np.int64)
    snippets_profiles = np.empty((k, D.shape[-1]))
    snippets_fractions = np.empty(k)
    snippets_areas = np.empty(k)
    Q = np.inf
    indices = np.arange(0, D.shape[0] * m, m)
    snippets_regimes_list = []

    for snippet_idx in range(k):
        min_area = np.inf
        for i in range(D.shape[0]):
            profile_area = np.sum(np.minimum(D[i], Q))
            if min_area > profile_area:
                min_area = profile_area
                idx = i

        snippets[snippet_idx] = T[indices[idx] : indices[idx] + m]
        snippets_indices[snippet_idx] = indices[idx]
        snippets_profiles[snippet_idx] = D[idx]
        snippets_areas[snippet_idx] = np.sum(np.minimum(D[idx], Q))

        Q = np.minimum(D[idx], Q)

    total_min = np.min(snippets_profiles, axis=0)

    for i in range(k):
        mask = snippets_profiles[i] <= total_min
        snippets_fractions[i] = np.sum(mask) / total_min.shape[0]
        total_min = total_min - mask.astype(float)
        slices = _get_mask_slices(mask)
        snippets_regimes_list.append(slices)

    n_slices = []
    for regime in snippets_regimes_list:
        n_slices.append(regime.shape[0])

    snippets_regimes = np.empty((sum(n_slices), 3), dtype=np.int64)
    i = 0
    j = 0
    for n_slice in n_slices:
        for _ in range(n_slice):
            snippets_regimes[i, 0] = j
            i += 1
        j += 1

    i = 0
    for regimes in snippets_regimes_list:
        for regime in regimes:
            snippets_regimes[i, 1:] = regime
            i += 1

    return (
        snippets,
        snippets_indices,
        snippets_profiles,
        snippets_fractions,
        snippets_areas,
        snippets_regimes,
    )


def get_all_aampdist_profiles(
    T,
    m,
    percentage=1.0,
    s=None,
    mpdist_percentage=0.05,
    mpdist_k=None,
    p=2.0,
):
    right_pad = 0
    if T.shape[0] % m != 0:
        right_pad = int(m * np.ceil(T.shape[0] / m) - T.shape[0])
        pad_width = (0, right_pad)
        T = np.pad(T, pad_width, mode="constant", constant_values=np.nan)

    n_padded = T.shape[0]
    D = np.empty(((n_padded // m) - 1, n_padded - m + 1))

    if s is not None:
        s = min(int(s), m)
    else:
        percentage = min(percentage, 1.0)
        percentage = max(percentage, 0.0)
        s = min(math.ceil(percentage * m), m)

    # Iterate over non-overlapping subsequences, see Definition 3
    for i in range((n_padded // m) - 1):
        start = i * m
        stop = (i + 1) * m
        S_i = T[start:stop]
        D[i, :] = aampdist_vect(
            S_i,
            T,
            s,
            percentage=mpdist_percentage,
            k=mpdist_k,
            p=p,
        )

    stop_idx = n_padded - m + 1 - right_pad
    D = D[:, :stop_idx]

    return D


def aampdist_snippets(
    T,
    m,
    k,
    percentage=1.0,
    s=None,
    mpdist_percentage=0.05,
    mpdist_k=None,
    p=2.0,
):
    D = get_all_aampdist_profiles(T, m, percentage, s, mpdist_percentage, mpdist_k, p=p)

    pad_width = (0, int(m * np.ceil(T.shape[0] / m) - T.shape[0]))
    T_padded = np.pad(T, pad_width, mode="constant", constant_values=np.nan)
    n_padded = T_padded.shape[0]

    snippets = np.empty((k, m))
    snippets_indices = np.empty(k, dtype=np.int64)
    snippets_profiles = np.empty((k, D.shape[-1]))
    snippets_fractions = np.empty(k)
    snippets_areas = np.empty(k)
    Q = np.inf
    indices = np.arange(0, n_padded - m, m)
    snippets_regimes_list = []

    for snippet_idx in range(k):
        min_area = np.inf
        for i in range(D.shape[0]):
            profile_area = np.sum(np.minimum(D[i], Q))
            if min_area > profile_area:
                min_area = profile_area
                idx = i

        snippets[snippet_idx] = T[indices[idx] : indices[idx] + m]
        snippets_indices[snippet_idx] = indices[idx]
        snippets_profiles[snippet_idx] = D[idx]
        snippets_areas[snippet_idx] = np.sum(np.minimum(D[idx], Q))

        Q = np.minimum(D[idx], Q)

    total_min = np.min(snippets_profiles, axis=0)

    for i in range(k):
        mask = snippets_profiles[i] <= total_min
        snippets_fractions[i] = np.sum(mask) / total_min.shape[0]
        total_min = total_min - mask.astype(float)
        slices = _get_mask_slices(mask)
        snippets_regimes_list.append(slices)

    n_slices = []
    for regime in snippets_regimes_list:
        n_slices.append(regime.shape[0])

    snippets_regimes = np.empty((sum(n_slices), 3), dtype=np.int64)
    i = 0
    j = 0
    for n_slice in n_slices:
        for _ in range(n_slice):
            snippets_regimes[i, 0] = j
            i += 1
        j += 1

    i = 0
    for regimes in snippets_regimes_list:
        for regime in regimes:
            snippets_regimes[i, 1:] = regime
            i += 1

    return (
        snippets,
        snippets_indices,
        snippets_profiles,
        snippets_fractions,
        snippets_areas,
        snippets_regimes,
    )


def prescrump(
    T_A,
    m,
    T_B,
    s,
    exclusion_zone=None,
    k=1,
    T_A_subseq_isconstant=None,
    T_B_subseq_isconstant=None,
):
    T_A_subseq_isconstant = rolling_isconstant(T_A, m, T_A_subseq_isconstant)
    T_B_subseq_isconstant = rolling_isconstant(T_B, m, T_B_subseq_isconstant)

    dist_matrix = distance_matrix(T_A, T_B, m)
    dist_matrix[np.isnan(dist_matrix)] = np.inf
    for i in range(dist_matrix.shape[0]):
        for j in range(dist_matrix.shape[1]):
            if np.isfinite(dist_matrix[i, j]):
                if T_A_subseq_isconstant[i] and T_B_subseq_isconstant[j]:
                    dist_matrix[i, j] = 0.0
                elif T_A_subseq_isconstant[i] or T_B_subseq_isconstant[j]:
                    dist_matrix[i, j] = np.sqrt(m)
                else:  # pragma: no cover
                    pass

    l = T_A.shape[0] - m + 1  # matrix profile length
    w = T_B.shape[0] - m + 1  # distance profile length

    P = np.full((l, k), np.inf, dtype=np.float64)
    I = np.full((l, k), -1, dtype=np.int64)

    for i in np.random.permutation(range(0, l, s)):
        distance_profile = dist_matrix[i]
        if exclusion_zone is not None:
            apply_exclusion_zone(distance_profile, i, exclusion_zone, np.inf)

        nn_idx = np.argmin(distance_profile)
        if distance_profile[nn_idx] < P[i, -1] and nn_idx not in I[i]:
            pos = np.searchsorted(P[i], distance_profile[nn_idx], side="right")
            P[i] = np.insert(P[i], pos, distance_profile[nn_idx])[:-1]
            I[i] = np.insert(I[i], pos, nn_idx)[:-1]

        if P[i, 0] == np.inf:
            I[i, 0] = -1
            continue

        j = nn_idx
        for g in range(1, min(s, l - i, w - j)):
            d = dist_matrix[i + g, j + g]
            # Do NOT optimize the `condition` in the following if statement
            # and similar ones in this naive function. This is to ensure
            # we are avoiding duplicates in each row of I.
            if d < P[i + g, -1] and (j + g) not in I[i + g]:
                pos = np.searchsorted(P[i + g], d, side="right")
                P[i + g] = np.insert(P[i + g], pos, d)[:-1]
                I[i + g] = np.insert(I[i + g], pos, j + g)[:-1]
            if (
                exclusion_zone is not None
                and d < P[j + g, -1]
                and (i + g) not in I[j + g]
            ):
                pos = np.searchsorted(P[j + g], d, side="right")
                P[j + g] = np.insert(P[j + g], pos, d)[:-1]
                I[j + g] = np.insert(I[j + g], pos, i + g)[:-1]

        for g in range(1, min(s, i + 1, j + 1)):
            d = dist_matrix[i - g, j - g]
            if d < P[i - g, -1] and (j - g) not in I[i - g]:
                pos = np.searchsorted(P[i - g], d, side="right")
                P[i - g] = np.insert(P[i - g], pos, d)[:-1]
                I[i - g] = np.insert(I[i - g], pos, j - g)[:-1]
            if (
                exclusion_zone is not None
                and d < P[j - g, -1]
                and (i - g) not in I[j - g]
            ):
                pos = np.searchsorted(P[j - g], d, side="right")
                P[j - g] = np.insert(P[j - g], pos, d)[:-1]
                I[j - g] = np.insert(I[j - g], pos, i - g)[:-1]

        # In the case of a self-join, the calculated distance profile can also be
        # used to refine the top-k for all non-trivial subsequences
        if exclusion_zone is not None:
            for idx in np.flatnonzero(distance_profile < P[:, -1]):
                if i not in I[idx]:
                    pos = np.searchsorted(P[idx], distance_profile[idx], side="right")
                    P[idx] = np.insert(P[idx], pos, distance_profile[idx])[:-1]
                    I[idx] = np.insert(I[idx], pos, i)[:-1]

    if k == 1:
        P = P.flatten()
        I = I.flatten()

    return P, I


def scrump(
    T_A,
    m,
    T_B,
    percentage,
    exclusion_zone,
    pre_scrump,
    s,
    k=1,
    T_A_subseq_isconstant=None,
    T_B_subseq_isconstant=None,
):
    n_A = T_A.shape[0]
    n_B = T_B.shape[0]
    l = n_A - m + 1

    T_A_subseq_isconstant = rolling_isconstant(T_A, m, T_A_subseq_isconstant)
    T_B_subseq_isconstant = rolling_isconstant(T_B, m, T_B_subseq_isconstant)
    dist_matrix = distance_matrix(T_A, T_B, m)
    dist_matrix[np.isnan(dist_matrix)] = np.inf
    for i in range(n_A - m + 1):
        for j in range(n_B - m + 1):
            if np.isfinite(dist_matrix[i, j]):
                if T_A_subseq_isconstant[i] and T_B_subseq_isconstant[j]:
                    dist_matrix[i, j] = 0
                elif T_A_subseq_isconstant[i] or T_B_subseq_isconstant[j]:
                    dist_matrix[i, j] = np.sqrt(m)
                else:  # pragma: no cover
                    pass

    if exclusion_zone is not None:
        diags = np.random.permutation(range(exclusion_zone + 1, n_A - m + 1)).astype(
            np.int64
        )
    else:
        diags = np.random.permutation(range(-(n_A - m + 1) + 1, n_B - m + 1)).astype(
            np.int64
        )

    n_chunks = int(np.ceil(1.0 / percentage))
    ndist_counts = core._count_diagonal_ndist(diags, m, n_A, n_B)
    diags_ranges = core._get_array_ranges(ndist_counts, n_chunks, False)
    diags_ranges_start = diags_ranges[0, 0]
    diags_ranges_stop = diags_ranges[0, 1]

    P = np.full((l, k), np.inf, dtype=np.float64)  # Topk
    PL = np.full(l, np.inf, dtype=np.float64)
    PR = np.full(l, np.inf, dtype=np.float64)

    I = np.full((l, k), -1, dtype=np.int64)
    IL = np.full(l, -1, dtype=np.int64)
    IR = np.full(l, -1, dtype=np.int64)

    for diag_idx in range(diags_ranges_start, diags_ranges_stop):
        g = diags[diag_idx]

        for i in range(n_A - m + 1):
            for j in range(n_B - m + 1):
                if j - i == g:
                    d = dist_matrix[i, j]
                    if d < P[i, -1]:  # update TopK of P[i]
                        idx = searchsorted_right(P[i], d)
                        if (i + g) not in I[i]:
                            P[i] = np.insert(P[i], idx, d)[:-1]
                            I[i] = np.insert(I[i], idx, i + g)[:-1]

                    if exclusion_zone is not None and d < P[i + g, -1]:
                        idx = searchsorted_right(P[i + g], d)
                        if i not in I[i + g]:
                            P[i + g] = np.insert(P[i + g], idx, d)[:-1]
                            I[i + g] = np.insert(I[i + g], idx, i)[:-1]

                    # left matrix profile and left matrix profile indices
                    if exclusion_zone is not None and i < i + g and d < PL[i + g]:
                        PL[i + g] = d
                        IL[i + g] = i

                    # right matrix profile and right matrix profile indices
                    if exclusion_zone is not None and i + g > i and d < PR[i]:
                        PR[i] = d
                        IR[i] = i + g

    if k == 1:
        P = P.flatten()
        I = I.flatten()

    return P, I, IL, IR


def prescraamp(T_A, m, T_B, s, exclusion_zone=None, p=2.0, k=1):
    distance_matrix = aamp_distance_matrix(T_A, T_B, m, p)

    l = T_A.shape[0] - m + 1  # matrix profile length
    w = T_B.shape[0] - m + 1  # distance profile length

    P = np.full((l, k), np.inf, dtype=np.float64)
    I = np.full((l, k), -1, dtype=np.int64)

    for i in np.random.permutation(range(0, l, s)):
        distance_profile = distance_matrix[i]
        if exclusion_zone is not None:
            apply_exclusion_zone(distance_profile, i, exclusion_zone, np.inf)

        nn_idx = np.argmin(distance_profile)
        if distance_profile[nn_idx] < P[i, -1] and nn_idx not in I[i]:
            pos = np.searchsorted(P[i], distance_profile[nn_idx], side="right")
            P[i] = np.insert(P[i], pos, distance_profile[nn_idx])[:-1]
            I[i] = np.insert(I[i], pos, nn_idx)[:-1]

        if P[i, 0] == np.inf:
            I[i, 0] = -1
            continue

        j = nn_idx
        for g in range(1, min(s, l - i, w - j)):
            d = distance_matrix[i + g, j + g]
            # Do NOT optimize the `condition` in the following if statement
            # and similar ones in this naive function. This is to ensure
            # we are avoiding duplicates in each row of I.
            if d < P[i + g, -1] and (j + g) not in I[i + g]:
                pos = np.searchsorted(P[i + g], d, side="right")
                P[i + g] = np.insert(P[i + g], pos, d)[:-1]
                I[i + g] = np.insert(I[i + g], pos, j + g)[:-1]
            if (
                exclusion_zone is not None
                and d < P[j + g, -1]
                and (i + g) not in I[j + g]
            ):
                pos = np.searchsorted(P[j + g], d, side="right")
                P[j + g] = np.insert(P[j + g], pos, d)[:-1]
                I[j + g] = np.insert(I[j + g], pos, i + g)[:-1]

        for g in range(1, min(s, i + 1, j + 1)):
            d = distance_matrix[i - g, j - g]
            if d < P[i - g, -1] and (j - g) not in I[i - g]:
                pos = np.searchsorted(P[i - g], d, side="right")
                P[i - g] = np.insert(P[i - g], pos, d)[:-1]
                I[i - g] = np.insert(I[i - g], pos, j - g)[:-1]
            if (
                exclusion_zone is not None
                and d < P[j - g, -1]
                and (i - g) not in I[j - g]
            ):
                pos = np.searchsorted(P[j - g], d, side="right")
                P[j - g] = np.insert(P[j - g], pos, d)[:-1]
                I[j - g] = np.insert(I[j - g], pos, i - g)[:-1]

        # In the case of a self-join, the calculated distance profile can also be
        # used to refine the top-k for all non-trivial subsequences
        if exclusion_zone is not None:
            for idx in np.flatnonzero(distance_profile < P[:, -1]):
                if i not in I[idx]:
                    pos = np.searchsorted(P[idx], distance_profile[idx], side="right")
                    P[idx] = np.insert(P[idx], pos, distance_profile[idx])[:-1]
                    I[idx] = np.insert(I[idx], pos, i)[:-1]

    if k == 1:
        P = P.flatten()
        I = I.flatten()

    return P, I


def scraamp(T_A, m, T_B, percentage, exclusion_zone, pre_scraamp, s, p=2.0, k=1):
    distance_matrix = aamp_distance_matrix(T_A, T_B, m, p)

    n_A = T_A.shape[0]
    n_B = T_B.shape[0]
    l = n_A - m + 1

    if exclusion_zone is not None:
        diags = np.random.permutation(range(exclusion_zone + 1, n_A - m + 1)).astype(
            np.int64
        )
    else:
        diags = np.random.permutation(range(-(n_A - m + 1) + 1, n_B - m + 1)).astype(
            np.int64
        )

    n_chunks = int(np.ceil(1.0 / percentage))
    ndist_counts = core._count_diagonal_ndist(diags, m, n_A, n_B)
    diags_ranges = core._get_array_ranges(ndist_counts, n_chunks, False)
    diags_ranges_start = diags_ranges[0, 0]
    diags_ranges_stop = diags_ranges[0, 1]

    P = np.full((l, k), np.inf, dtype=np.float64)  # Topk
    PL = np.full(l, np.inf, dtype=np.float64)
    PR = np.full(l, np.inf, dtype=np.float64)

    I = np.full((l, k), -1, dtype=np.int64)
    IL = np.full(l, -1, dtype=np.int64)
    IR = np.full(l, -1, dtype=np.int64)

    for diag_idx in range(diags_ranges_start, diags_ranges_stop):
        g = diags[diag_idx]

        for i in range(n_A - m + 1):
            for j in range(n_B - m + 1):
                if j - i == g:
                    d = distance_matrix[i, j]
                    if d < P[i, -1]:
                        idx = searchsorted_right(P[i], d)
                        if (i + g) not in I[i]:
                            P[i] = np.insert(P[i], idx, d)[:-1]
                            I[i] = np.insert(I[i], idx, i + g)[:-1]

                    if exclusion_zone is not None and d < P[i + g, -1]:
                        idx = searchsorted_right(P[i + g], d)
                        if i not in I[i + g]:
                            P[i + g] = np.insert(P[i + g], idx, d)[:-1]
                            I[i + g] = np.insert(I[i + g], idx, i)[:-1]

                    # left matrix profile and left matrix profile indices
                    if exclusion_zone is not None and i < i + g and d < PL[i + g]:
                        PL[i + g] = d
                        IL[i + g] = i

                    # right matrix profile and right matrix profile indices
                    if exclusion_zone is not None and i + g > i and d < PR[i]:
                        PR[i] = d
                        IR[i] = i + g

    if k == 1:
        P = P.flatten()
        I = I.flatten()

    return P, I, IL, IR


def normalize_pan(pan, ms, bfs_indices, n_processed, T_min=None, T_max=None, p=2.0):
    idx = bfs_indices[:n_processed]
    for i in range(n_processed):
        if T_min is not None and T_max is not None:
            norm = 1.0 / (np.abs(T_max - T_min) * np.power(ms[i], 1.0 / p))
        else:
            norm = 1.0 / (2.0 * np.sqrt(ms[i]))
        pan[idx[i]] = np.minimum(1.0, pan[idx[i]] * norm)


def contrast_pan(pan, threshold, bfs_indices, n_processed):
    idx = bfs_indices[:n_processed]
    l = n_processed * pan.shape[1]
    tmp = pan[idx].argsort(kind="mergesort", axis=None)
    ranks = np.empty(l, dtype=np.int64)
    for i in range(l):
        ranks[tmp[i]] = i

    percentile = np.full(ranks.shape, np.nan)
    percentile[:l] = np.linspace(0, 1, l)
    percentile = percentile[ranks].reshape(pan[idx].shape)
    for i in range(percentile.shape[0]):
        pan[idx[i]] = 1.0 / (1.0 + np.exp(-10 * (percentile[i] - threshold)))


def binarize_pan(pan, threshold, bfs_indices, n_processed):
    idx = bfs_indices[:n_processed]
    for i in range(idx.shape[0]):
        mask = pan[idx[i]] <= threshold
        pan[idx[i], mask] = 0.0
        mask = pan[idx[i]] > threshold
        pan[idx[i], mask] = 1.0


def transform_pan(
    pan, ms, threshold, bfs_indices, n_processed, T_min=None, T_max=None, p=2.0
):
    pan = pan.copy()
    idx = bfs_indices[:n_processed]
    sorted_idx = np.sort(idx)
    pan[pan == np.inf] = np.nan
    normalize_pan(pan, ms, bfs_indices, n_processed, T_min, T_max, p)
    contrast_pan(pan, threshold, bfs_indices, n_processed)
    binarize_pan(pan, threshold, bfs_indices, n_processed)

    pan[idx] = np.clip(pan[idx], 0.0, 1.0)

    nrepeat = np.diff(np.append(-1, sorted_idx))
    pan[: np.sum(nrepeat)] = np.repeat(pan[sorted_idx], nrepeat, axis=0)
    pan[np.isnan(pan)] = np.nanmax(pan)

    return pan


def _get_mask_slices(mask):
    idx = []

    tmp = np.r_[0, mask]
    for i, val in enumerate(np.diff(tmp)):
        if val == 1:
            idx.append(i)
        if val == -1:
            idx.append(i)

    if tmp[-1]:
        idx.append(len(mask))

    return np.array(idx).reshape(len(idx) // 2, 2)


def _total_trapezoid_ndists(a, b, h):
    return (a + b) * h // 2


def _total_diagonal_ndists(tile_lower_diag, tile_upper_diag, tile_height, tile_width):
    total_ndists = 0

    if tile_width < tile_height:
        # Transpose inputs, adjust for inclusive/exclusive diags
        tile_width, tile_height = tile_height, tile_width
        tile_lower_diag, tile_upper_diag = 1 - tile_upper_diag, 1 - tile_lower_diag

    if tile_lower_diag > tile_upper_diag:  # pragma: no cover
        # Swap diags
        tile_lower_diag, tile_upper_diag = tile_upper_diag, tile_lower_diag

    min_tile_diag = 1 - tile_height
    max_tile_diag = tile_width  # Exclusive

    if (
        tile_lower_diag < min_tile_diag
        or tile_upper_diag < min_tile_diag
        or tile_lower_diag > max_tile_diag
        or tile_upper_diag > max_tile_diag
    ):
        return total_ndists

    if tile_lower_diag == min_tile_diag and tile_upper_diag == max_tile_diag:
        total_ndists = tile_height * tile_width
    elif min_tile_diag <= tile_lower_diag < 0:
        lower_ndists = tile_height + tile_lower_diag
        if min_tile_diag <= tile_upper_diag <= 0:
            upper_ndists = tile_height + (tile_upper_diag - 1)
            total_ndists = _total_trapezoid_ndists(
                upper_ndists, lower_ndists, tile_upper_diag - tile_lower_diag
            )
        elif 0 < tile_upper_diag <= tile_width - tile_height + 1:
            total_ndists = _total_trapezoid_ndists(
                tile_height, lower_ndists, 1 - tile_lower_diag
            )
            total_ndists += (tile_upper_diag - 1) * tile_height
        else:  # tile_upper_diag > tile_width - tile_height + 1
            upper_ndists = tile_width - (tile_upper_diag - 1)
            total_ndists = _total_trapezoid_ndists(
                tile_height, lower_ndists, 1 - tile_lower_diag
            )
            total_ndists += (tile_width - tile_height) * tile_height
            total_ndists += _total_trapezoid_ndists(
                tile_height - 1,
                upper_ndists,
                tile_upper_diag - (tile_width - tile_height + 1),
            )
    elif 0 <= tile_lower_diag <= tile_width - tile_height:
        if tile_upper_diag == 0:
            total_ndists = 0
        elif 0 < tile_upper_diag <= tile_width - tile_height + 1:
            total_ndists = (tile_upper_diag - tile_lower_diag) * tile_height
        else:  # tile_upper_diag > tile_width - tile_height + 1
            upper_ndists = tile_width - (tile_upper_diag - 1)
            total_ndists = (
                tile_width - tile_height - tile_lower_diag + 1
            ) * tile_height
            total_ndists += _total_trapezoid_ndists(
                tile_height - 1,
                upper_ndists,
                tile_upper_diag - (tile_width - tile_height + 1),
            )
    else:  # tile_lower_diag > tile_width - tile_height
        lower_ndists = tile_width - tile_lower_diag
        upper_ndists = tile_width - (tile_upper_diag - 1)
        total_ndists = _total_trapezoid_ndists(
            upper_ndists, lower_ndists, tile_upper_diag - tile_lower_diag
        )

    return total_ndists


def merge_topk_PI(PA, PB, IA, IB):
    if PA.ndim == 1:
        for i in range(PA.shape[0]):
            if PB[i] < PA[i] and IB[i] != IA[i]:
                PA[i] = PB[i]
                IA[i] = IB[i]
        return

    else:
        k = PA.shape[1]
        for i in range(PA.shape[0]):
            _, _, overlap_idx_B = np.intersect1d(IA[i], IB[i], return_indices=True)
            PB[i, overlap_idx_B] = np.inf
            IB[i, overlap_idx_B] = -1

        profile = np.column_stack((PA, PB))
        indices = np.column_stack((IA, IB))
        IDX = np.argsort(profile, axis=1, kind="mergesort")
        profile[:, :] = np.take_along_axis(profile, IDX, axis=1)
        indices[:, :] = np.take_along_axis(indices, IDX, axis=1)

        PA[:, :] = profile[:, :k]
        IA[:, :] = indices[:, :k]

        return


def merge_topk_ρI(ρA, ρB, IA, IB):
    # This function merges two pearson profiles `ρA` and `ρB`, and updates `ρA`
    # and `IA` accordingly. When the inputs are 1D, `ρA[i]` is updated if
    #  `ρA[i] < ρB[i]` and IA[i] != IB[i]. When the inputs are 2D, each row in
    #  `ρA` and `ρB` is sorted in ascending order. we want to keep top-k largest
    # values in merging row `ρA[i]` and `ρB[i]`.

    # In case of ties between `ρA` and `ρB`, the priority is with `ρA`. In case
    # of ties within `ρA, the priority is with an element with greater index.
    # Example
    # note: the prime symbol is to distinguish two elements with same value
    # ρA = [0, 0', 1], and ρB = [0, 1, 1'].
    # merging outcome: [1_B, 1'_B, 1_A]

    # Naive Implementation:
    # keeping top-k largest with the aforementioned priority rules is the same as
    # `merge_topk_PI` but with swapping `ρA` and `ρB`

    # For the same example:
    # merging `ρB` and `ρA` in ascending order while choosing `ρB` over `ρA` in
    # case of ties: [0_B, 0_A, 0'_A, 1_B, 1'_B, 1_A], and the second half of this array
    # is the desirable outcome.
    if ρA.ndim == 1:
        for i in range(ρA.shape[0]):
            if ρB[i] > ρA[i] and IB[i] != IA[i]:
                ρA[i] = ρB[i]
                IA[i] = IB[i]
        return

    else:
        k = ρA.shape[1]
        for i in range(ρA.shape[0]):
            _, _, overlap_idx_B = np.intersect1d(IA[i], IB[i], return_indices=True)
            ρB[i, overlap_idx_B] = -np.inf
            IB[i, overlap_idx_B] = -1

        profile = np.column_stack((ρB, ρA))
        indices = np.column_stack((IB, IA))

        idx = np.argsort(profile, axis=1, kind="mergesort")
        profile[:, :] = np.take_along_axis(profile, idx, axis=1)
        indices[:, :] = np.take_along_axis(indices, idx, axis=1)

        # keep the last k elements (top-k largest values)
        ρA[:, :] = profile[:, k:]
        IA[:, :] = indices[:, k:]

        return


def find_matches(D, excl_zone, max_distance, max_matches=None):
    if max_matches is None:
        max_matches = len(D)

    matches = []
    for i in range(D.size):
        dist = D[i]
        if dist <= max_distance:
            matches.append(i)

    # Removes indices that are inside the exclusion zone of some occurrence with
    # a smaller distance to the query
    matches.sort(key=lambda x: D[x])
    result = []
    while len(matches) > 0:
        idx = matches[0]
        result.append([D[idx], idx])
        matches = [x for x in matches if x < idx - excl_zone or x > idx + excl_zone]

    return np.array(result[:max_matches], dtype=object)


def isconstant_func_stddev_threshold(a, w, quantile_threshold=0, stddev_threshold=None):
    sliding_stddev = rolling_nanstd(a, w)
    if stddev_threshold is None:
        stddev_threshold = np.quantile(sliding_stddev, quantile_threshold)
        if quantile_threshold == 0:  # pragma: no cover
            stddev_threshold = 0

    return sliding_stddev <= stddev_threshold


def mpdist_custom_func(P_ABBA, m, percentage, n_A, n_B):
    percentage = min(percentage, 1.0)
    percentage = max(percentage, 0.0)
    k = min(math.ceil(percentage * (n_A + n_B)), n_A - m + 1 + n_B - m + 1 - 1)
    P_ABBA.sort()
    MPdist = P_ABBA[k]
    if ~np.isfinite(MPdist):  # pragma: no cover
        k = np.count_nonzero(np.isfinite(P_ABBA[:k])) - 1
        MPdist = P_ABBA[k]

    return MPdist
