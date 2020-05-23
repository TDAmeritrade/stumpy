import numpy as np
from stumpy import core


def z_norm(a, axis=0, threshold=1e-7):
    std = np.std(a, axis, keepdims=True)
    std[np.less(std, threshold, where=~np.isnan(std))] = 1.0

    return (a - np.mean(a, axis, keepdims=True)) / std


def naive_distance_profile(Q, T, m):
    T = T.copy()
    Q = Q.copy()

    T[np.isinf(T)] = np.nan
    Q[np.isinf(Q)] = np.nan

    D = np.linalg.norm(z_norm(core.rolling_window(T, m), 1) - z_norm(Q), axis=1)

    return D


def naive_mass(Q, T, m, trivial_idx=None, excl_zone=0, ignore_trivial=False):
    T = T.copy()
    Q = Q.copy()

    T[np.isinf(T)] = np.nan
    Q[np.isinf(Q)] = np.nan

    D = np.linalg.norm(z_norm(core.rolling_window(T, m), 1) - z_norm(Q), axis=1)
    if ignore_trivial:
        start = max(0, trivial_idx - excl_zone)
        stop = min(T.shape[0] - Q.shape[0] + 1, trivial_idx + excl_zone)
        D[start : stop + 1] = np.inf
    D[np.isnan(D)] = np.inf

    I = np.argmin(D)
    P = D[I]

    if P == np.inf:
        I = -1

    # Get left and right matrix profiles for self-joins
    if ignore_trivial and trivial_idx > 0:
        PL = np.inf
        IL = -1
        for i in range(trivial_idx):
            if D[i] < PL:
                IL = i
                PL = D[i]
        if start <= IL < stop:
            IL = -1
    else:
        IL = -1

    if ignore_trivial and trivial_idx + 1 < D.shape[0]:
        PR = np.inf
        IR = -1
        for i in range(trivial_idx + 1, D.shape[0]):
            if D[i] < PR:
                IR = i
                PR = D[i]
        if start <= IR < stop:
            IR = -1
    else:
        IR = -1

    return P, I, IL, IR


def naive_stamp(T_A, m, exclusion_zone=None, T_B=None):
    if T_B is None:  # self-join
        result = np.array(
            [
                naive_mass(Q, T_A, m, i, exclusion_zone, True)
                for i, Q in enumerate(core.rolling_window(T_A, m))
            ],
            dtype=object,
        )
    else:
        result = np.array(
            [naive_mass(Q, T_A, m) for Q in core.rolling_window(T_B, m)], dtype=object,
        )
    return result


def replace_inf(x, value=0):
    x[x == np.inf] = value
    x[x == -np.inf] = value
    return


def naive_multi_mass(Q, T, m, include=None):
    T = T.copy()
    Q = Q.copy()

    T[np.isinf(T)] = np.nan
    Q[np.isinf(Q)] = np.nan

    d, n = T.shape

    D = np.empty((d, n - m + 1))
    for i in range(d):
        D[i] = np.linalg.norm(
            z_norm(core.rolling_window(T[i], m), 1) - z_norm(Q[i]), axis=1
        )
    D[np.isnan(D)] = np.inf

    if include is not None:
        restricted_indices = include[include < include.shape[0]]
        unrestricted_indices = include[include >= include.shape[0]]
        mask = np.ones(include.shape[0], bool)
        mask[restricted_indices] = False
        tmp_swap = D[: include.shape[0]].copy()
        D[: include.shape[0]] = D[include]
        D[unrestricted_indices] = tmp_swap[mask]
        D[include.shape[0] :].sort(axis=0)
    else:
        D = np.sort(D, axis=0)

    D_prime = np.zeros(n - m + 1)
    D_prime_prime = np.zeros((d, n - m + 1))
    for i in range(d):
        D_prime[:] = D_prime + D[i]
        D_prime_prime[i, :] = D_prime / (i + 1)

    return D_prime_prime


def naive_PI(D, trivial_idx, excl_zone):
    d, k = D.shape

    P = np.full((d, k), np.inf)
    I = np.ones((d, k), dtype="int64") * -1

    zone_start = max(0, trivial_idx - excl_zone)
    zone_end = min(k, trivial_idx + excl_zone)
    D[:, zone_start : zone_end + 1] = np.inf

    for i in range(d):
        col_mask = P[i] > D[i]
        P[i, col_mask] = D[i, col_mask]
        I[i, col_mask] = trivial_idx

    return P, I


def naive_mstump(T, m, excl_zone, include=None):
    T = T.copy()

    d, n = T.shape
    k = n - m + 1

    P = np.full((d, k), np.inf)
    I = np.ones((d, k), dtype="int64") * -1

    for i in range(k):
        Q = T[:, i : i + m]
        D = naive_multi_mass(Q, T, m, include)

        P_i, I_i = naive_PI(D, i, excl_zone)

        for dim in range(T.shape[0]):
            col_mask = P[dim] > P_i[dim]
            P[dim, col_mask] = P_i[dim, col_mask]
            I[dim, col_mask] = I_i[dim, col_mask]

    return P.T, I.T


def get_naive_array_ranges(a, n_chunks, truncate=False):
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
