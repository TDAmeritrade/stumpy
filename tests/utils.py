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


def replace_inf(x, value=0):
    x[x == np.inf] = value
    x[x == -np.inf] = value
    return


def naive_multi_mass(Q, T, m):
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


def naive_mstump(T, m, excl_zone):
    T = T.copy()

    d, n = T.shape
    k = n - m + 1

    P = np.full((d, k), np.inf)
    I = np.ones((d, k), dtype="int64") * -1

    for i in range(k):
        Q = T[:, i : i + m]
        D = naive_multi_mass(Q, T, m)

        P_i, I_i = naive_PI(D, i, excl_zone)

        for dim in range(T.shape[0]):
            col_mask = P[dim] > P_i[dim]
            P[dim, col_mask] = P_i[dim, col_mask]
            I[dim, col_mask] = I_i[dim, col_mask]

    return P.T, I.T
