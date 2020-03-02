import numpy as np
from stumpy import core


def z_norm(a, axis=0, threshold=1e-7):
    std = np.std(a, axis, keepdims=True)
    std[np.less(std, threshold, where=~np.isnan(std))] = 1.0

    return (a - np.mean(a, axis, keepdims=True)) / std


def naive_mass(Q, T, m, trivial_idx=None, excl_zone=0, ignore_trivial=False):
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


def naive_compute_mean_std(T, m):
    n = T.shape[0]

    cumsum_T = np.empty(len(T) + 1)
    np.cumsum(T, out=cumsum_T[1:])  # store output in cumsum_T[1:]
    cumsum_T[0] = 0

    cumsum_T_squared = np.empty(len(T) + 1)
    np.cumsum(np.square(T), out=cumsum_T_squared[1:])
    cumsum_T_squared[0] = 0

    subseq_sum_T = cumsum_T[m:] - cumsum_T[: n - m + 1]
    subseq_sum_T_squared = cumsum_T_squared[m:] - cumsum_T_squared[: n - m + 1]
    M_T = subseq_sum_T / m
    Σ_T = np.abs((subseq_sum_T_squared / m) - np.square(M_T))
    Σ_T = np.sqrt(Σ_T)

    return M_T, Σ_T


def naive_compute_mean_std_multidimensional(T, m):
    n = T.shape[1]
    nrows, ncols = T.shape

    cumsum_T = np.empty((nrows, ncols + 1))
    np.cumsum(T, axis=1, out=cumsum_T[:, 1:])  # store output in cumsum_T[1:]
    cumsum_T[:, 0] = 0

    cumsum_T_squared = np.empty((nrows, ncols + 1))
    np.cumsum(np.square(T), axis=1, out=cumsum_T_squared[:, 1:])
    cumsum_T_squared[:, 0] = 0

    subseq_sum_T = cumsum_T[:, m:] - cumsum_T[:, : n - m + 1]
    subseq_sum_T_squared = cumsum_T_squared[:, m:] - cumsum_T_squared[:, : n - m + 1]
    M_T = subseq_sum_T / m
    Σ_T = np.abs((subseq_sum_T_squared / m) - np.square(M_T))
    Σ_T = np.sqrt(Σ_T)

    return M_T, Σ_T


def replace_inf(x, value=0):
    x[x == np.inf] = value
    x[x == -np.inf] = value
    return
