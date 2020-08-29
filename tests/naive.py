import numpy as np
from scipy.spatial.distance import cdist
from stumpy import core


def z_norm(a, axis=0, threshold=1e-7):
    std = np.std(a, axis, keepdims=True)
    std[np.less(std, threshold, where=~np.isnan(std))] = 1.0

    return (a - np.mean(a, axis, keepdims=True)) / std


def distance(a, b):
    return np.linalg.norm(a - b)


def apply_exclusion_zone(D, trivial_idx, excl_zone):
    start = max(0, trivial_idx - excl_zone)
    stop = min(D.shape[-1], trivial_idx + excl_zone + 1)
    for i in range(start, stop):
        D[..., i] = np.inf


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


def distance_matrix(T_A, T_B, m):
    distance_matrix = np.array(
        [distance_profile(Q, T_A, m) for Q in core.rolling_window(T_B, m)]
    )

    return distance_matrix


def mass(Q, T, m, trivial_idx=None, excl_zone=0, ignore_trivial=False):
    Q = np.asarray(Q)
    T = np.asarray(T)

    D = distance_profile(Q, T, m)
    if ignore_trivial:
        apply_exclusion_zone(D, trivial_idx, excl_zone)
        start = max(0, trivial_idx - excl_zone)
        stop = min(D.shape[0], trivial_idx + excl_zone + 1)
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


def stamp(T_A, m, exclusion_zone=None, T_B=None):
    if T_B is None:  # self-join
        result = np.array(
            [
                mass(Q, T_A, m, i, exclusion_zone, True)
                for i, Q in enumerate(core.rolling_window(T_A, m))
            ],
            dtype=object,
        )
    else:
        result = np.array(
            [mass(Q, T_A, m) for Q in core.rolling_window(T_B, m)],
            dtype=object,
        )
    return result


def stump(T_A, m, exclusion_zone=None, T_B=None):
    """
    Traverse distance matrix along the diagonals and update the matrix profile and
    matrix profile indices
    """
    if T_B is None:  # self-join:
        ignore_trivial = True
        T_B = T_A.copy()
        distance_matrix = np.array(
            [distance_profile(Q, T_A, m) for Q in core.rolling_window(T_B, m)]
        )
    else:
        ignore_trivial = False
        distance_matrix = np.array(
            [distance_profile(Q, T_A, m) for Q in core.rolling_window(T_B, m)]
        )

    distance_matrix[np.isnan(distance_matrix)] = np.inf

    n_A = T_A.shape[0]
    n_B = T_B.shape[0]
    l = n_B - m + 1
    if exclusion_zone is None:
        exclusion_zone = int(np.ceil(m / 4))

    if ignore_trivial:
        diags = np.arange(exclusion_zone + 1, n_B - m + 1)
    else:
        diags = np.arange(-(n_B - m + 1) + 1, n_A - m + 1)

    P = np.full((l, 3), np.inf)
    I = np.full((l, 3), -1, dtype=np.int64)

    for k in diags:
        if k >= 0:
            iter_range = range(0, min(n_B - m + 1, n_A - m + 1 - k))
        else:
            iter_range = range(-k, min(n_B - m + 1, n_A - m + 1 - k))

        for i in iter_range:
            D = distance_matrix[i, i + k]
            if D < P[i, 0]:
                P[i, 0] = D
                I[i, 0] = i + k

            if ignore_trivial:  # Self-joins only
                if D < P[i + k, 0]:
                    P[i + k, 0] = D
                    I[i + k, 0] = i

                if i < i + k:
                    # Left matrix profile and left matrix profile index
                    if D < P[i + k, 1]:
                        P[i + k, 1] = D
                        I[i + k, 1] = i

                    if D < P[i, 2]:
                        # right matrix profile and right matrix profile index
                        P[i, 2] = D
                        I[i, 2] = i + k

    result = np.empty((l, 4), dtype=object)
    result[:, 0] = P[:, 0]
    result[:, 1:4] = I[:, :]

    return result


def replace_inf(x, value=0):
    x[x == np.inf] = value
    x[x == -np.inf] = value
    return


def multi_mass(Q, T, m, include=None, discords=False):
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
        D[i] = distance_profile(Q[i], T[i], m)

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


def mstump(T, m, excl_zone, include=None, discords=False):
    T = T.copy()

    d, n = T.shape
    k = n - m + 1

    P = np.full((d, k), np.inf)
    I = np.ones((d, k), dtype="int64") * -1

    for i in range(k):
        Q = T[:, i : i + m]
        D = multi_mass(Q, T, m, include, discords)

        start_row_idx = 0
        if include is not None:
            restricted_indices = include[include < include.shape[0]]
            unrestricted_indices = include[include >= include.shape[0]]
            mask = np.ones(include.shape[0], bool)
            mask[restricted_indices] = False
            tmp_swap = D[: include.shape[0]].copy()
            D[: include.shape[0]] = D[include]
            D[unrestricted_indices] = tmp_swap[mask]
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

        apply_exclusion_zone(D_prime_prime, i, excl_zone)

        P_i, I_i = PI(D_prime_prime, i, excl_zone)

        for dim in range(T.shape[0]):
            col_mask = P[dim] > P_i[dim]
            P[dim, col_mask] = P_i[dim, col_mask]
            I[dim, col_mask] = I_i[dim, col_mask]

    return P.T, I.T


def get_array_ranges(a, n_chunks, truncate=False):
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


def aamp(T_A, m, T_B=None, exclusion_zone=None):
    T_A = np.asarray(T_A)
    T_A = T_A.copy()

    if T_B is None:
        T_B = T_A.copy()
        ignore_trivial = True
    else:
        T_B = np.asarray(T_B)
        T_B = T_B.copy()
        ignore_trivial = False

    T_A[np.isinf(T_A)] = np.nan
    T_B[np.isinf(T_B)] = np.nan

    rolling_T_A = core.rolling_window(T_A, m)
    rolling_T_B = core.rolling_window(T_B, m)

    n_A = T_A.shape[0]
    n_B = T_B.shape[0]
    l = n_B - m + 1
    if exclusion_zone is None:
        exclusion_zone = int(np.ceil(m / 4))

    distance_matrix = cdist(rolling_T_B, rolling_T_A)

    if ignore_trivial:
        diags = np.arange(exclusion_zone + 1, n_B - m + 1)
    else:
        diags = np.arange(-(n_B - m + 1) + 1, n_A - m + 1)

    P = np.full((l, 3), np.inf)
    I = np.full((l, 3), -1, dtype=np.int64)

    for k in diags:
        if k >= 0:
            iter_range = range(0, min(n_B - m + 1, n_A - m + 1 - k))
        else:
            iter_range = range(-k, min(n_B - m + 1, n_A - m + 1 - k))

        for i in iter_range:
            D = distance_matrix[i, i + k]
            if D < P[i, 0]:
                P[i, 0] = D
                I[i, 0] = i + k

            if ignore_trivial:  # Self-joins only
                if D < P[i + k, 0]:
                    P[i + k, 0] = D
                    I[i + k, 0] = i

                if i < i + k:
                    # Left matrix profile and left matrix profile index
                    if D < P[i + k, 1]:
                        P[i + k, 1] = D
                        I[i + k, 1] = i

                    if D < P[i, 2]:
                        # right matrix profile and right matrix profile index
                        P[i, 2] = D
                        I[i, 2] = i + k

    result = np.empty((l, 4), dtype=object)
    result[:, 0] = P[:, 0]
    result[:, 1:4] = I[:, :]

    return result


class aampi_egress(object):
    def __init__(self, T, m, excl_zone=None):
        self._T = np.asarray(T)
        self._T = self._T.copy()
        self._T_isfinite = np.isfinite(self._T)
        self._m = m
        if excl_zone is None:
            self._excl_zone = int(np.ceil(self._m / 4))

        self._l = self._T.shape[0] - m + 1
        mp = aamp(T, m)
        self.P_ = mp[:, 0]
        self.I_ = mp[:, 1].astype(np.int64)
        self.left_P_ = np.full(self.P_.shape, np.inf)
        self.left_I_ = mp[:, 2].astype(np.int64)
        for i, j in enumerate(self.left_I_):
            if j >= 0:
                D = core.mass_absolute(
                    self._T[i : i + self._m], self._T[j : j + self._m]
                )
                self.left_P_[i] = D[0]

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

        self.P_[:] = np.roll(self.P_, -1)
        self.I_[:] = np.roll(self.I_, -1)
        self.left_P_[:] = np.roll(self.left_P_, -1)
        self.left_I_[:] = np.roll(self.left_I_, -1)

        D = core.mass_absolute(self._T[-self._m :], self._T)
        T_subseq_isfinite = np.all(
            core.rolling_window(self._T_isfinite, self._m), axis=1
        )
        D[~T_subseq_isfinite] = np.inf
        if np.any(~self._T_isfinite[-self._m :]):
            D[:] = np.inf

        apply_exclusion_zone(D, D.shape[0] - 1, self._excl_zone)
        for j in range(D.shape[0]):
            if D[j] < self.P_[j]:
                self.I_[j] = D.shape[0] - 1 + self._n_appended
                self.P_[j] = D[j]

        I_last = np.argmin(D)

        if np.isinf(D[I_last]):
            self.I_[-1] = -1
            self.P_[-1] = np.inf
        else:
            self.I_[-1] = I_last + self._n_appended
            self.P_[-1] = D[I_last]

        self.left_I_[-1] = I_last + self._n_appended
        self.left_P_[-1] = D[I_last]


class stumpi_egress(object):
    def __init__(self, T, m, excl_zone=None):
        self._T = np.asarray(T)
        self._T = self._T.copy()
        self._T_isfinite = np.isfinite(self._T)
        self._m = m
        if excl_zone is None:
            self._excl_zone = int(np.ceil(self._m / 4))

        self._l = self._T.shape[0] - m + 1
        mp = stump(T, m)
        self.P_ = mp[:, 0]
        self.I_ = mp[:, 1].astype(np.int64)
        self.left_P_ = np.full(self.P_.shape, np.inf)
        self.left_I_ = mp[:, 2].astype(np.int64)
        for i, j in enumerate(self.left_I_):
            if j >= 0:
                D = core.mass(self._T[i : i + self._m], self._T[j : j + self._m])
                self.left_P_[i] = D[0]

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

        self.P_[:] = np.roll(self.P_, -1)
        self.I_[:] = np.roll(self.I_, -1)
        self.left_P_[:] = np.roll(self.left_P_, -1)
        self.left_I_[:] = np.roll(self.left_I_, -1)

        D = core.mass(self._T[-self._m :], self._T)
        T_subseq_isfinite = np.all(
            core.rolling_window(self._T_isfinite, self._m), axis=1
        )
        D[~T_subseq_isfinite] = np.inf
        if np.any(~self._T_isfinite[-self._m :]):
            D[:] = np.inf

        apply_exclusion_zone(D, D.shape[0] - 1, self._excl_zone)
        for j in range(D.shape[0]):
            if D[j] < self.P_[j]:
                self.I_[j] = D.shape[0] - 1 + self._n_appended
                self.P_[j] = D[j]

        I_last = np.argmin(D)

        if np.isinf(D[I_last]):
            self.I_[-1] = -1
            self.P_[-1] = np.inf
        else:
            self.I_[-1] = I_last + self._n_appended
            self.P_[-1] = D[I_last]

        self.left_I_[-1] = I_last + self._n_appended
        self.left_P_[-1] = D[I_last]
