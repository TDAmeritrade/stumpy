import math
import numpy as np
from scipy.spatial.distance import cdist
from stumpy import core, config


def z_norm(a, axis=0, threshold=1e-7):
    std = np.std(a, axis, keepdims=True)
    std[np.less(std, threshold, where=~np.isnan(std))] = 1.0

    return (a - np.mean(a, axis, keepdims=True)) / std


def distance(a, b, axis=0):
    return np.linalg.norm(a - b, axis=axis)


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


def aamp_distance_profile(Q, T, m):
    T_inf = np.isinf(T)
    if np.any(T_inf):
        T = T.copy()
        T[T_inf] = np.nan

    Q_inf = np.isinf(Q)
    if np.any(Q_inf):
        Q = Q.copy()
        Q[Q_inf] = np.nan

    D = np.linalg.norm(core.rolling_window(T, m) - Q, axis=1)

    return D


def distance_matrix(T_A, T_B, m):
    distance_matrix = np.array(
        [distance_profile(Q, T_B, m) for Q in core.rolling_window(T_A, m)]
    )

    return distance_matrix


def aamp_distance_matrix(T_A, T_B, m):
    T_A[np.isinf(T_A)] = np.nan
    T_B[np.isinf(T_B)] = np.nan

    rolling_T_A = core.rolling_window(T_A, m)
    rolling_T_B = core.rolling_window(T_B, m)

    distance_matrix = cdist(rolling_T_A, rolling_T_B)

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


def stamp(T_A, m, T_B=None, exclusion_zone=None):
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
            [mass(Q, T_B, m) for Q in core.rolling_window(T_A, m)],
            dtype=object,
        )
    return result


def stump(T_A, m, T_B=None, exclusion_zone=None):
    """
    Traverse distance matrix along the diagonals and update the matrix profile and
    matrix profile indices
    """
    if T_B is None:  # self-join:
        ignore_trivial = True
        distance_matrix = np.array(
            [distance_profile(Q, T_A, m) for Q in core.rolling_window(T_A, m)]
        )
        T_B = T_A.copy()
    else:
        ignore_trivial = False
        distance_matrix = np.array(
            [distance_profile(Q, T_B, m) for Q in core.rolling_window(T_A, m)]
        )

    distance_matrix[np.isnan(distance_matrix)] = np.inf

    n_A = T_A.shape[0]
    n_B = T_B.shape[0]
    l = n_A - m + 1
    if exclusion_zone is None:
        exclusion_zone = int(np.ceil(m / config.STUMPY_EXCL_ZONE_DENOM))

    if ignore_trivial:
        diags = np.arange(exclusion_zone + 1, n_A - m + 1)
    else:
        diags = np.arange(-(n_A - m + 1) + 1, n_B - m + 1)

    P = np.full((l, 3), np.inf)
    I = np.full((l, 3), -1, dtype=np.int64)

    for k in diags:
        if k >= 0:
            iter_range = range(0, min(n_A - m + 1, n_B - m + 1 - k))
        else:
            iter_range = range(-k, min(n_A - m + 1, n_B - m + 1 - k))

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
    l = n_A - m + 1
    if exclusion_zone is None:
        exclusion_zone = int(np.ceil(m / config.STUMPY_EXCL_ZONE_DENOM))

    distance_matrix = cdist(rolling_T_A, rolling_T_B)

    if ignore_trivial:
        diags = np.arange(exclusion_zone + 1, n_A - m + 1)
    else:
        diags = np.arange(-(n_A - m + 1) + 1, n_B - m + 1)

    P = np.full((l, 3), np.inf)
    I = np.full((l, 3), -1, dtype=np.int64)

    for k in diags:
        if k >= 0:
            iter_range = range(0, min(n_A - m + 1, n_B - m + 1 - k))
        else:
            iter_range = range(-k, min(n_A - m + 1, n_B - m + 1 - k))

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


def multi_mass_absolute(Q, T, m, include=None, discords=False):
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
        D[i] = aamp_distance_profile(Q[i], T[i], m)

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


def multi_distance_profile(query_idx, T, m, include=None, discords=False):
    excl_zone = int(np.ceil(m / config.STUMPY_EXCL_ZONE_DENOM))
    d, n = T.shape
    Q = T[:, query_idx : query_idx + m]
    D = multi_mass(Q, T, m, include, discords)

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

    apply_exclusion_zone(D_prime_prime, query_idx, excl_zone)

    return D_prime_prime


def mstump(T, m, excl_zone, include=None, discords=False):
    T = T.copy()

    d, n = T.shape
    k = n - m + 1

    P = np.full((d, k), np.inf)
    I = np.ones((d, k), dtype="int64") * -1

    for i in range(k):
        D = multi_distance_profile(i, T, m, include, discords)
        P_i, I_i = PI(D, i, excl_zone)

        for dim in range(T.shape[0]):
            col_mask = P[dim] > P_i[dim]
            P[dim, col_mask] = P_i[dim, col_mask]
            I[dim, col_mask] = I_i[dim, col_mask]

    return P, I


def maamp_multi_distance_profile(query_idx, T, m, include=None, discords=False):
    excl_zone = int(np.ceil(m / config.STUMPY_EXCL_ZONE_DENOM))
    d, n = T.shape
    Q = T[:, query_idx : query_idx + m]
    D = multi_mass_absolute(Q, T, m, include, discords)

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

    apply_exclusion_zone(D_prime_prime, query_idx, excl_zone)

    return D_prime_prime


def maamp(T, m, excl_zone, include=None, discords=False):
    T = T.copy()

    d, n = T.shape
    k = n - m + 1

    P = np.full((d, k), np.inf)
    I = np.ones((d, k), dtype="int64") * -1

    for i in range(k):
        D = maamp_multi_distance_profile(i, T, m, include, discords)
        P_i, I_i = PI(D, i, excl_zone)

        for dim in range(T.shape[0]):
            col_mask = P[dim] > P_i[dim]
            P[dim, col_mask] = P_i[dim, col_mask]
            I[dim, col_mask] = I_i[dim, col_mask]

    return P, I


def subspace(T, m, motif_idx, nn_idx, k, include=None, discords=False):
    D = distance(
        z_norm(T[:, motif_idx : motif_idx + m], axis=1),
        z_norm(T[:, nn_idx : nn_idx + m], axis=1),
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


def maamp_subspace(T, m, motif_idx, nn_idx, k, include=None, discords=False):
    D = distance(
        T[:, motif_idx : motif_idx + m],
        T[:, nn_idx : nn_idx + m],
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
    def __init__(self, T, m, excl_zone=None):
        self._T = np.asarray(T)
        self._T = self._T.copy()
        self._T_isfinite = np.isfinite(self._T)
        self._m = m
        if excl_zone is None:
            self._excl_zone = int(np.ceil(self._m / config.STUMPY_EXCL_ZONE_DENOM))

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
            self._excl_zone = int(np.ceil(self._m / config.STUMPY_EXCL_ZONE_DENOM))

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


def across_series_nearest_neighbors(Ts, Ts_idx, subseq_idx, m):
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
        dist_profile = distance_profile(Q, Ts[i], len(Q))
        nns_subseq_idx[i] = np.argmin(dist_profile)
        nns_radii[i] = dist_profile[nns_subseq_idx[i]]

    return nns_radii, nns_subseq_idx


def get_central_motif(Ts, bsf_radius, bsf_Ts_idx, bsf_subseq_idx, m):
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
        Ts, bsf_Ts_idx, bsf_subseq_idx, m
    )
    bsf_nns_mean_radii = bsf_nns_radii.mean()

    candidate_nns_Ts_idx = np.flatnonzero(np.isclose(bsf_nns_radii, bsf_radius))
    candidate_nns_subseq_idx = bsf_nns_subseq_idx[candidate_nns_Ts_idx]

    for Ts_idx, subseq_idx in zip(candidate_nns_Ts_idx, candidate_nns_subseq_idx):
        candidate_nns_radii, _ = across_series_nearest_neighbors(
            Ts, Ts_idx, subseq_idx, m
        )
        if (
            np.isclose(candidate_nns_radii.max(), bsf_radius)
            and candidate_nns_radii.mean() < bsf_nns_mean_radii
        ):
            bsf_Ts_idx = Ts_idx
            bsf_subseq_idx = subseq_idx
            bsf_nns_mean_radii = candidate_nns_radii.mean()

    return bsf_radius, bsf_Ts_idx, bsf_subseq_idx


def consensus_search(Ts, m):
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
                mp = stump(Ts[j], m, Ts[i])
                radii = np.maximum(radii, mp[:, 0])
        min_radius_idx = np.argmin(radii)
        min_radius = radii[min_radius_idx]
        if min_radius < bsf_radius:
            bsf_radius = min_radius
            bsf_Ts_idx = j
            bsf_subseq_idx = min_radius_idx

    return bsf_radius, bsf_Ts_idx, bsf_subseq_idx


def ostinato(Ts, m):
    bsf_radius, bsf_Ts_idx, bsf_subseq_idx = consensus_search(Ts, m)
    radius, Ts_idx, subseq_idx = get_central_motif(
        Ts, bsf_radius, bsf_Ts_idx, bsf_subseq_idx, m
    )
    return radius, Ts_idx, subseq_idx


def aamp_across_series_nearest_neighbors(Ts, Ts_idx, subseq_idx, m):
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
        dist_profile = aamp_distance_profile(Q, Ts[i], len(Q))
        nns_subseq_idx[i] = np.argmin(dist_profile)
        nns_radii[i] = dist_profile[nns_subseq_idx[i]]

    return nns_radii, nns_subseq_idx


def get_aamp_central_motif(Ts, bsf_radius, bsf_Ts_idx, bsf_subseq_idx, m):
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
    bsf_nns_radii, bsf_nns_subseq_idx = aamp_across_series_nearest_neighbors(
        Ts, bsf_Ts_idx, bsf_subseq_idx, m
    )
    bsf_nns_mean_radii = bsf_nns_radii.mean()

    candidate_nns_Ts_idx = np.flatnonzero(np.isclose(bsf_nns_radii, bsf_radius))
    candidate_nns_subseq_idx = bsf_nns_subseq_idx[candidate_nns_Ts_idx]

    for Ts_idx, subseq_idx in zip(candidate_nns_Ts_idx, candidate_nns_subseq_idx):
        candidate_nns_radii, _ = aamp_across_series_nearest_neighbors(
            Ts, Ts_idx, subseq_idx, m
        )
        if (
            np.isclose(candidate_nns_radii.max(), bsf_radius)
            and candidate_nns_radii.mean() < bsf_nns_mean_radii
        ):
            bsf_Ts_idx = Ts_idx
            bsf_subseq_idx = subseq_idx
            bsf_nns_mean_radii = candidate_nns_radii.mean()

    return bsf_radius, bsf_Ts_idx, bsf_subseq_idx


def aamp_consensus_search(Ts, m):
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
                mp = aamp(Ts[j], m, Ts[i])
                radii = np.maximum(radii, mp[:, 0])
        min_radius_idx = np.argmin(radii)
        min_radius = radii[min_radius_idx]
        if min_radius < bsf_radius:
            bsf_radius = min_radius
            bsf_Ts_idx = j
            bsf_subseq_idx = min_radius_idx

    return bsf_radius, bsf_Ts_idx, bsf_subseq_idx


def aamp_ostinato(Ts, m):
    bsf_radius, bsf_Ts_idx, bsf_subseq_idx = aamp_consensus_search(Ts, m)
    radius, Ts_idx, subseq_idx = get_aamp_central_motif(
        Ts, bsf_radius, bsf_Ts_idx, bsf_subseq_idx, m
    )
    return radius, Ts_idx, subseq_idx


def mpdist_vect(T_A, T_B, m, percentage=0.05, k=None):
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

    for i in range(n_B - n_A + 1):
        P_ABBA[:j] = stump(T_A, m, T_B[i : i + n_A])[:, 0]
        P_ABBA[j:] = stump(T_B[i : i + n_A], m, T_A)[:, 0]
        P_ABBA.sort()
        MPdist_vect[i] = P_ABBA[min(k, P_ABBA.shape[0] - 1)]

    return MPdist_vect


def aampdist_vect(T_A, T_B, m, percentage=0.05, k=None):
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
        P_ABBA[:j] = aamp(T_A, m, T_B[i : i + n_A])[:, 0]
        P_ABBA[j:] = aamp(T_B[i : i + n_A], m, T_A)[:, 0]
        P_ABBA.sort()
        aaMPdist_vect[i] = P_ABBA[k]

    return aaMPdist_vect


def mpdist(T_A, T_B, m, percentage=0.05, k=None):
    percentage = min(percentage, 1.0)
    percentage = max(percentage, 0.0)
    n_A = T_A.shape[0]
    n_B = T_B.shape[0]
    P_ABBA = np.empty(n_A - m + 1 + n_B - m + 1, dtype=np.float64)
    if k is not None:
        k = int(k)
    else:
        k = min(math.ceil(percentage * (n_A + n_B)), n_A - m + 1 + n_B - m + 1 - 1)

    P_ABBA[: n_A - m + 1] = stump(T_A, m, T_B)[:, 0]
    P_ABBA[n_A - m + 1 :] = stump(T_B, m, T_A)[:, 0]

    P_ABBA.sort()
    MPdist = P_ABBA[k]
    if ~np.isfinite(MPdist):
        k = np.isfinite(P_ABBA[:k]).sum() - 1
        MPdist = P_ABBA[k]

    return MPdist


def aampdist(T_A, T_B, m, percentage=0.05, k=None):
    percentage = min(percentage, 1.0)
    percentage = max(percentage, 0.0)
    n_A = T_A.shape[0]
    n_B = T_B.shape[0]
    P_ABBA = np.empty(n_A - m + 1 + n_B - m + 1, dtype=np.float64)
    if k is not None:
        k = int(k)
    else:
        k = min(math.ceil(percentage * (n_A + n_B)), n_A - m + 1 + n_B - m + 1 - 1)

    P_ABBA[: n_A - m + 1] = aamp(T_A, m, T_B)[:, 0]
    P_ABBA[n_A - m + 1 :] = aamp(T_B, m, T_A)[:, 0]

    P_ABBA.sort()
    MPdist = P_ABBA[k]
    if ~np.isfinite(MPdist):
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
    mpdist_vect_func=mpdist_vect,
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
        D[i, :] = mpdist_vect_func(
            S_i,
            T,
            s,
            percentage=mpdist_percentage,
            k=mpdist_k,
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
):

    D = get_all_mpdist_profiles(
        T,
        m,
        percentage,
        s,
        mpdist_percentage,
        mpdist_k,
    )

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


def aampdist_snippets(
    T,
    m,
    k,
    percentage=1.0,
    s=None,
    mpdist_percentage=0.05,
    mpdist_k=None,
):

    D = get_all_mpdist_profiles(
        T,
        m,
        percentage,
        s,
        mpdist_percentage,
        mpdist_k,
        aampdist_vect,
    )

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


def prescrump(T_A, m, T_B, s, exclusion_zone=None):
    dist_matrix = distance_matrix(T_A, T_B, m)

    n_A = T_A.shape[0]
    l = n_A - m + 1

    P = np.empty(l)
    I = np.empty(l, dtype=np.int64)
    P[:] = np.inf
    I[:] = -1

    for i in np.random.permutation(range(0, l, s)):
        distance_profile = dist_matrix[i]
        if exclusion_zone is not None:
            apply_exclusion_zone(distance_profile, i, exclusion_zone)
        I[i] = np.argmin(distance_profile)
        P[i] = distance_profile[I[i]]
        if P[i] == np.inf:
            I[i] = -1
        else:
            j = I[i]
            for k in range(1, min(s, l - max(i, j))):
                d = dist_matrix[i + k, j + k]
                if d < P[i + k]:
                    P[i + k] = d
                    I[i + k] = j + k
                if d < P[j + k]:
                    P[j + k] = d
                    I[j + k] = i + k

            for k in range(1, min(s, i + 1, j + 1)):
                d = dist_matrix[i - k, j - k]
                if d < P[i - k]:
                    P[i - k] = d
                    I[i - k] = j - k
                if d < P[j - k]:
                    P[j - k] = d
                    I[j - k] = i - k

    return P, I


def scrump(T_A, m, T_B, percentage, exclusion_zone, pre_scrump, s):
    dist_matrix = distance_matrix(T_A, T_B, m)

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

    out = np.full((l, 4), np.inf, dtype=object)
    out[:, 1:] = -1
    left_P = np.full(l, np.inf, dtype=np.float64)
    right_P = np.full(l, np.inf, dtype=np.float64)

    for diag_idx in range(diags_ranges_start, diags_ranges_stop):
        k = diags[diag_idx]

        for i in range(n_A - m + 1):
            for j in range(n_B - m + 1):
                if j - i == k:
                    if dist_matrix[i, j] < out[i, 0]:
                        out[i, 0] = dist_matrix[i, j]
                        out[i, 1] = i + k

                    if exclusion_zone is not None and dist_matrix[i, j] < out[i + k, 0]:
                        out[i + k, 0] = dist_matrix[i, j]
                        out[i + k, 1] = i

                    # left matrix profile and left matrix profile indices
                    if (
                        exclusion_zone is not None
                        and i < i + k
                        and dist_matrix[i, j] < left_P[i + k]
                    ):
                        left_P[i + k] = dist_matrix[i, j]
                        out[i + k, 2] = i

                    # right matrix profile and right matrix profile indices
                    if (
                        exclusion_zone is not None
                        and i + k > i
                        and dist_matrix[i, j] < right_P[i]
                    ):
                        right_P[i] = dist_matrix[i, j]
                        out[i, 3] = i + k

    return out


def normalize_pan(pan, ms, bfs_indices, n_processed):
    idx = bfs_indices[:n_processed]
    for i in range(n_processed):
        norm = 1.0 / np.sqrt(2.0 * ms[i])
        pan[idx] = np.minimum(1.0, pan[idx] * norm)


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


def transform_pan(pan, ms, threshold, bfs_indices, n_processed):
    idx = bfs_indices[:n_processed]
    sorted_idx = np.sort(idx)
    pan[pan == np.inf] = np.nan
    normalize_pan(pan, ms, bfs_indices, n_processed)
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
