import numpy as np
from scipy import interpolate
from tslearn import metrics
from numba import njit, prange
import numba
import stumpy

#################################################################
# DTW
#################################################################
@njit(
    # "(f8[:], f8[:], f8[:, :])",
    fastmath=True
)
def _dtw(s1, s2, mask):
    l1 = s1.shape[0]
    l2 = s2.shape[0]
    cum_sum = np.full((l1 + 1, l2 + 1), np.inf)
    cum_sum[0, 0] = 0.

    for i in range(l1):
        for j in range(l2):
            if mask[i, j] == 0.0:
                cum_sum[i + 1, j + 1] = np.square(s1[i] - s2[j])
                cum_sum[i + 1, j + 1] += min(cum_sum[i, j + 1],
                                             cum_sum[i + 1, j],
                                             cum_sum[i, j])
    return cum_sum[-1, -1]

@njit(
    # "(f8[:], f8[:], i8, i8, i8[:], f8[:], f8[:], f8[:], f8[:], f8[:, :], b1)"
    parallel=True,
    fastmath=True,
)
def _dtwMP(
    T_A,
    T_B,
    m,
    r,
    indices_pruned,
    M_T,
    μ_Q,
    Σ_T,
    σ_Q,
    dtw_mask,
    ignore_trivial,
):
    n_A = T_A.shape[0]
    n_B = T_B.shape[0]
    l = n_A - m + 1

    n_threads = numba.config.NUMBA_NUM_THREADS

    excl_zone = int(np.ceil(m / stumpy.config.STUMPY_EXCL_ZONE_DENOM))

    if ignore_trivial:
        a_start = np.arange(l)
        b_start = np.arange(l)
        ndist_counts = np.full(l, l)
        if len(indices_pruned) > 0:
            mask_pruned = np.full(l, True)
            mask_pruned[indices_pruned] = False
            a_start = a_start[mask_pruned]
            b_start = a_start
            ndist_counts = ndist_counts[mask_pruned]
    else:
        lb = n_B - m + 1
        a_start = np.arange(l)
        b_start = np.arange(l)
        ndist_counts = np.full(lb, l)
        if len(indices_pruned) > 0:
            mask_pruned = np.full(l, True)
            mask_pruned[indices_pruned] = False
            a_start = a_start[mask_pruned]
            ndist_counts = ndist_counts[mask_pruned]

    a_ranges = stumpy.core._get_array_ranges(ndist_counts, n_threads, False)

    ρ = np.full((n_threads, l), np.Inf, dtype=np.float64)
    I = np.full((n_threads, l), -1, dtype=np.int64)
        
    uint64_m = np.uint64(m)
    for thread_idx in prange(n_threads):
        for i_a in range(a_ranges[thread_idx, 0], a_ranges[thread_idx, 1]):
            uint64_ia = np.uint64(a_start[i_a])
            a = (T_A[uint64_ia:uint64_ia+uint64_m] - μ_Q[uint64_ia])/σ_Q[uint64_ia]

            for i_b in b_start:
                if ignore_trivial == True:
                    uint64_ib = np.uint64(i_b)
                    if np.abs(i_b - uint64_ia) > excl_zone:
                        b = (T_B[uint64_ib:uint64_ib+uint64_m] - M_T[uint64_ib])/Σ_T[uint64_ib]
                        dist = _dtw(a, b, dtw_mask)
                        if dist < ρ[thread_idx, uint64_ia]:
                            ρ[thread_idx, uint64_ia] = dist
                            I[thread_idx, uint64_ia] = uint64_ib

                else:
                    uint64_ib = np.uint64(i_b)
                    b = (T_B[uint64_ib:uint64_ib+uint64_m] - M_T[uint64_ib])/Σ_T[uint64_ib]
                    dist = _dtw(a, b, dtw_mask)
                    if dist < ρ[thread_idx, uint64_ia]:
                        ρ[thread_idx, uint64_ia] = dist
                        I[thread_idx, uint64_ia] = uint64_ib

    for thread_idx in range(1, n_threads):
        mask = (ρ[thread_idx, :] < ρ[0, :]) & (I[thread_idx, :] != I[0, :])
        ρ[0, :][mask] = ρ[thread_idx, :][mask]
        I[0, :][mask] = I[thread_idx, :][mask]
        
    return np.sqrt(ρ[0, :]), I[0, :]

    
def dtwMP(T_A, m, r=0, T_B=None, indices_pruned=None, ignore_trivial=True):

    if T_B is None:
        T_B = T_A
        ignore_trivial = True

    if indices_pruned is None:
        indices_pruned = np.array([]).astype(np.int64)

    (T_A, μ_Q, σ_Q) = stumpy.core.preprocess(T_A, m)

    (T_B, M_T, Σ_T) = stumpy.core.preprocess(T_B, m)

    n_A = T_A.shape[0]
    n_B = T_B.shape[0]
    l = n_A - m + 1

    dtw_mask = metrics.sakoe_chiba_mask(n_A, n_B, radius=r)

    P, I = _dtwMP(
        T_A,
        T_B,
        m,
        r,
        indices_pruned,
        M_T,
        μ_Q,
        Σ_T,
        σ_Q,
        dtw_mask,
        ignore_trivial,
    )
    
    return P, I
#################################################################
# LB
#################################################################
@njit(
    # "(f8[:], i8)"
    parallel=True,
    fastmath=True,
)
def lb_envelope_keogh(ts, radius):
    n_t = ts.shape[0]
    env_up = np.empty(n_t)
    env_down = np.empty(n_t)

    for i in prange(n_t):
        min_idx = i - radius
        max_idx = i + radius + 1
        if min_idx < 0: min_idx = 0
        if max_idx > n_t: max_idx = n_t

        env_down[i] = np.min(ts[min_idx:max_idx])
        env_up[i] = np.max(ts[min_idx:max_idx])

    return env_down, env_up

@njit(
    # "(f8[:], f8[:], f8[:])"
    fastmath=True,
)
def lb_dist(L, U, ts):
    idx_up = ts > U
    idx_down = ts < L
    dist_up = np.linalg.norm(ts[idx_up]-U[idx_up])
    dist_down = np.linalg.norm(L[idx_down]-ts[idx_down])
    return dist_up**2 + dist_down**2

@njit(
    # "(f8[:], f8[:], i8, i8, i8[:], f8[:], f8[:], f8[:], f8[:], b1)"
    parallel=True,
    fastmath=True,
)
def _LB_MP_Keogh(
    T_A,
    T_B,
    m,
    r,
    indices_pruned,
    M_T,
    μ_Q,
    Σ_T,
    σ_Q,
    ignore_trivial,
):
    n_A = T_A.shape[0]
    n_B = T_B.shape[0]
    l = n_A - m + 1

    n_threads = numba.config.NUMBA_NUM_THREADS

    excl_zone = int(np.ceil(m / stumpy.config.STUMPY_EXCL_ZONE_DENOM))

    if ignore_trivial:
        a_start = np.arange(l)
        b_start = np.arange(l)
        ndist_counts = np.full(l, l)
        if len(indices_pruned) > 0:
            mask_pruned = np.full(l, True)
            mask_pruned[indices_pruned] = False
            a_start = a_start[mask_pruned]
            b_start = a_start
            ndist_counts = ndist_counts[mask_pruned]
    else:
        lb = n_B - m + 1
        a_start = np.arange(l)
        b_start = np.arange(l)
        ndist_counts = np.full(lb, l)
        if len(indices_pruned) > 0:
            mask_pruned = np.full(l, True)
            mask_pruned[indices_pruned] = False
            a_start = a_start[mask_pruned]
            ndist_counts = ndist_counts[mask_pruned]

    a_ranges = stumpy.core._get_array_ranges(ndist_counts, n_threads, False)

    ρ = np.full((n_threads, l), np.Inf, dtype=np.float64)
    I = np.full((n_threads, l), -1, dtype=np.int64)
        
    uint64_m = np.uint64(m)
    for thread_idx in prange(n_threads):
        for i_a in range(a_ranges[thread_idx, 0], a_ranges[thread_idx, 1]):
            uint64_ia = np.uint64(a_start[i_a])
            a = (T_A[uint64_ia:uint64_ia+uint64_m] - μ_Q[uint64_ia])/σ_Q[uint64_ia]
            L, U = lb_envelope_keogh(a, r)

            for i_b in b_start:
                if ignore_trivial == True:
                    uint64_ib = np.uint64(i_b)
                    if np.abs(i_b - uint64_ia) > excl_zone:
                        b = (T_B[uint64_ib:uint64_ib+uint64_m] - M_T[uint64_ib])/Σ_T[uint64_ib]
                        dist = lb_dist(L, U, b)
                        if dist < ρ[thread_idx, uint64_ia]:
                            ρ[thread_idx, uint64_ia] = dist
                            I[thread_idx, uint64_ia] = uint64_ib

                else:
                    uint64_ib = np.uint64(i_b)
                    b = (T_B[uint64_ib:uint64_ib+uint64_m] - M_T[uint64_ib])/Σ_T[uint64_ib]
                    dist = lb_dist(L, U, b)
                    if dist < ρ[thread_idx, uint64_ia]:
                        ρ[thread_idx, uint64_ia] = dist
                        I[thread_idx, uint64_ia] = uint64_ib

    for thread_idx in range(1, n_threads):
        mask = (ρ[thread_idx, :] < ρ[0, :]) & (I[thread_idx, :] != I[0, :])
        ρ[0, :][mask] = ρ[thread_idx, :][mask]
        I[0, :][mask] = I[thread_idx, :][mask]
        
    return np.sqrt(ρ[0, :]), I[0, :]
    

def lbKeogh_MP(T_A, m, r=0, T_B=None, indices_pruned=None, ignore_trivial=True):

    if T_B is None:
        T_B = T_A
        ignore_trivial = True

    if indices_pruned is None:
        indices_pruned = np.array([]).astype(np.int64)
    else:
        indices_pruned = indices_pruned.astype(np.int64)

    (T_A, μ_Q, σ_Q) = stumpy.core.preprocess(T_A, m)

    (T_B, M_T, Σ_T) = stumpy.core.preprocess(T_B, m)

    n_A = T_A.shape[0]
    n_B = T_B.shape[0]
    l = n_A - m + 1

    P, I = _LB_MP_Keogh(
        T_A,
        T_B,
        m,
        r,
        indices_pruned,
        M_T,
        μ_Q,
        Σ_T,
        σ_Q,
        ignore_trivial,
    )
    
    return P, I
#################################################################
# LB DSMP
#################################################################
def lbKeogh_DSMP(T_A, m, D, r=0, T_B=None, indices_pruned=None, ignore_trivial=True):
    l = len(T_A) - m + 1
    # PAA
    Ta_D = stumpy.paa(T_A, D).astype(np.float64)
    m_D = int(m/D)
    r_D = int(r/D) if r/D > 1 else 1

    if indices_pruned is None:
        indices_pruned_D = None
    else:
        indices_pruned_ = np.round(indices_pruned/D)
        indices_pruned_D = np.unique(indices_pruned_).astype(np.int64)

    if T_B is None:
        out_dslb = stumpy.lbKeogh_MP(Ta_D, m_D, r_D, indices_pruned=indices_pruned_D)
    else:
        Tb_D = stumpy.paa(T_B, D).astype(np.float64)
        out_dslb = stumpy.lbKeogh_MP(Ta_D, m_D, r_D, T_B=Tb_D,
                                    indices_pruned=indices_pruned_D, ignore_trivial=False)

    # Cost rescaling
    mp_D = out_dslb[0] * np.sqrt(D)
    n_mp_D = len(mp_D)
    x_mp = np.arange(n_mp_D)
    x_D = np.linspace(0, n_mp_D-1, l)

    # intrpolate to raw series length
    f_interpolate = interpolate.PchipInterpolator

    mask_inf = np.isinf(mp_D)
    if np.sum(mask_inf) > 0:
        fitted_curve = f_interpolate(x_mp[~mask_inf], mp_D[~mask_inf])
        mp_i = fitted_curve(x_D)
        mp_i[indices_pruned] = np.inf
    else:
        fitted_curve = f_interpolate(x_mp, mp_D)
        mp_i = fitted_curve(x_D)
    
    return mp_i
#################################################################
# PAA
#################################################################
def paa(ts, ratio):
    n_ts = len(ts)
    n_seg = n_ts // ratio
    ts_window = ts[:n_seg*ratio].reshape(n_seg, ratio)
    ts_paa = np.mean(ts_window, axis=1)
    return ts_paa

#################################################################
# dtw_dist
#################################################################
def dtw_dist(q, c, r):
    n_c = len(c)
    q_z = (q-np.mean(q))/np.std(q)

    m = len(q)
    c_win = np.lib.stride_tricks.sliding_window_view(c, m)
    c_m = np.mean(c_win, axis=1)
    c_std = np.std(c_win, axis=1)

    dtw_dist = []
    for i in range(n_c-m+1):
        c_z = (c[i:i+m]-c_m[i])/c_std[i]
        dtw_ = metrics.dtw(q_z, c_z, global_constraint="sakoe_chiba", sakoe_chiba_radius=r)
        dtw_dist.append(dtw_)

    return np.array(dtw_dist)