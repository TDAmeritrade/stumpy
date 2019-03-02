#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
from . import core, stamp
from numba import njit, prange
import logging

logger = logging.getLogger(__name__)

@njit()
def calculate_distance_profile(m, QT, μ_Q, σ_Q, M_T, Σ_T):
    """
    DOI: 10.1109/ICDM.2016.0179
    See Equation on Page 4
    """

    m_μ_Q = m*μ_Q
    m_σ_Q = m*σ_Q
    m_2 = 2.0*m

    denom = (m*σ_Q*Σ_T)
    denom[denom == 0] = 1E-10  # Avoid divide by zero
    D_squared = np.abs(2*m*(1.0-(QT-m_μ_Q*M_T)/denom))
    
    return np.sqrt(D_squared)

def _stump(T_A, T_B, m, profile, indices, l, zone, 
           M_T, Σ_T, QT, QT_first, μ_Q, σ_Q, k, ignore_trivial=False):
    """
    DOI: 10.1109/ICDM.2016.0085
    See Table II

    Timeseries, T_B, will be annotated with the distance location
    (or index) of all its subsequences in another times series, T_A.

    Return: For every subsequence, Q, in T_B, you will get a distance
    and index for the closest subsequence in T_A. Thus, the array
    returned will have length T_B.shape[0]-m+1. Additionally, the 
    left and right matrix profiles are also returned.

    Note: Unlike in the Table II where T_A.shape is expected to be equal 
    to T_B.shape, this implementation is generalized so that the shapes of 
    T_A and T_B can be different. In the case where T_A.shape == T_B.shape, 
    then our algorithm reduces down to the same algorithm found in Table II. 

    Additionally, unlike STAMP where the exclusion zone is m/2, the default 
    exclusion zone for STOMP is m/4 (See Definition 3 and Figure 3).

    For self-joins, set `ignore_trivial = True` in order to avoid the 
    trivial match. 

    Note that the Numba jit compilation happens within the wrapper function
    `stump`
    """

    QT_odd = QT.copy()
    QT_even = QT.copy()
    tmp_indices = np.arange(k)
    
    for i in range(1, l):
        # Numba's prange requires incrementing a range by 1 so replace
        # `for j in range(k-1,0,-1)` with its incrementing compliment
        for rev_j in prange(1, k):
            j = k - rev_j
            # GPU Stomp Parallel Implementation with Numba
            # DOI: 10.1109/ICDM.2016.0085
            # See Figure 5
            if i % 2 == 0:
                # Even
                QT_even[j] = QT_odd[j-1] - T_B[i-1]*T_A[j-1] + T_B[i+m-1]*T_A[j+m-1]
            else:
                # Odd
                QT_odd[j] = QT_even[j-1] - T_B[i-1]*T_A[j-1] + T_B[i+m-1]*T_A[j+m-1]

        if i % 2 == 0:
            QT_even[0] = QT_first[i]
            D = calculate_distance_profile(m, QT_even, μ_Q[i], σ_Q[i], M_T, Σ_T)
        else:
            QT_odd[0] = QT_first[i]
            D = calculate_distance_profile(m, QT_odd, μ_Q[i], σ_Q[i], M_T, Σ_T)

        if ignore_trivial:
            start = max(0, i-zone)
            stop = i+zone+1
            D[start:stop] = np.inf
        I = np.argmin(D)
        P = D[I]

        # Get left and right matrix profiles
        if i > 0:
            left_subset_idx = np.argmin(D[:i])
            IL = tmp_indices[:i][left_subset_idx]
        else:
            IL = -1

        if i+1 < D.shape[0]:
            right_subset_idx = np.argmin(D[i+1:])
            IR = tmp_indices[i+1:][right_subset_idx]
        else:
            IR = -1

        profile[i] = P
        indices[i] = I, IL, IR
    
    return

def stump(T_A, T_B, m, ignore_trivial=False, parallel=False):
    """
    """

    global _stump

    core.check_dtype(T_A)
    core.check_dtype(T_B)

    if ignore_trivial == False and core.are_arrays_equal(T_A, T_B):
        logger.warn("Arrays T_A, T_B are equal, which implies a self-join.")
        logger.warn("Try setting `ignore_trivial = True`.")
    
    n = T_B.shape[0]
    l = n-m+1
    zone = int(np.ceil(m/4))  # See Definition 3 and Figure 3

    M_T, Σ_T = core.compute_mean_std(T_A, m)
    QT = core.sliding_dot_product(T_B[:m], T_A)
    QT_first = core.sliding_dot_product(T_A[:m], T_B)

    μ_Q, σ_Q = core.compute_mean_std(T_B, m)

    out = np.empty((l, 4), dtype=object)
    profile = np.empty((l,), dtype='float64')
    indices = np.empty((l,3), dtype='int64')

    # Handle first subsequence, add exclusionary zone
    if ignore_trivial:
        P, I = stamp.mass(T_B[:m], T_A, M_T, Σ_T, 0, zone)
    else:
        P, I = stamp.mass(T_B[:m], T_A, M_T, Σ_T)
    profile[0] = P
    indices[0] = I , -1, I
    
    k = T_A.shape[0]-m+1

    jitted_stump = njit(_stump, parallel=parallel)

    jitted_stump(T_A, T_B, m, profile, indices, l, zone, 
                 M_T, Σ_T, QT, QT_first, μ_Q, σ_Q, k,
                 ignore_trivial)
    
    out[:, 0] = profile
    out[:, 1:4] = indices
    
    threshold = 10e-6
    if core.are_distances_too_small(out[:, 0], threshold=threshold):
        logger.warn(f"A large number of values are smaller than {threshold}.")
        logger.warn("For a self-join, try setting `ignore_trivial = True`.")
        
    return out

if __name__ == '__main__':
    core.check_python_version()
    #parser = get_parser()
    #args = parser.parse_args()
    #N = 17279800  # GPU-STOMP Comparison
    #N = 12*24*365  # Every 5 minutes: 12 times an hour, 24 hours, 365 days
    #M = 12*24
    N = 1000
    M = 100
    # Select 50 random floats in range [-1000, 1000]
    T = np.random.uniform(-1000, 1000, [N])
    # Select 5 random floats in range [-1000, 1000]
    Q = np.random.uniform(-1000, 1000, [M])
    stump(Q, T, 10)
