#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
from . import core, stamp
from numba import njit, prange
import logging

logger = logging.getLogger(__name__)

def _get_first_stump_profile(start, T_A, T_B, m, zone, M_T, Σ_T, μ_Q, σ_Q, 
                             ignore_trivial):
    """
    Given `start` set the corresponding `profile` and `indices`
    that correspond to `start-1`. Essentially, this is the value
    of the matrix profile for the first window of a given range.

    Note: If you have chunk that begins at 0  and ends at 100, 
    `start = 1`. However, the profile and index is assessed
    from (start-1, start-1+m).
    """

    # Handle first subsequence, add exclusionary zone
    if ignore_trivial:
        P, I = stamp.mass(T_B[start-1:start-1+m], T_A, M_T, Σ_T, 
                          start-1, zone)
        PL, IL = stamp.mass(T_B[start-1:start-1+m], T_A, M_T, Σ_T, 
                            start-1, zone, left=True)        
        PR, IR = stamp.mass(T_B[start-1:start-1+m], T_A, M_T, Σ_T, 
                            start-1, zone, right=True)
    else:
        P, I = stamp.mass(T_B[start-1:start-1+m], T_A, M_T, Σ_T)
        # No left and right matrix profile available
        IL = -1
        IR = -1

    return P, (I, IL, IR)


def _get_QT(start, T_A, T_B, m):
    """
    Given `start` return the corresponding QT and QT_first
    that correspond to (start-1, start-1+m).
    """

    QT = core.sliding_dot_product(T_B[start-1:start-1+m], T_A)
    QT_first = core.sliding_dot_product(T_A[:m], T_B)

    return QT, QT_first

@njit(parallel=True, fastmath=True)
def _calculate_squared_distance_profile(m, QT, μ_Q, σ_Q, M_T, Σ_T):
    """
    DOI: 10.1109/ICDM.2016.0179
    See Equation on Page 4
    """

    denom = (m*σ_Q*Σ_T)
    denom[denom == 0] = 1E-10  # Avoid divide by zero
    D_squared = np.abs(2*m*(1.0-(QT-m*μ_Q*M_T)/denom))
    
    return D_squared

@njit(parallel=True, fastmath=True) 
def _stump(T_A, T_B, m, stop, zone, 
           M_T, Σ_T, QT, QT_first, μ_Q, σ_Q, k, 
           ignore_trivial=False, start=1):
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

    Note that left and right matrix profiles are only available for self-joins.
    """

    QT_odd = QT.copy()
    QT_even = QT.copy()
    profile = np.empty((stop-start,))  # float64
    indices = np.empty((stop-start, 3))  # int64
    
    for i in range(start, stop):
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
            D = _calculate_squared_distance_profile(m, QT_even, μ_Q[i], σ_Q[i], M_T, Σ_T)
        else:
            QT_odd[0] = QT_first[i]
            D = _calculate_squared_distance_profile(m, QT_odd, μ_Q[i], σ_Q[i], M_T, Σ_T)

        if ignore_trivial:
            zone_start = max(0, i-zone)
            zone_stop = i+zone+1
            D[zone_start:zone_stop] = np.inf
        I = np.argmin(D)
        P = np.sqrt(D[I])

        # Get left and right matrix profiles for self-joins
        if ignore_trivial and i > 0:
            IL = np.argmin(D[:i])
            if zone_start <= IL < zone_stop:
                IL = -1
        else:
            IL = -1

        if ignore_trivial and i+1 < D.shape[0]:
            IR = i + 1 + np.argmin(D[i+1:])
            if zone_start <= IR < zone_stop:
                IR = -1
        else:
            IR = -1

        # Only a part of the profile/indices array are passed
        profile[i-start] = P
        indices[i-start] = I, IL, IR
    
    return profile, indices

def stump(T_A, T_B, m, ignore_trivial=False, dask_client=None):
    """
    DOI: 10.1109/ICDM.2016.0085
    See Table II

    This is a convenience wrapper around the parallelized `_stump` function

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

    Note that left and right matrix profiles are only available for self-joins.
    """

    core.check_dtype(T_A)
    core.check_dtype(T_B)

    if ignore_trivial == False and core.are_arrays_equal(T_A, T_B):  # pragma: no cover
        logger.warn("Arrays T_A, T_B are equal, which implies a self-join.")
        logger.warn("Try setting `ignore_trivial = True`.")
    
    n = T_B.shape[0]
    k = T_A.shape[0]-m+1
    l = n-m+1
    zone = int(np.ceil(m/4))  # See Definition 3 and Figure 3

    M_T, Σ_T = core.compute_mean_std(T_A, m)
    μ_Q, σ_Q = core.compute_mean_std(T_B, m)

    out = np.empty((l, 4), dtype=object)
    profile = np.empty((l,), dtype='float64')
    indices = np.empty((l, 3), dtype='int64')

    start = 1
    stop = l

    nworkers = 1
    if dask_client is not None:  # pragma: no cover
        nworkers = len(dask_client.ncores())

    step = l//nworkers
    for start in range(1, l, step):
        stop = min(l, start + step)

        profile[start-1], indices[start-1, :] = _get_first_stump_profile(start,
                                                    T_A, T_B, m, zone, M_T, 
                                                    Σ_T, μ_Q, σ_Q, 
                                                    ignore_trivial)

        QT, QT_first = _get_QT(start, T_A, T_B, m)

        profile[start:stop], indices[start:stop, :] = _stump(T_A, T_B, m, stop,
                                                             zone, M_T, Σ_T, 
                                                             QT, QT_first, μ_Q,
                                                             σ_Q, k, 
                                                             ignore_trivial, 
                                                             start)

    out[:, 0] = profile
    out[:, 1:4] = indices
    
    threshold = 10e-6
    if core.are_distances_too_small(out[:, 0], threshold=threshold):  # pragma: no cover
        logger.warn(f"A large number of values are smaller than {threshold}.")
        logger.warn("For a self-join, try setting `ignore_trivial = True`.")
        
    return out