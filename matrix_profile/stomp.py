#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
from matrix_profile import core, stamp

def _stomp(T_A, T_B, m, ignore_trivial=False):
    """
    DO NOT USE! Here for reference only.

    DOI: 10.1109/ICDM.2016.0085
    See Table II

    Timeseries, T_B, will be annotated with the distance location
    (or index) of all its subsequences in another times series, T_A.

    Return: For every subsequence, Q, in T_B, you will get a distance
    and index for the closest subsequence in T_A. Thus, the array
    returned will have length T_B.shape[0]-m+1

    Note that this implementation matches Table II but is here for
    reference purposes only and thus should not be used in practice 
    since it is not vectorized. See vectorized STOMP documentation and
    implementation below. 
    """
    core.check_dtype(T_A)
    core.check_dtype(T_B)
    n = T_B.shape[0]
    l = n-m+1
    zone = int(np.ceil(m/4))  # See Definition 3 and Figure 3
    M_T, Σ_T = core.compute_mean_std(T_A, m)
    QT = core.sliding_dot_product(T_B[:m], T_A)
    QT_first = core.sliding_dot_product(T_A[:m], T_B)

    μ_Q, σ_Q = core.compute_mean_std(T_B, m)

    out = [None] * l

    # Handle first subsequence, add exclusionary zone
    if ignore_trivial:
        out[0] = stamp.mass(T_B[:m], T_A, M_T, Σ_T, 0, zone)
    else:
        out[0] = stamp.mass(T_B[:m], T_A, M_T, Σ_T)

    k = T_A.shape[0]-m+1
    for i in range(1, l):
        for j in range(k-1,0,-1):
            QT[j] = QT[j-1] - T_B[i-1]*T_A[j-1] + T_B[i+m-1]*T_A[j+m-1]
        QT[0] =QT_first[i]
        D = core.calculate_distance_profile(m, QT, μ_Q[i], σ_Q[i], M_T, Σ_T)
        if ignore_trivial:
             start = max(0, i-zone)
             stop = i+zone+1
             D[start:stop] = np.inf
        I = np.argmin(D)
        P = D[I]
        out[i] = P, I
    out = np.array(out, dtype=object)
    
    return out

def stomp(T_A, T_B, m, ignore_trivial=False):
    """
    DOI: 10.1109/ICDM.2016.0085
    See Table II

    Timeseries, T_B, will be annotated with the distance location
    (or index) of all its subsequences in another times series, T_A.

    Return: For every subsequence, Q, in T_B, you will get a distance
    and index for the closest subsequence in T_A. Thus, the array
    returned will have length T_B.shape[0]-m+1

    Note: Unlike in the Table II where T_A.shape is expected to be equal 
    to T_B.shape, this implementation is generalized so that the shapes of 
    T_A and T_B can be different. In the case where T_A.shape == T_B.shape, 
    then our algorithm reduces down to the same algorithm found in Table II. 

    Additionally, unlike STAMP where the exclusion zone is m/2, the default 
    exclusion zone for STOMP is m/4 (See Definition 3 and Figure 3).
    """
    core.check_dtype(T_A)
    core.check_dtype(T_B)
    n = T_B.shape[0]
    l = n-m+1
    zone = int(np.ceil(m/4))  # See Definition 3 and Figure 3
    M_T, Σ_T = core.compute_mean_std(T_A, m)
    QT = core.sliding_dot_product(T_B[:m], T_A)
    QT_first = core.sliding_dot_product(T_A[:m], T_B)

    μ_Q, σ_Q = core.compute_mean_std(T_B, m)

    out = [None] * l

    # Handle first subsequence, add exclusionary zone
    if ignore_trivial:
        out[0] = stamp.mass(T_B[:m], T_A, M_T, Σ_T, 0, zone)
    else:
        out[0] = stamp.mass(T_B[:m], T_A, M_T, Σ_T)

    k = T_A.shape[0]-m+1
    # Expand and vectorize - T_B[i-1]*T_A[:k-1] + T_B[i-1+m]*T_A[-(k-1):]
    shift = -T_B[:l-1, np.newaxis]*T_A[:k-1, np.newaxis].T + T_B[m:l-1+m, np.newaxis]*T_A[-(k-1):, np.newaxis].T
    
    for i in range(1, l):
        #QT[1:] = QT[:k-1] - T_B[i-1]*T_A[:k-1] + T_B[i-1+m]*T_A[-(k-1):]
        QT[1:] = QT[:k-1] + shift[i-1]
        QT[0] = QT_first[i]
        D = core.calculate_distance_profile(m, QT, μ_Q[i], σ_Q[i], M_T, Σ_T)
        if ignore_trivial:
            start = max(0, i-zone)
            stop = i+zone+1
            D[start:stop] = np.inf
        I = np.argmin(D)
        P = D[I]
        out[i] = P, I
    out = np.array(out, dtype=object)
    
    return out


if __name__ == '__main__':
    core.check_python_version()
    #parser = get_parser()
    #args = parser.parse_args()
    #N = 17279800  # GPU-STOMP Comparison
    N = 12*24*365  # Every 5 minutes: 12 times an hour, 24 hours, 365 days
    M = 12*24
    # Select 50 random floats in range [-1000, 1000]
    T = np.random.uniform(-1000, 1000, [N])
    # Select 5 random floats in range [-1000, 1000]
    Q = np.random.uniform(-1000, 1000, [M])