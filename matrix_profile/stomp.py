#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
from matrix_profile import core, stamp

def stomp(T_A, T_B, m, ignore_trivial=False):
    """
    DOI: 10.1109/ICDM.2016.0085
    See Table II

    Timeseries, T_B, will be annotated with the distance location
    (or index) of all its subsequences in another times series, T_A.

    Return: For every subsequence, Q, in T_B, you will get a distance
    and index for the closest subsequence in T_A. Thus, the array
    returned will have length T_B.shape[0]-m+1

    Note that the initial QT only equals QT_first iff T_A == T_B 
    and, subsequently, T_A[:m] == T_B[:m].
    Otherwise, QT_first is calculated from T_A[:m] and T_B rather
    than T_B[:m] and T_A.
    """
    core.check_dtype(T_A)
    core.check_dtype(T_B)
    n = T_B.shape[0]
    l = n-m+1
    zone = int(np.ceil(m/4))  # See Definition 3 and Figure 3
    M_T, Σ_T = core.compute_mean_std(T_B, m)
    QT = core.sliding_dot_product(T_B[:m], T_A)
    QT_first = core.sliding_dot_product(T_A[:m], T_B)

    μ_Q, σ_Q = core.compute_mean_std(T_A, m)

    out = [None] * l

    # Handle first subsequence, add exclusionary zone
    if ignore_trivial:
        out[0] = stamp.mass(T_A[:m], T_B, M_T, Σ_T, 0, zone)
    else:
        out[0] = stamp.mass(T_A[:m], T_B, M_T, Σ_T)

    k = T_A.shape[0]-m+1
    for i in range(1, l):
        QT[1:] = QT[:k-1] - T_B[i-1]*T_A[:k-1] + T_B[i-1+m]*T_A[-(k-1):]
        QT[0] = QT_first[i]
        print(M_T.shape, Σ_T.shape, μ_Q[i].shape, σ_Q[i].shape, QT.shape)
        D = core.calculate_distance_profile(m, QT, μ_Q[i], σ_Q[i], M_T, Σ_T)
        I = np.argmin(D)
        P = D[I]
        out[i] = P, I

    # k = T_A.shape[0]-m+1
    # for i in range(1, l):
    #     for j in range(k-1,0,-1):
    #         # Ensure QT calculation works for any T_A, T_B of different lengths
    #         QT[j] = QT[j-1] - T_B[i-1]*T_A[j-1] + T_B[i+m-1]*T_A[j+m-1]
    #     QT[0] =QT_first[i]
    #     print(QT)
    #     #D = core.calculate_distance_profile(m, QT, np.array([μ_Q[i]]), np.array([σ_Q[i]]), M_T, Σ_T)
    #     #I = np.argmin(D)
    #     #P = D[I]
    #     #out[i] = P, I
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