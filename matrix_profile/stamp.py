#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
from . import core

def mass(Q, T, M_T, Σ_T, trivial_idx=None, excl_zone=0):
    D = core.mass(Q, T, M_T, Σ_T)
    if trivial_idx is not None:
        start = max(0, trivial_idx-excl_zone)
        stop = trivial_idx+excl_zone+1
        D[start:stop] = np.inf
    # Element-wise Min
    I = np.argmin(D)
    P = D[I]
    return P, I

def stamp(T_A, T_B, m, ignore_trivial=False):
    """
    DOI: 10.1109/ICDM.2016.0179
    See Table III

    Timeseries, T_B, will be annotated with the distance location
    (or index) of all its subsequences in another times series, T_A.

    Return: For every subsequence, Q, in T_B, you will get a distance
    and index for the closest subsequence in T_A. Thus, the array
    returned will have length T_B.shape[0]-m+1
    """
    core.check_dtype(T_A)
    core.check_dtype(T_B)
    subseq_T_B = core.rolling_window(T_B, m)
    zone = int(np.ceil(m/2))
    M_T, Σ_T = core.compute_mean_std(T_A, m)

    # Add exclusionary zone
    if ignore_trivial:
        out = [mass(subseq, T_A, M_T, Σ_T, i, zone) for i, subseq in enumerate(subseq_T_B)]
    else:
        out = [mass(subseq, T_A, M_T, Σ_T) for subseq in subseq_T_B]
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
