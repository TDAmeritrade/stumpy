#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
from matrix_profile import core

def check_dtype(arr, dtype=np.float):
    """
    Check if array has correct dtype
    """
    if not issubclass(arr.dtype.type, dtype):
        msg = '{} type expected but found {}'.format(dtype, arr.dtype.type)
        raise TypeError(msg)

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
    check_dtype(T_A)
    check_dtype(T_B)
    subseq_T_B = core.rolling_window(T_B, m)

    # Create new function that calls MASS and handles idx, profile
    def subseq_mass(Q, T, trivial_idx=None):
        D = core.mass(Q, T)
        if trivial_idx is not None:
            D[trivial_idx] = np.inf
        # Element-wise Min
        I = np.argmin(D)
        P = D[I]
        return P, I

    # Check if T_A and T_B are the same (i.e., self-join)

    # Add exclusionary zone
    if ignore_trivial:
        out = [subseq_mass(subseq, T_A, i) for i, subseq in enumerate(subseq_T_B)]
    else:
        out = [subseq_mass(subseq, T_A) for subseq in subseq_T_B]
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