#!/usr/bin/env python
# -*- coding: utf-8 -*-

import sys
import argparse
import numpy as np
import scipy.signal
import time

def check_python_version():
    if (sys.version_info < (3, 0)):
        raise Exception('Matrix Profile is only compatible with python3.x')

def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('ts_file', help='Time series input file')
    parser.add_argument('subseq_length', help='Subsequence length', type=int)

    return parser

def rolling_window(a, window):
    shape = a.shape[:-1] + (a.shape[-1] - window + 1, window)
    strides = a.strides + (a.strides[-1],)
    return np.lib.stride_tricks.as_strided(a, shape=shape, strides=strides)

def z_norm(x, axis=0):
    return (x - np.mean(x, axis, keepdims=True))/np.std(x, axis, keepdims=True)

def timeit(func):
    """
    Timing decorator
    """
    def timed(*args, **kw):
        ts = time.time()
        result = func(*args, **kw)
        te = time.time()

        print('{} sec {}'.format(te-ts, func.__name__))
        return result

    return timed

def sliding_dot_product(Q, T):
    """
    DOI: 10.1109/ICDM.2016.0179
    See Table I, Figure 4

    Following the inverse FFT, Fig. 4 states that only cells [m-1:n] 
    contain valid dot products

    Padding is done automatically in fftconvolve step
    """
    n = T.shape[0]
    m = Q.shape[0]
    Qr = np.flipud(Q)  # Reverse/flip Q
    QT = convolution(Qr, T)
    return QT.real[m-1:n]

def calculate_distance_profile(Q, T):
    """
    DOI: 10.1109/ICDM.2016.0179
    See Table II

    DOI: 10.1145/2020408.2020587
    See Page 2 and Equations 1, 2

    DOI: 10.1145/2339530.2339576
    See Page 4

    http://www.cs.unm.edu/~mueen/FastestSimilaritySearch.html

    Note that Mueen's algorithm has an off-by-one bug where the
    sum for the first subsequence is omitted and we fixed that!

    DOI: 10.1109/ICDM.2016.0179
    Note that distance equation on Page 4 leads to catostrophic 
    cancellation! So we are using Mueen's code
    """
    n = T.shape[0]
    m = Q.shape[0]

    μ_Q = np.mean(Q, keepdims=True)
    σ_Q = np.std(Q, keepdims=True)
    Q_norm = (Q-μ_Q)/σ_Q

    QT = sliding_dot_product(Q_norm, T)

    cumsum_T = np.empty(len(T)+1)  # Add one element, fix off-by-one
    np.cumsum(T, out=cumsum_T[1:])  # store output in cumsum_T[1:]
    cumsum_T[0] = 0

    cumsum_T_squared = np.empty(len(T)+1)
    np.cumsum(np.square(T), out=cumsum_T_squared[1:])
    cumsum_T_squared[0] = 0
    
    subseq_sum_T = cumsum_T[m:] - cumsum_T[:n-m+1]
    subseq_sum_T_squared = cumsum_T_squared[m:] - cumsum_T_squared[:n-m+1]
    M_T =  subseq_sum_T/m
    Σ_T_squared = subseq_sum_T_squared/m-np.square(M_T)
    Σ_T = np.sqrt(Σ_T_squared)

    D = np.abs((subseq_sum_T_squared-2*subseq_sum_T*M_T+m*np.square(M_T))/Σ_T_squared - 2*QT/Σ_T + m)
    
    return np.sqrt(D) 

def mass(Q, T):
    """
    DOI: 10.1109/ICDM.2016.0179
    See Table II

    """

    return calculate_distance_profile(Q, T)

convolution = scipy.signal.fftconvolve  # Swap for other convolution function

if __name__ == '__main__':
    check_python_version()
    #parser = get_parser()
    #args = parser.parse_args()
    N = 17279800  # GPU-STOMP Comparison
    # Select 50 random floats in range [-1000, 1000]
    T = np.random.uniform(-1000, 1000, [N])
    # Select 5 random floats in range [-1000, 1000]
    Q = np.random.uniform(-1000, 1000, [2000])
    mass(Q, T)