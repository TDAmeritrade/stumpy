#!/usr/bin/env python

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
    n = len(T)
    m = len(Q)
    Qr = np.flipud(Q)  # Reverse/flip Q
    QT = convolution(Qr, T)
    return QT.real[m-1:n]

convolution = scipy.signal.fftconvolve  # Swap for other convolution function

if __name__ == '__main__':
    check_python_version()
    #parser = get_parser()
    #args = parser.parse_args()
    N = 17279800  # GPU-STOMP Comparison
    #print("N = {}".format(N))
    # Select 50 random floats in range [-1000, 1000]
    T = np.random.uniform(-1000, 1000, [N])
    # Select 5 random floats in range [-1000, 1000]
    Q = np.random.uniform(-1000, 1000, [2000])
    #out = numpy_sliding_dot_product(Q, T)
    out = sliding_dot_product(Q, T)