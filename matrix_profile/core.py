#!/usr/bin/env python

import sys
import argparse
import numpy as np
from scipy.fftpack import fft, ifft
import pyfftw
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

def rolling_window(a, window):
    shape = a.shape[:-1] + (a.shape[-1] - window + 1, window)
    strides = a.strides + (a.strides[-1],)
    return np.lib.stride_tricks.as_strided(a, shape=shape, strides=strides)

@timeit
def sliding_dot_product(Q, T):
    """
    DOI: 10.1109/ICDM.2016.0179
    See Table I, Figure 4

    Following the inverse FFT, Fig. 4 states that only cells [m-1:n] 
    contain valid dot products
    """
    n = len(T)
    m = len(Q)
    pad_width = (0, n)  # prepend zero 0's and append n 0's to T 
    Ta = np.pad(T, (0, n), mode='constant', constant_values=0)
    Qr = np.flipud(Q)  # Reverse/flip Q
    pad_width = (0, 2 * n - m)  # prepend zero 0's and append n 0's to Qr
    Qra = np.pad(Qr, pad_width, mode='constant', constant_values=0)
    Qraf = np.fft.fft(Qra)
    Taf = np.fft.fft(Ta)
    QT = np.fft.ifft(np.multiply(Qraf, Taf))

    return QT.real[m-1:n]

@timeit
def scipy_sliding_dot_product(Q, T):
    """
    DOI: 10.1109/ICDM.2016.0179
    See Table I, Figure 4

    Following the inverse FFT, Fig. 4 states that only cells [m-1:n] 
    contain valid dot products
    """
    n = len(T)
    m = len(Q)
    pad_width = (0, n)  # prepend zero 0's and append n 0's to T 
    Ta = np.pad(T, (0, n), mode='constant', constant_values=0)
    Qr = np.flipud(Q)  # Reverse/flip Q
    pad_width = (0, 2 * n - m)  # prepend zero 0's and append n 0's to Qr
    Qra = np.pad(Qr, pad_width, mode='constant', constant_values=0)
    Qraf = fft(Qra)
    Taf = fft(Ta)
    QT = ifft(np.multiply(Qraf, Taf))

    return QT.real[m-1:n]

@timeit
def stride_rolling_window_dot_product(Q, T):
    return np.dot(rolling_window(T, len(Q)), Q)    


if __name__ == '__main__':
    check_python_version()
    parser = get_parser()
    args = parser.parse_args()
    print(args)
    N = 100000000
    # Select 50 random floats in range [-1000, 1000]
    T = np.random.uniform(-1000, 1000, [N])
    # Select 5 random floats in range [-1000, 1000]
    Q = np.random.uniform(-1000, 1000, [5])
    out = sliding_dot_product(Q, T)
    out = stride_rolling_window_dot_product(Q, T)
    out = scipy_sliding_dot_product(Q, T)

