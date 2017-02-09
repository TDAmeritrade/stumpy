#!/usr/bin/env python

import sys
import argparse
import numpy as np

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


if __name__ == '__main__':
    check_python_version()
    parser = get_parser()
    args = parser.parse_args()
    print(args)