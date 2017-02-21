#!/usr/bin/env python

import sys
import argparse
import pyfftw
import numpy as np
from scipy.fftpack import rfft, ifft
import scipy.signal
import time
import multiprocessing

nthread = multiprocessing.cpu_count()

def check_python_version():
    if (sys.version_info < (3, 0)):
        raise Exception('Matrix Profile is only compatible with python3.x')

def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('ts_file', help='Time series input file')
    parser.add_argument('subseq_length', help='Subsequence length', type=int)

    return parser

def next_nearest_pow_of_two(target):
    if target > 1:
        for i in range(1, int(target)):
            if (2 ** i >= target):
                return 2 ** i
    else:
        return 1

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
def numpy_sliding_dot_product(Q, T):
    """
    DOI: 10.1109/ICDM.2016.0179
    See Table I, Figure 4

    Following the inverse FFT, Fig. 4 states that only cells [m-1:n] 
    contain valid dot products
    """
    n = len(T)
    m = len(Q)
    pad_width = (0, n)  # prepend zero 0's and append n 0's to T 
    Ta = np.pad(T, pad_width, mode='constant', constant_values=0)
    Qr = np.flipud(Q)  # Reverse/flip Q
    pad_width = (0, 2 * n - m)  # prepend zero 0's and append n 0's to Qr
    Qra = np.pad(Qr, pad_width, mode='constant', constant_values=0)

    return np.convolve(Qra, Ta).real[m-1:n]

@timeit
def scipy_fftconvolve_sliding_dot_product(Q, T):
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
    QT = scipy.signal.fftconvolve(Qr, T)
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
    Qraf = rfft(Qra)
    Taf = rfft(Ta)
    QT = ifft(np.multiply(Qraf, Taf))

    return QT.real[m-1:n]

@timeit
def fftw_sliding_dot_product(Q, T, zeros, fftw_obj, ifftw_obj):
    """
    DOI: 10.1109/ICDM.2016.0179
    See Table I, Figure 4

    Following the inverse FFT, Fig. 4 states that only cells [m-1:n] 
    contain valid dot products
    """

    n = len(T)
    m = len(Q)

    zeros.fill(0.0)
    zeros[:m] = np.flipud(Q)
    Qraf = fftw_obj.__call__(input_array=zeros)

    zeros[:n] = T
    Taf = fftw_obj.__call__(input_array=zeros)

    #a = pyfftw.empty_aligned(4, dtype='complex128')
    #a[:] = [1,2,3,4]
    #fft_a = pyfftw.interfaces.numpy_fft.fft(a, planning_timelimit=5) 
    #fft_Qra = pyfftw.builders.fft(a, overwrite_input=True, planner_effort='FFTW_MEASURE', threads=1, auto_align_input=False, auto_contiguous=False, avoid_copy=True)  # Create pyfftw.builders object
    #fft_Ta = pyfftw.builders.fft(Ta)  # Create pyfftw.builders object

    #Qraf = fft_Qra()
    #Taf = fft_Ta()

    #ifft_Qraf_Taf = pyfftw.builders.ifft(np.multiply(Qraf, Taf))
    #QT = ifft_Qraf_Taf()

    #return QT.real[m-1:n]

@timeit
def builders_sliding_dot_product(Q, T, fftw_obj, inp):
    """
    """
    n = len(T)
    inp[:n] = T
    x = fftw_obj(inp) # returns the output

    return

@timeit
def stride_rolling_window_dot_product(Q, T):
    return np.dot(rolling_window(T, len(Q)), Q)    

def get_fftw_objects(T, nthread=1, ptime=30):
    """
    """
    n = len(T)
    n_pad = next_nearest_pow_of_two(2*n)
    inp1_array = pyfftw.empty_aligned(n_pad, dtype='float32')
    out1_array = pyfftw.empty_aligned(n_pad//2 + 1, dtype='complex64')
    fftw_object = pyfftw.FFTW(inp1_array, out1_array, flags=('FFTW_ESTIMATE',), threads=nthread, planning_timelimit=ptime)

    out2_array = pyfftw.empty_aligned(n_pad, dtype='float32')
    inp2_array = pyfftw.empty_aligned(n_pad//2 + 1, dtype='complex64')
    ifftw_object = pyfftw.FFTW(inp2_array, out2_array, flags=('FFTW_ESTIMATE',), direction='FFTW_BACKWARD', threads=nthread, planning_timelimit=ptime)    

    return fftw_object, ifftw_object

@timeit
def interface_sliding_dot_product(Q, T):
    """
    """
    #pyfftw.interfaces.scipy_fftpack.rfft(Q)
    pyfftw.interfaces.scipy_fftpack.rfft(T, planner_effort='FFTW_ESTIMATE')
    return

if __name__ == '__main__':
    check_python_version()
    parser = get_parser()
    args = parser.parse_args()
    print(args)
    #N = 20000000
    N = 17279800  # GPU-STOMP Comparison
    #N = 2**25
    #N = 100000
    print("N = {}".format(N))
    # Select 50 random floats in range [-1000, 1000]
    T = np.random.uniform(-1000, 1000, [N])
    # Select 5 random floats in range [-1000, 1000]
    Q = np.random.uniform(-1000, 1000, [2000])
    #out = numpy_sliding_dot_product(Q, T)
    out = scipy_fftconvolve_sliding_dot_product(Q, T)
    #N = 2**17
    #print("N = {}".format(N))
    #T = np.random.uniform(-1000, 1000, [N])
    #Q = np.random.uniform(-1000, 1000, [256])
    #out = scipy_fftconvolve_sliding_dot_product(Q, T)
    #out = stride_rolling_window_dot_product(Q, T)
    #out = scipy_sliding_dot_product(Q, T)
    #fftw_object, ifftw_object = get_fftw_objects(T, nthread)

    n = len(T)
    m = len(Q)
    padding = next_nearest_pow_of_two(2*n)

    #pyfftw.interfaces.cache.enable()
    #pyfftw.interfaces.cache.set_keepalive_time(2)
    #inp = pyfftw.empty_aligned(padding, dtype='float32')
    #inp[:n] = T
    #interface_sliding_dot_product(Q, inp)
    #interface_sliding_dot_product(Q, inp)

    #inp = pyfftw.empty_aligned(padding, dtype='float32')
    #fftw_obj = pyfftw.builders.rfft(inp, planner_effort='FFTW_MEASURE', threads=nthread, auto_align_input=True, overwrite_input=True, auto_contiguous=True, avoid_copy=True)
    #builders_sliding_dot_product(Q, T, fftw_obj, inp)
    #builders_sliding_dot_product(Q, T, fftw_obj, inp)

    print(padding)
    nthread = 1
    inp1_array = pyfftw.empty_aligned(padding, dtype='float32')
    out1_array = pyfftw.empty_aligned(padding//2 + 1, dtype='complex64')
    fftw_object = pyfftw.FFTW(inp1_array, out1_array, flags=('FFTW_ESTIMATE',), threads=nthread)

    out2_array = pyfftw.empty_aligned(padding, dtype='float32')
    inp2_array = pyfftw.empty_aligned(padding//2 + 1, dtype='complex64')
    ifftw_object = pyfftw.FFTW(inp2_array, out2_array, flags=('FFTW_ESTIMATE',), direction='FFTW_BACKWARD', threads=nthread)

    zeros = pyfftw.zeros_aligned(padding, dtype='float32')

    out = fftw_sliding_dot_product(Q, T, zeros, fftw_object, ifftw_object)
    out = fftw_sliding_dot_product(Q, T, zeros, fftw_object, ifftw_object)
