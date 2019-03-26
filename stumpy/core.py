import sys
import argparse
import numpy as np
import scipy.signal
import time

def get_pkg_name():  # pragma: no cover
    """
    Return package name
    """

    return __name__.split('.')[0]

def rolling_window(a, window):
    """
    Use strides to generate rolling/sliding windows for a numpy array

    Parameters
    ----------
    a : ndarray
        numpy array
    window : int
        Size of the rolling window

    Returns
    -------
    output : ndarray
        This will be a new view of the original input array.
    """

    shape = a.shape[:-1] + (a.shape[-1] - window + 1, window)
    strides = a.strides + (a.strides[-1],)

    return np.lib.stride_tricks.as_strided(a, shape=shape, strides=strides)

def z_norm(a, axis=0):
    """
    Calculate the z-normalized input array `a` by subtracting the mean and 
    dividing by the standard deviation along a given axis.

    Parameters
    ----------
    a : ndarray
        numpy array
    axis : int
        numpy axis

    Returns
    -------
    output : ndarray
        An ndarray with z-normalized values computed along a specified axis.
    """

    return (a - np.mean(a, axis, keepdims=True))/np.std(a, axis, keepdims=True)

def check_dtype(a, dtype=np.float):  # pragma: no cover
    """
    Check if the array type of `a` is of type specified by `dtype` parameter.

    Raises
    ------
    TypeError
        If the array type does not match `dtype`
    """

    if not issubclass(a.dtype.type, dtype):
        msg = '{} type expected but found {}'.format(dtype, a.dtype.type)
        raise TypeError(msg)

def are_arrays_equal(a, b):  # pragma: no cover
    """
    Check if two arrays are equal; first by comparing memory addresses, 
    and secondly by their values.

    Parameters
    ----------
    a : ndarray
        First argument.
    b : ndarray
        Second argument.

    Returns
    -------
    output : bool
        Returns `True` if the arrays are equal and `False` otherwise.
    """

    if id(a) == id(b):
        return True
    
    return np.array_equal(a, b)

def are_distances_too_small(a, threshold=10e-6):  # pragma: no cover
    """
    Check the distance values from a matrix profile.

    If the values are smaller than the threshold (i.e., less than 10e-6) then 
    it could suggest that this is a self-join.

    Parameters
    ----------
    a : ndarray
        First argument.
    threshold : float
        Minimum value in which to compare the matrix profile to

    Returns
    -------
    output : bool
        Returns `True` if the matrix profile distances are all below the 
        threshold and `False` if they are all above the threshold.
    """

    if a.mean() < threshold or np.all(a < threshold):
        return True

    return False

def sliding_dot_product(Q, T):
    """
    Use FFT convolution to calculate the sliding window dot product.

    Parameters
    ----------
    Q : ndarray
        Query array or subsequence
    T : ndarray
        Time series or sequence

    Returns
    -------
    output : ndarray
        Sliding dot product between `Q` and `T`.

    Notes
    -----
    Calculate the sliding dot product
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

def compute_mean_std(T, m):
    """
    Compute the sliding mean and standard deviation for the array `T` with 
    a window size of `m`

    Parameters
    ----------
    T : ndarray
        Time series or sequence
    m : int
        Window size

    Returns
    -------
    M_T : ndarray
        Sliding mean
    Σ_T : ndarray
        Sliding standard deviation

    Notes
    -----
    DOI: 10.1109/ICDM.2016.0179
    See Table II

    DOI: 10.1145/2020408.2020587
    See Page 2 and Equations 1, 2

    DOI: 10.1145/2339530.2339576
    See Page 4

    http://www.cs.unm.edu/~mueen/FastestSimilaritySearch.html

    Note that Mueen's algorithm has an off-by-one bug where the
    sum for the first subsequence is omitted and we fixed that!
    """
    n = T.shape[0]

    cumsum_T = np.empty(len(T)+1)
    np.cumsum(T, out=cumsum_T[1:])  # store output in cumsum_T[1:]
    cumsum_T[0] = 0

    cumsum_T_squared = np.empty(len(T)+1)
    np.cumsum(np.square(T), out=cumsum_T_squared[1:])
    cumsum_T_squared[0] = 0
    
    subseq_sum_T = cumsum_T[m:] - cumsum_T[:n-m+1]
    subseq_sum_T_squared = cumsum_T_squared[m:] - cumsum_T_squared[:n-m+1]
    M_T =  subseq_sum_T/m
    Σ_T = np.abs((subseq_sum_T_squared/m)-np.square(M_T))
    Σ_T = np.sqrt(Σ_T)

    return M_T, Σ_T

def calculate_distance_profile(m, QT, μ_Q, σ_Q, M_T, Σ_T):
    """
    Compute the distance profile

    Parameters
    ----------
    m : int
        Window size
    QT : ndarray
        Dot product between `Q` and `T`
    μ_Q : ndarray
        Mean of `Q`
    σ_Q : ndarray
        Standard deviation of `Q`
    M_T : ndarray
        Sliding mean of `T`
    Σ_T : ndarray
        Sliding standard deviation of `T`

    Returns
    -------
    output : ndarray
        Distance profile

    Notes
    -----
    DOI: 10.1109/ICDM.2016.0179
    See Equation on Page 4
    """

    denom = (m*σ_Q*Σ_T)
    denom[denom == 0] = 1E-10  # Avoid divide by zero
    D_squared = np.abs(2*m*(1.0-(QT-m*μ_Q*M_T)/denom))
    return np.sqrt(D_squared)

def mueen_calculate_distance_profile(Q, T):
    """
    Compute the mueen distance profile

    Parameters
    ----------
    Q : ndarray
        Query array or subsequence
    T : ndarray
        Time series or sequence

    Returns
    -------
    output : ndarray
        Distance profile

    Notes
    -----
    DOI: 10.1109/ICDM.2016.0179
    See Table II

    DOI: 10.1145/2020408.2020587
    See Page 2 and Equations 1, 2

    DOI: 10.1145/2339530.2339576
    See Page 4

    http://www.cs.unm.edu/~mueen/FastestSimilaritySearch.html

    Note that Mueen's algorithm has an off-by-one bug where the
    sum for the first subsequence is omitted and we fixed that!
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
    Σ_T_squared = np.abs(subseq_sum_T_squared/m-np.square(M_T))
    Σ_T = np.sqrt(Σ_T_squared)

    D = np.abs((subseq_sum_T_squared-2*subseq_sum_T*M_T+m*np.square(M_T))/Σ_T_squared-2*QT/Σ_T+m)
    return np.sqrt(D) 

def mass(Q, T, M_T=None, Σ_T=None):
    """
    Compute the distance profile using the MASS algorithm

    Parameters
    ----------
    Q : ndarray
        Query array or subsequence
    T : ndarray
        Time series or sequence
    M_T : ndarray (optional)
        Sliding mean of `T`
    Σ_T : ndarray (optional)
        Sliding standard deviation of `T`

    Returns
    -------
    output : ndarray
        Distance profile

    Notes
    -----
    DOI: 10.1109/ICDM.2016.0179
    See Table II

    Note that Q, T are not directly required to calculate D 

    Note: Unlike the Matrix Profile I paper, here, M_T, Σ_T can be calculated
    once for all subsequences of T and passed in so the redundancy is removed
    """

    QT = sliding_dot_product(Q,T)
    m = Q.shape[0]
    μ_Q, σ_Q = compute_mean_std(Q, m)
    if M_T is None or Σ_T is None:
        M_T, Σ_T = compute_mean_std(T, m)

    return calculate_distance_profile(m, QT, μ_Q, σ_Q, M_T, Σ_T)

convolution = scipy.signal.fftconvolve  # Swap for other convolution function
