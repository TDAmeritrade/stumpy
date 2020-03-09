# STUMPY
# Copyright 2019 TD Ameritrade. Released under the terms of the 3-Clause BSD license.  # noqa: E501
# STUMPY is a trademark of TD Ameritrade IP Company, Inc. All rights reserved.

import numpy as np
from numba import njit, prange
import scipy.signal
import tempfile
import math

try:
    from numba.cuda.cudadrv.driver import _raise_driver_not_found
except ImportError:
    pass


def driver_not_found(*args, **kwargs):  # pragma: no cover
    """
    Helper function to raise CudaSupportError driver not found error
    """

    _raise_driver_not_found()


def get_pkg_name():  # pragma: no cover
    """
    Return package name
    """

    return __name__.split(".")[0]


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
    std = np.std(a, axis, keepdims=True)
    std[std == 0] = 1

    return (a - np.mean(a, axis, keepdims=True)) / std


def check_nan(a):  # pragma: no cover
    """
    Check if the array contains NaNs

    Raises
    ------
    ValueError
        If the array contains a NaN
    """

    if np.any(np.isnan(a)):
        msg = f"Input array contains one or more NaNs"
        raise ValueError(msg)

    return


def check_dtype(a, dtype=np.floating):  # pragma: no cover
    """
    Check if the array type of `a` is of type specified by `dtype` parameter.

    Raises
    ------
    TypeError
        If the array type does not match `dtype`
    """

    if not np.issubdtype(a.dtype, dtype):
        msg = f"{dtype} type expected but found {a.dtype}"
        raise TypeError(msg)

    return True


def transpose_dataframe(a):  # pragma: no cover
    """
    Check if the input is a column-wise Pandas `DataFrame`. If `True`, return a
    transpose dataframe since stumpy assumes that each row represents data from a
    different dimension while each column represents data from the same dimension.
    If `False`, return `a` unchanged. Pandas `Series` do not need to be transposed.

    Note that this function has zero dependency on Pandas (not even a soft dependency).

    Parameters
    ----------
    a : ndarray
        First argument.

    Returns
    -------
    output : a
        If a is a Pandas `DataFrame` then return `a.T`. Otherwise, return `a`
    """

    if type(a).__name__ == "DataFrame":
        return a.T

    return a


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

    if a.shape != b.shape:
        return False

    return ((a == b) | (np.isnan(a) & np.isnan(b))).all()


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


def check_window_size(m):
    """

    Parameters
    ----------
    m : int
        Window size

    Returns
    -------
    None
    """
    if m <= 2:
        raise ValueError(
            "All window sizes must be greater than or equal to three",
            """A window size that is less than or equal to two is meaningless when
            it comes to computing the z-normalized Euclidean distance. In the case of
            `m=1` produces a standard deviation of zero. In the case of `m=2`, both
            the mean and standard deviation for any given subsequence are identical
            and so the z-normalization for any sequence will either be [-1., 1.] or
            [1., -1.]. Thus, the z-normalized Euclidean distance will be (very likely)
            zero between any subsequence and its nearest neighbor (assuming that the
            time series is large enough to contain both scenarios).
            """,
        )


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

    `DOI: 10.1109/ICDM.2016.0179 \
    <https://www.cs.ucr.edu/~eamonn/PID4481997_extend_Matrix%20Profile_I.pdf>`__

    See Table I, Figure 4

    Following the inverse FFT, Fig. 4 states that only cells [m-1:n]
    contain valid dot products

    Padding is done automatically in fftconvolve step
    """

    n = T.shape[0]
    m = Q.shape[0]
    Qr = np.flipud(Q)  # Reverse/flip Q
    QT = convolution(Qr, T)

    return QT.real[m - 1 : n]


def compute_mean_std(T, m, num_chunks=1, max_iter=10):
    """
    Compute the sliding mean and standard deviation for the array `T` with
    a window size of `m` by splitting up T in `n_chunks`

    Parameters
    ----------
    T : ndarray
        Time series or sequence

    m : int
        Window size

    num_chunks : int
        Number of chunks to use for the first iteration. If a Memory Error is raised,
        this number is doubled and the computation tried again.

    max_iter : int
        Maximum number of iterations

    Returns
    -------
    M_T : ndarray
        Sliding mean. All nan values are replaced with np.inf

    Σ_T : ndarray
        Sliding standard deviation

    Notes
    -----
    `DOI: 10.1109/ICDM.2016.0179 \
    <https://www.cs.ucr.edu/~eamonn/PID4481997_extend_Matrix%20Profile_I.pdf>`__

    See Table II

    DOI: 10.1145/2020408.2020587

    See Page 2 and Equations 1, 2

    DOI: 10.1145/2339530.2339576

    See Page 4

    http://www.cs.unm.edu/~mueen/FastestSimilaritySearch.html

    Note that Mueen's algorithm has an off-by-one bug where the
    sum for the first subsequence is omitted and we fixed that!
    """
    if T.ndim > 2:  # pragma nocover
        raise ValueError("T has to be one or two dimensional!")

    for iteration in range(max_iter):
        try:
            chunk_size = math.ceil((T.shape[-1] + 1) / num_chunks)

            mean_chunks = []
            std_chunks = []
            for chunk in range(num_chunks):
                start = chunk * chunk_size
                stop = min(start + chunk_size + m - 1, T.shape[-1])

                tmp_mean = np.mean(rolling_window(T[start:stop], m), axis=T.ndim)
                mean_chunks.append(tmp_mean)
                tmp_std = np.nanstd(rolling_window(T[start:stop], m), axis=T.ndim)
                std_chunks.append(tmp_std)

            M_T = np.hstack(mean_chunks)
            Σ_T = np.hstack(std_chunks)
            break

        except MemoryError:  # pragma nocover
            num_chunks *= 2

    if iteration < max_iter - 1:
        M_T[np.isnan(M_T)] = np.inf
        Σ_T[np.isnan(Σ_T)] = 0

        return M_T, Σ_T
    else:  # pragma nocover
        raise MemoryError(
            "Could not calculate mean and standard deviation. "
            "Increase the number of chunks or maximal iterations."
        )


@njit(parallel=True, fastmath=True)
def _calculate_squared_distance_profile(m, QT, μ_Q, σ_Q, M_T, Σ_T):
    """
    Compute the squared distance profile

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
        Distance profile squared

    Notes
    -----
    `DOI: 10.1109/ICDM.2016.0179 \
    <https://www.cs.ucr.edu/~eamonn/PID4481997_extend_Matrix%20Profile_I.pdf>`__

    See Equation on Page 4
    """

    k = M_T.shape[0]
    D_squared = np.empty(k)

    threshold = 1e-10
    for i in prange(k):
        if np.isinf(M_T[i]) or np.isinf(μ_Q):
            D_squared[i] = np.inf
        else:
            if σ_Q < threshold or Σ_T[i] < threshold:
                D_squared[i] = m
            else:
                denom = m * σ_Q * Σ_T[i]
                if np.abs(denom) < threshold:  # pragma nocover
                    denom = threshold
                D_squared[i] = np.abs(
                    2 * m * (1.0 - (QT[i] - m * μ_Q * M_T[i]) / denom)
                )

            if σ_Q < threshold and Σ_T[i] < threshold:
                D_squared[i] = 0

    return D_squared


@njit(parallel=True, fastmath=True)
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
    `DOI: 10.1109/ICDM.2016.0179 \
    <https://www.cs.ucr.edu/~eamonn/PID4481997_extend_Matrix%20Profile_I.pdf>`__

    See Equation on Page 4
    """

    D_squared = _calculate_squared_distance_profile(m, QT, μ_Q, σ_Q, M_T, Σ_T)

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
    `DOI: 10.1109/ICDM.2016.0179 \
    <https://www.cs.ucr.edu/~eamonn/PID4481997_extend_Matrix%20Profile_I.pdf>`__

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
    Q_norm = (Q - μ_Q) / σ_Q
    QT = sliding_dot_product(Q_norm, T)

    cumsum_T = np.empty(len(T) + 1)  # Add one element, fix off-by-one
    np.cumsum(T, out=cumsum_T[1:])  # store output in cumsum_T[1:]
    cumsum_T[0] = 0

    cumsum_T_squared = np.empty(len(T) + 1)
    np.cumsum(np.square(T), out=cumsum_T_squared[1:])
    cumsum_T_squared[0] = 0

    subseq_sum_T = cumsum_T[m:] - cumsum_T[: n - m + 1]
    subseq_sum_T_squared = cumsum_T_squared[m:] - cumsum_T_squared[: n - m + 1]
    M_T = subseq_sum_T / m
    Σ_T_squared = np.abs(subseq_sum_T_squared / m - np.square(M_T))
    Σ_T = np.sqrt(Σ_T_squared)

    D = np.abs(
        (subseq_sum_T_squared - 2 * subseq_sum_T * M_T + m * np.square(M_T))
        / Σ_T_squared
        - 2 * QT / Σ_T
        + m
    )
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
    `DOI: 10.1109/ICDM.2016.0179 \
    <https://www.cs.ucr.edu/~eamonn/PID4481997_extend_Matrix%20Profile_I.pdf>`__

    See Table II

    Note that Q, T are not directly required to calculate D

    Note: Unlike the Matrix Profile I paper, here, M_T, Σ_T can be calculated
    once for all subsequences of T and passed in so the redundancy is removed
    """

    Q = np.asarray(Q)
    check_dtype(Q)

    if Q.ndim != 1:  # pragma: no cover
        raise ValueError(f"Q is {Q.ndim}-dimensional and must be 1-dimensional. ")
    m = Q.shape[0]

    T = np.asarray(T)
    check_dtype(T)

    if T.ndim != 1:  # pragma: no cover
        raise ValueError(f"T is {T.ndim}-dimensional and must be 1-dimensional. ")
    n = T.shape[0]

    distance_profile = np.empty(n - m + 1)
    if np.any(np.isnan(Q)):
        distance_profile[:] = np.inf
    else:
        QT = sliding_dot_product(Q, T)
        μ_Q, σ_Q = compute_mean_std(Q, m)
        if M_T is None or Σ_T is None:
            M_T, Σ_T = compute_mean_std(T, m)
        distance_profile = calculate_distance_profile(
            m, QT, μ_Q.item(0), σ_Q.item(0), M_T, Σ_T
        )

    return distance_profile


def array_to_temp_file(a):
    """
    Write an ndarray to a file

    Parameters
    ----------
    a : ndarray
        An array to be written to a file

    Returns
    -------
    fname : str
        The output file name
    """

    fname = tempfile.NamedTemporaryFile(delete=False)
    fname = fname.name + ".npy"
    np.save(fname, a, allow_pickle=False)

    return fname


convolution = scipy.signal.fftconvolve  # Swap for other convolution function
