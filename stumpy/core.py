# STUMPY
# Copyright 2019 TD Ameritrade. Released under the terms of the 3-Clause BSD license.  # noqa: E501
# STUMPY is a trademark of TD Ameritrade IP Company, Inc. All rights reserved.

import numpy as np
from numba import njit, prange
import tempfile
import math

from . import config

try:
    from numba.cuda.cudadrv.driver import _raise_driver_not_found
except ImportError:
    pass


def driver_not_found(*args, **kwargs):  # pragma: no cover
    """
    Helper function to raise CudaSupportError driver not found error.
    """
    _raise_driver_not_found()


def _gpu_stump_driver_not_found(*args, **kwargs):  # pragma: no cover
    """
    Dummy function to raise CudaSupportError driver not found error.
    """
    driver_not_found()


def _gpu_aamp_driver_not_found(*args, **kwargs):  # pragma: no cover
    """
    Dummy function to raise CudaSupportError driver not found error.
    """
    driver_not_found()


def get_pkg_name():  # pragma: no cover
    """
    Return package name.
    """
    return __name__.split(".")[0]


def rolling_window(a, window):
    """
    Use strides to generate rolling/sliding windows for a numpy array.

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
    a = np.asarray(a)
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
    Check if the array contains NaNs.

    Raises
    ------
    ValueError
        If the array contains a NaN
    """
    if np.any(np.isnan(a)):
        msg = "Input array contains one or more NaNs"
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
        msg = f"{dtype} type expected but found {a.dtype}\n"
        msg += "Please change your input `dtype` with `.astype(float)`"
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
    Check the window size and ensure that it is greater than or equal to 3

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
    QT = np.convolve(Qr, T)

    return QT.real[m - 1 : n]


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
    num_chunks = config.STUMPY_MEAN_STD_NUM_CHUNKS
    max_iter = config.STUMPY_MEAN_STD_MAX_ITER

    if T.ndim > 2:  # pragma nocover
        raise ValueError("T has to be one or two dimensional!")

    for iteration in range(max_iter):
        try:
            chunk_size = math.ceil((T.shape[-1] + 1) / num_chunks)
            if chunk_size < m:
                chunk_size = m

            mean_chunks = []
            std_chunks = []
            for chunk in range(num_chunks):
                start = chunk * chunk_size
                stop = min(start + chunk_size + m - 1, T.shape[-1])
                if stop - start < m:
                    break

                tmp_mean = np.mean(rolling_window(T[..., start:stop], m), axis=T.ndim)
                mean_chunks.append(tmp_mean)
                tmp_std = np.nanstd(rolling_window(T[..., start:stop], m), axis=T.ndim)
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


@njit(fastmath=True)
def _calculate_squared_distance(m, QT, μ_Q, σ_Q, M_T, Σ_T):
    """
    Compute a single squared distance given all scalar inputs. This function serves as
    the single source of truth for how all distances should be calculated.

    Parameters
    ----------
    m : int
        Window size

    QT : float
        Dot product between `Q[i]` and `T[i]`

    μ_Q : float
        Mean of `Q[i]`

    σ_Q : float
        Standard deviation of `Q[i]`

    M_T : float
        Sliding mean of `T[i]`

    Σ_T : float
        Sliding standard deviation of `T[i]`

    Returns
    -------
    D_squared : float
        Squared distance

    Notes
    -----
    `DOI: 10.1109/ICDM.2016.0179 \
    <https://www.cs.ucr.edu/~eamonn/PID4481997_extend_Matrix%20Profile_I.pdf>`__

    See Equation on Page 4
    """
    if np.isinf(M_T) or np.isinf(μ_Q):
        D_squared = np.inf
    else:
        if σ_Q < config.STUMPY_STDDEV_THRESHOLD or Σ_T < config.STUMPY_STDDEV_THRESHOLD:
            D_squared = m
        else:
            denom = m * σ_Q * Σ_T
            if np.abs(denom) < config.STUMPY_DENOM_THRESHOLD:  # pragma nocover
                denom = config.STUMPY_DENOM_THRESHOLD
            D_squared = np.abs(2 * m * (1.0 - (QT - m * μ_Q * M_T) / denom))

        if (
            σ_Q < config.STUMPY_STDDEV_THRESHOLD
            and Σ_T < config.STUMPY_STDDEV_THRESHOLD
        ) or D_squared < config.STUMPY_D_SQUARED_THRESHOLD:
            D_squared = 0

    return D_squared


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
    D_squared : ndarray
        Squared distance profile

    Notes
    -----
    `DOI: 10.1109/ICDM.2016.0179 \
    <https://www.cs.ucr.edu/~eamonn/PID4481997_extend_Matrix%20Profile_I.pdf>`__

    See Equation on Page 4
    """
    k = M_T.shape[0]
    D_squared = np.empty(k)

    for i in prange(k):
        D_squared[i] = _calculate_squared_distance(m, QT[i], μ_Q, σ_Q, M_T[i], Σ_T[i])

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


@njit(fastmath=True)
def _mass(Q, T, QT, μ_Q, σ_Q, M_T, Σ_T):
    """
    A Numba JIT compiled algorithm for computing the distance profile using the MASS
    algorithm.

    Parameters
    ----------
    Q : ndarray
        Query array or subsequence

    T : ndarray
        Time series or sequence

    QT : ndarray
        The sliding dot product of Q and T

    μ_Q : float
        The scalar mean of Q

    σ_Q : float
        The scalar standard deviation of Q

    M_T : ndarray
        Sliding mean of `T`

    Σ_T : ndarray
        Sliding standard deviation of `T`

    Notes
    -----
    `DOI: 10.1109/ICDM.2016.0179 \
    <https://www.cs.ucr.edu/~eamonn/PID4481997_extend_Matrix%20Profile_I.pdf>`__

    See Table II

    Note that Q, T are not directly required to calculate D

    Note: Unlike the Matrix Profile I paper, here, M_T, Σ_T can be calculated
    once for all subsequences of T and passed in so the redundancy is removed
    """
    m = Q.shape[0]

    return calculate_distance_profile(m, QT, μ_Q, σ_Q, M_T, Σ_T)


def mass(Q, T, M_T=None, Σ_T=None):
    """
    Compute the distance profile using the MASS algorithm. This is a convenience
    wrapper around the Numba JIT compiled `_mass` function.

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
    Q = Q.copy()
    Q = np.asarray(Q)
    check_dtype(Q)
    m = Q.shape[0]

    if Q.ndim == 2 and Q.shape[1] == 1:  # pragma: no cover
        Q = Q.flatten()

    if Q.ndim != 1:  # pragma: no cover
        raise ValueError(f"Q is {Q.ndim}-dimensional and must be 1-dimensional. ")

    T = T.copy()
    T = np.asarray(T)
    check_dtype(T)
    n = T.shape[0]

    if T.ndim == 2 and T.shape[1] == 1:  # pragma: no cover
        T = T.flatten()

    if T.ndim != 1:  # pragma: no cover
        raise ValueError(f"T is {T.ndim}-dimensional and must be 1-dimensional. ")

    distance_profile = np.empty(n - m + 1)
    if np.any(~np.isfinite(Q)):
        distance_profile[:] = np.inf
    else:
        if M_T is None or Σ_T is None:
            T, M_T, Σ_T = preprocess(T, m)

        QT = sliding_dot_product(Q, T)
        μ_Q, σ_Q = compute_mean_std(Q, m)
        μ_Q = μ_Q[0]
        σ_Q = σ_Q[0]
        distance_profile[:] = _mass(Q, T, QT, μ_Q, σ_Q, M_T, Σ_T)

    return distance_profile


@njit(fastmath=True)
def _mass_absolute(Q_squared, T_squared, QT):
    """
    A Numba JIT compiled algorithm for computing the non-normalized distance profile
    using the MASS absolute algorithm.

    Parameters
    ----------
    Q_squared : ndarray
        Squared query array or subsequence

    T_squared : ndarray
        Squared time series or sequence

    QT : ndarray
        Sliding window dot product of `Q` and `T`

    Returns
    -------
    output : ndarray
        Unnormalized distance profile

    Notes
    -----
    `See Mueen's Absolute Algorithm for Similarity Search \
    <https://www.cs.unm.edu/~mueen/MASS_absolute.m>`__
    """
    return np.sqrt(Q_squared + T_squared - 2 * QT)


def mass_absolute(Q, T):
    """
    Compute the non-normalized distance profile (i.e., without z-normalization) using
    the "MASS absolute" algorithm. This is a convenience wrapper around the Numba JIT
    compiled `_mass_absolute` function.

    Parameters
    ----------
    Q : ndarray
        Query array or subsequence

    T : ndarray
        Time series or sequence

    Returns
    -------
    output : ndarray
        Unnormalized Distance profile

    Notes
    -----
    `See Mueen's Absolute Algorithm for Similarity Search \
    <https://www.cs.unm.edu/~mueen/MASS_absolute.m>`__
    """
    Q = Q.copy()
    Q = np.asarray(Q)
    check_dtype(Q)
    m = Q.shape[0]

    if Q.ndim == 2 and Q.shape[1] == 1:  # pragma: no cover
        Q = Q.flatten()

    if Q.ndim != 1:  # pragma: no cover
        raise ValueError(f"Q is {Q.ndim}-dimensional and must be 1-dimensional. ")

    T = T.copy()
    T = np.asarray(T)
    check_dtype(T)
    n = T.shape[0]

    if T.ndim == 2 and T.shape[1] == 1:  # pragma: no cover
        T = T.flatten()

    if T.ndim != 1:  # pragma: no cover
        raise ValueError(f"T is {T.ndim}-dimensional and must be 1-dimensional. ")

    distance_profile = np.empty(n - m + 1)
    if np.any(~np.isfinite(Q)):
        distance_profile[:] = np.inf
    else:
        T_subseq_isfinite = np.all(np.isfinite(rolling_window(T, m)), axis=1)
        T[~np.isfinite(T)] = 0.0
        QT = sliding_dot_product(Q, T)
        Q_squared = np.sum(Q * Q)
        T_squared = np.sum(rolling_window(T * T, m), axis=1)
        distance_profile[:] = _mass_absolute(Q_squared, T_squared, QT)
        distance_profile[~T_subseq_isfinite] = np.inf

    return distance_profile


def _get_QT(start, T_A, T_B, m):
    """
    Compute the sliding dot product between the query, `T_A`, (from
    [start:start+m]) and the time series, `T_B`. Additionally, compute
    QT for the first window.

    Parameters
    ----------
    start : int
        The window index for T_B from which to calculate the QT dot product

    T_A : ndarray
        The time series or sequence for which to compute the dot product

    T_B : ndarray
        The time series or sequence that will be used to annotate T_A. For every
        subsequence in T_A, its nearest neighbor in T_B will be recorded.

    m : int
        Window size

    Returns
    -------
    QT : ndarray
        Given `start`, return the corresponding QT

    QT_first : ndarray
         QT for the first window
    """
    QT = sliding_dot_product(T_B[start : start + m], T_A)
    QT_first = sliding_dot_product(T_A[:m], T_B)

    return QT, QT_first


@njit(fastmath=True)
def apply_exclusion_zone(D, idx, excl_zone):
    """
    Apply an exclusion zone to an array (inplace), i.e. set all values
    to np.inf in a window around a given index.

    All values in D in [idx - excl_zone, idx + excl_zone] (endpoints included)
    will be set to np.inf.

    Parameters
    ----------
    D : ndarray
        The array you want to apply the exclusion zone to

    idx : int
        The index around which the window should be centered

    excl_zone : int
        Size of the exclusion zone.
    """
    zone_start = max(0, idx - excl_zone)
    zone_stop = min(D.shape[-1], idx + excl_zone)
    D[..., zone_start : zone_stop + 1] = np.inf


def preprocess(T, m):
    """
    Creates a copy of the time series where all NaN and inf values
    are replaced with zero. Also computes mean and standard deviation
    for every subsequence. Every subsequence that contains at least
    one NaN or inf value, will have a mean of np.inf. For the standard
    deviation these values are ignored. If all values are illegal, the
    standard deviation will be 0 (see `core.compute_mean_std`)

    Parameters
    ----------
    T : ndarray
        Time series or sequence

    m : int
        Window size

    Returns
    -------
    T : ndarray
        Modified time series
    M_T : ndarray
        Rolling mean
    Σ_T : ndarray
        Rolling standard deviation
    """
    T = T.copy()
    T = np.asarray(T)
    check_dtype(T)

    T[np.isinf(T)] = np.nan
    M_T, Σ_T = compute_mean_std(T, m)
    T[np.isnan(T)] = 0

    return T, M_T, Σ_T


def preprocess_non_normalized(T, m):
    """
    Preprocess a time series that is to be used when computing a non-normalized (i.e.,
    without z-normalization) distance matrix.

    Creates a copy of the time series where all NaN and inf values
    are replaced with zero. Every subsequence that contains at least
    one NaN or inf value will have a `False` value in its `T_subseq_isfinite` `bool`
    array.

    Parameters
    ----------
    T : ndarray
        Time series or sequence

    m : int
        Window size

    Returns
    -------
    T : ndarray
        Modified time series

    T_subseq_isfinite : ndarray
        A boolean array that indicates whether a subsequence in `T` contains a
        `np.nan`/`np.inf` value (False)
    """
    T = T.copy()
    T = np.asarray(T)
    check_dtype(T)

    T[np.isinf(T)] = np.nan
    T_subseq_isfinite = np.all(np.isfinite(rolling_window(T, m)), axis=1)
    T[np.isnan(T)] = 0

    return T, T_subseq_isfinite


def preprocess_diagonal(T, m):
    """
    Preprocess a time series that is to be used when traversing the diagonals of a
    distance matrix.

    Creates a copy of the time series where all NaN and inf values
    are replaced with zero. Also computes means, `M_T` and `M_T_m_1`, for every
    subsequence using awindow size of `m` and `m-1`, respectively, and the inverse
    standard deviation, `Σ_T_inverse`. Every subsequence that contains at least
    one NaN or inf value will have a `False` value in its `T_subseq_isfinite` `bool`
    arra and it will also have a mean of np.inf. For the standard
    deviation these values are ignored. If all values are illegal, the
    standard deviation will be 0 (see `core.compute_mean_std`). Additionally,
    the inverse standard deviation, σ_inverse, will also be computed and returned.
    Finally, constant subsequences (i.e., subsequences with a standard deviation of
    zero), will have a corresponding `True` value in its `T_subseq_isconstant` array.

    Parameters
    ----------
    T : ndarray
        Time series or sequence

    m : int
        Window size

    Returns
    -------
    T : ndarray
        Modified time series

    M_T : ndarray
        Rolling mean with a subsequence length of `m`

    Σ_T_inverse : ndarray
        Inverted rolling standard deviation

    M_T_m_1 : ndarray
        Rolling mean with a subsequence length of `m-1`

    T_subseq_isfinite : ndarray
        A boolean array that indicates whether a subsequence in `T` contains a
        `np.nan`/`np.inf` value (False)

    T_subseq_isconstant : ndarray
        A boolean array that indicates whether a subsequence in `T` is constant (True)
    """
    T, T_subseq_isfinite = preprocess_non_normalized(T, m)
    M_T, Σ_T = compute_mean_std(T, m)
    T_subseq_isconstant = Σ_T < config.STUMPY_STDDEV_THRESHOLD
    Σ_T[T_subseq_isconstant] = 1.0  # Avoid divide by zero in next inversion step
    Σ_T_inverse = 1.0 / Σ_T
    M_T_m_1, _ = compute_mean_std(T, m - 1)

    return T, M_T, Σ_T_inverse, M_T_m_1, T_subseq_isfinite, T_subseq_isconstant


def replace_distance(D, search_val, replace_val, epsilon=0.0):
    """
    Replace values in distance array inplace

    Parameters
    ----------
    D : ndarray
        Distance array

    search_val : float
        Value to search for

    replace_val : float
        Value to replace with

    epsilon : float
        Threshold below `search_val` in which to still allow for a replacement

    Return
    ------
    None
    """
    D[D == search_val - epsilon] = replace_val


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


@njit(parallel=True, fastmath=True)
def _count_diagonal_ndist(diags, m, n_A, n_B):
    """
    Count the number of distances that would be computed for each diagonal index
    referenced in `diags`

    Parameters
    ----------
    m : int
        Window size

    n_A : ndarray
        The length of time series `T_A`

    n_B : ndarray
        The length of time series `T_B`

    diags : ndarray
        The diagonal indices of interest

    Returns
    -------
    diag_ndist_counts : ndarray
        Counts of distances computed along each diagonal of interest
    """
    diag_ndist_counts = np.zeros(diags.shape[0], dtype=np.int64)
    for diag_idx in prange(diags.shape[0]):
        k = diags[diag_idx]
        if k >= 0:
            diag_ndist_counts[diag_idx] = min(n_B - m + 1 - k, n_A - m + 1)
        else:
            diag_ndist_counts[diag_idx] = min(n_B - m + 1, n_A - m + 1 + k)

    return diag_ndist_counts


@njit()
def _get_array_ranges(a, n_chunks, truncate=False):
    """
    Given an input array, split it into `n_chunks`.

    Parameters
    ----------
    a : ndarray
        An array to be split

    n_chunks : int
        Number of chunks to split the array into

    truncate : bool
        If `truncate=True`, truncate the rows of `array_ranges` if there are not enough
        elements in `a` to be chunked up into `n_chunks`.  Otherwise, if
        `truncate=False`, all extra chunks will have their start and stop indices set
        to `a.shape[0]`.

    Returns
    -------
    array_ranges : ndarray
        A two column array where each row consists of a start and (exclusive) stop index
        pair. The first column contains the start indices and the second column
        contains the stop indices.
    """
    array_ranges = np.zeros((n_chunks, 2), np.int64)
    cumsum = a.cumsum() / a.sum()
    insert = np.linspace(0, 1, n_chunks + 1)[1:-1]
    idx = 1 + np.searchsorted(cumsum, insert)
    array_ranges[1:, 0] = idx  # Fill the first column with start indices
    array_ranges[:-1, 1] = idx  # Fill the second column with exclusive stop indices
    array_ranges[-1, 1] = a.shape[0]  # Handle the stop index for the final chunk

    diff_idx = np.diff(idx)
    if np.any(diff_idx == 0):
        row_truncation_idx = np.argmin(diff_idx) + 2
        array_ranges[row_truncation_idx:, 0] = a.shape[0]
        array_ranges[row_truncation_idx - 1 :, 1] = a.shape[0]
        if truncate:
            array_ranges = array_ranges[:row_truncation_idx]

    return array_ranges
