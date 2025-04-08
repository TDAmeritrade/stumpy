# STUMPY
# Copyright 2019 TD Ameritrade. Released under the terms of the 3-Clause BSD license.  # noqa: E501
# STUMPY is a trademark of TD Ameritrade IP Company, Inc. All rights reserved.

import functools
import inspect
import math
import tempfile
import warnings

import numpy as np
from numba import cuda, njit, prange
from scipy import linalg
from scipy.ndimage import maximum_filter1d, minimum_filter1d
from scipy.signal import convolve
from scipy.spatial.distance import cdist

from . import config

try:
    from numba.cuda.cudadrv.driver import _raise_driver_not_found
except ImportError:
    pass


def _compare_parameters(norm, non_norm, exclude=None):
    """
    Compare if the parameters in `norm` and `non_norm` are the same

    Parameters
    ----------
    norm : function
        The normalized function (or class) that is complementary to the
        non-normalized function (or class)

    non_norm : function
        The non-normalized function (or class) that is complementary to the
        z-normalized function (or class)

    exclude : list
        A list of parameters to exclude for the comparison

    Returns
    -------
    is_same_params : bool
        `True` if parameters from both `norm` and `non-norm` are the same. `False`
        otherwise.
    """
    norm_params = list(inspect.signature(norm).parameters.keys())
    non_norm_params = list(inspect.signature(non_norm).parameters.keys())

    if exclude is not None:
        for param in exclude:
            if param in norm_params:
                norm_params.remove(param)
            if param in non_norm_params:
                non_norm_params.remove(param)

    is_same_params = set(norm_params) == set(non_norm_params)
    if not is_same_params:
        msg = ""
        if exclude is not None or (isinstance(exclude, list) and len(exclude)):
            msg += f"Excluding `{exclude}` parameters, "
        msg += f"function `{norm.__name__}({norm_params}) and "
        msg += f"function `{non_norm.__name__}({non_norm_params}) "
        msg += "have different arguments/parameters."
        warnings.warn(msg)

    return is_same_params


def non_normalized(non_norm, exclude=None, replace=None):
    """
    Decorator for swapping a z-normalized function (or class) for its complementary
    non-normalized function (or class) as defined by `non_norm`. This requires that
    the z-normalized function (or class) has a `normalize` parameter.

    With the exception of `normalize` parameter, the `non_norm` function (or class)
    must have the same siganture as the `norm` function (or class) signature in order
    to be compatible. Please use a combination of the `exclude` and/or `replace`
    parameters when necessary.

    ```
    def non_norm_func(Q, T, A_non_norm):
        ...
        return


    @non_normalized(
        non_norm_func,
        exclude=["normalize", "p", "A_norm", "A_non_norm"],
        replace={"A_norm": "A_non_norm", "other_norm": None},
    )
    def norm_func(Q, T, A_norm=None, other_norm=None, normalize=True, p=2.0):
        ...
        return
    ```

    Parameters
    ----------
    non_norm : function
        The non-normalized function (or class) that is complementary to the
        z-normalized function (or class)

    exclude : list, default None
        A list of function (or class) parameter names to exclude when comparing the
        function (or class) signatures. When `exlcude is None`, this parameter is
        automatically set to `exclude = ["normalize", "p", "T_A_subseq_isconstant",
        T_B_subseq_isconstant]` by default.

    replace : dict, default None
        A dictionary of function (or class) parameter key-value pairs. Each key that
        is found as a parameter name in the `norm` function (or class) will be replaced
        by its corresponding or complementary parameter name in the `non_norm` function
        (or class) (e.g., {"norm_param": "non_norm_param"}). To remove any parameter in
        the `norm` function (or class) that does not exist in the `non_norm` function,
        simply set the value to `None` (i.e., {"norm_param": None}).

    Returns
    -------
    outer_wrapper : function
        The desired z-normalized/non-normalized function (or class)
    """
    if exclude is None:
        exclude = [
            "normalize",
            "p",
            "T_A_subseq_isconstant",
            "T_B_subseq_isconstant",
        ]

    @functools.wraps(non_norm)
    def outer_wrapper(norm):
        @functools.wraps(norm)
        def inner_wrapper(*args, **kwargs):
            is_same_params = _compare_parameters(norm, non_norm, exclude=exclude)
            if not is_same_params or kwargs.get("normalize", True):
                return norm(*args, **kwargs)
            else:
                kwargs = {k: v for k, v in kwargs.items() if k != "normalize"}
                if replace is not None:
                    for k, v in replace.items():
                        if k in kwargs.keys():
                            if v is None:  # pragma: no cover
                                _ = kwargs.pop(k)
                            else:
                                kwargs[v] = kwargs.pop(k)
                return non_norm(*args, **kwargs)

        return inner_wrapper

    return outer_wrapper


def driver_not_found(*args, **kwargs):  # pragma: no cover
    """
    Helper function to raise CudaSupportError driver not found error.

    Parameters
    ----------
    None

    Returns
    -------
    None
    """
    _raise_driver_not_found()


def _gpu_stump_driver_not_found(*args, **kwargs):  # pragma: no cover
    """
    Dummy function to raise CudaSupportError driver not found error.

    Parameters
    ----------
    None

    Returns
    -------
    None
    """
    driver_not_found()


def _gpu_aamp_driver_not_found(*args, **kwargs):  # pragma: no cover
    """
    Dummy function to raise CudaSupportError driver not found error.

    Parameters
    ----------
    None

    Returns
    -------
    None
    """
    driver_not_found()


def _gpu_ostinato_driver_not_found(*args, **kwargs):  # pragma: no cover
    """
    Dummy function to raise CudaSupportError driver not found error.

    Parameters
    ----------
    None

    Returns
    -------
    None
    """
    driver_not_found()


def _gpu_aamp_ostinato_driver_not_found(*args, **kwargs):  # pragma: no cover
    """
    Dummy function to raise CudaSupportError driver not found error.

    Parameters
    ----------
    None

    Returns
    -------
    None
    """
    driver_not_found()


def _gpu_mpdist_driver_not_found(*args, **kwargs):  # pragma: no cover
    """
    Dummy function to raise CudaSupportError driver not found error.

    Parameters
    ----------
    None

    Returns
    -------
    None
    """
    driver_not_found()


def _gpu_aampdist_driver_not_found(*args, **kwargs):  # pragma: no cover
    """
    Dummy function to raise CudaSupportError driver not found error.

    Parameters
    ----------
    None

    Returns
    -------
    None
    """
    driver_not_found()


def _gpu_stimp_driver_not_found(*args, **kwargs):  # pragma: no cover
    """
    Dummy function to raise CudaSupportError driver not found error.

    Parameters
    ----------
    None

    Returns
    -------
    None
    """
    driver_not_found()


def _gpu_aamp_stimp_driver_not_found(*args, **kwargs):  # pragma: no cover
    """
    Dummy function to raise CudaSupportError driver not found error.

    Parameters
    ----------
    None

    Returns
    -------
    None
    """
    driver_not_found()


def _gpu_searchsorted_left_driver_not_found(*args, **kwargs):  # pragma: no cover
    """
    Dummy function to raise CudaSupportError driver not found error.

    Parameters
    ----------
    None

    Returns
    -------
    None
    """
    driver_not_found()


def _gpu_searchsorted_right_driver_not_found(*args, **kwargs):  # pragma: no cover
    """
    Dummy function to raise CudaSupportError driver not found error.

    Parameters
    ----------
    None

    Returns
    -------
    None
    """
    driver_not_found()


def get_pkg_name():  # pragma: no cover
    """
    Return package name.

    Parameters
    ----------
    None

    Returns
    -------
    None
    """
    return __name__.split(".")[0]


def rolling_window(a, window):
    """
    Use strides to generate rolling/sliding windows for a numpy array.

    Parameters
    ----------
    a : numpy.ndarray
        numpy array

    window : int
        Size of the rolling window

    Returns
    -------
    output : numpy.ndarray
        This will be a new view of the original input array.
    """
    a = np.asarray(a)
    shape = a.shape[:-1] + (a.shape[-1] - window + 1, window)
    strides = a.strides + (a.strides[-1],)

    return np.lib.stride_tricks.as_strided(a, shape=shape, strides=strides)


def z_norm(a, axis=0, threshold=config.STUMPY_STDDEV_THRESHOLD):
    """
    Calculate the z-normalized input array `a` by subtracting the mean and
    dividing by the standard deviation along a given axis.

    Parameters
    ----------
    a : numpy.ndarray
        NumPy array

    axis : int, default 0
        NumPy array axis

    threshold : float, default to config.STUMPY_STDDEV_THRESHOLD
        A non-nan std value being less than `threshold` will be replaced with 1.0

    Returns
    -------
    output : numpy.ndarray
        An array with z-normalized values computed along a specified axis.
    """
    std = np.std(a, axis, keepdims=True)
    std[np.less(std, threshold, where=~np.isnan(std))] = 1.0

    return (a - np.mean(a, axis, keepdims=True)) / std


def check_nan(a):  # pragma: no cover
    """
    Check if the array contains NaNs.

    Parameters
    ----------
    a : numpy.ndarray
        NumPy array

    Returns
    -------
    None

    Raises
    ------
    ValueError
        If the array contains a NaN
    """
    if np.any(np.isnan(a)):
        msg = "Input array contains one or more NaNs"
        raise ValueError(msg)

    return


def check_dtype(a, dtype=np.float64):  # pragma: no cover
    """
    Check if the array type of `a` is of type specified by `dtype` parameter.

    Parameters
    ----------
    a : numpy.ndarray
        NumPy array

    dtype : dtype, default np.float64
        NumPy `dtype`

    Returns
    -------
    None

    Raises
    ------
    TypeError
        If the array type does not match `dtype`
    """
    if dtype is int:
        dtype = np.int64
    if dtype is float:
        dtype = np.float64
    if dtype is bool:
        dtype = np.bool_
    if not np.issubdtype(a.dtype, dtype):
        msg = f"{dtype} dtype expected but found {a.dtype} in input array\n"
        msg += "Please change your input `dtype` with `.astype(dtype)`"
        raise TypeError(msg)

    return True


def transpose_dataframe(df):  # pragma: no cover
    """
    Check if the input is a column-wise pandas/polars `DataFrame`. If `True`, return a
    transpose dataframe since stumpy assumes that each row represents data from a
    different dimension while each column represents data from the same dimension.
    If `False`, return `a` unchanged. Pandas/polars `Series` do not need to be
    transposed.

    Note that this function has zero dependency on Pandas (not even a soft dependency).

    Parameters
    ----------
    df : DataFrame
        pandas/polars dataframe

    Returns
    -------
    output : df
        If `df` is a Pandas `DataFrame` then return `df.T`. Otherwise, return `df`
    """
    if type(df).__name__ == "DataFrame":
        return df.transpose()

    return df


def are_arrays_equal(a, b):  # pragma: no cover
    """
    Check if two arrays are equal; first by comparing memory addresses,
    and secondly by their values.

    Parameters
    ----------
    a : numpy.ndarray
        NumPy array

    b : numpy.ndarray
        NumPy array

    Returns
    -------
    output : bool
        This is `True` if the arrays are equal and `False` otherwise.
    """
    if id(a) == id(b):
        return True

    # For numpy >= 1.19
    # return np.array_equal(a, b, equal_nan=True)

    if a.shape != b.shape:
        return False

    return bool(((a == b) | (np.isnan(a) & np.isnan(b))).all())


def are_distances_too_small(a, threshold=10e-6):  # pragma: no cover
    """
    Check the distance values from a matrix profile.

    If the values are smaller than the threshold (i.e., less than 10e-6) then
    it could suggest that this is a self-join.

    Parameters
    ----------
    a : numpy.ndarray
        NumPy array

    threshold : float, default 10e-6
        Minimum value in which to compare the matrix profile to

    Returns
    -------
    output : bool
        This is `True` if the matrix profile distances are all below the
        threshold and `False` if they are all above the threshold.
    """
    if a.mean() < threshold or np.all(a < threshold):
        return True

    return False


def get_max_window_size(n):
    """
    Get the maximum window size for a self-join

    Parameters
    ----------
    n : int
        The length of the time series

    Returns
    -------
    max_m : int
        The maximum window size allowed given `config.STUMPY_EXCL_ZONE_DENOM`
    """
    max_m = (
        int(
            n
            - np.floor(
                (n + (config.STUMPY_EXCL_ZONE_DENOM - 1))
                // (config.STUMPY_EXCL_ZONE_DENOM + 1)
            )
        )
        - 1
    )

    return max_m


def check_window_size(m, max_size=None, n=None):
    """
    Check the window size and ensure that it is greater than or equal to 3 and, if
    ``max_size`` is provided, ensure that the window size is less than or equal to
    the ``max_size``. Furthermore, if ``n`` is provided, then a self-join is assumed
    and it checks whether all subsequences have at least one non-trivial neighbor.

    Parameters
    ----------
    m : int
        Window size

    max_size : int, default None
        The maximum window size allowed

    n : int, default None
        The length of the time series in the case of a self-join.
        ``n`` should not be supplied (or set to ``None``) in the case of an AB-join.

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

    if max_size is not None and m > max_size:
        raise ValueError(f"The window size must be less than or equal to {max_size}")

    if n is not None:
        # Raise warning if there is at least one subsequence with no eligible
        # (non-trivial) neighbor in the case of a self-join.

        # For any time series `T`, an "eligible nearest neighbor" subsequence for
        # the central-most subsequence must be located outside the `excl_zone`,
        # and the central-most subsequence will ALWAYS have the smallest relative
        # (index-wise) distance to its farthest neighbor amongst all other subsequences.
        # Therefore, we only need to check whether the `excl_zone` eliminates all
        # "neighbors" for the central-most subsequence in `T`. In fact, we just need to
        # verify whether the `excl_zone` eliminates the "neighbor" that is farthest
        # away (index-wise) from the central-most subsequence. If it does not, this
        # implies that all subsequences in `T` will have at least one "eligible nearest
        # neighbor" that is located outside of their respective excl_zone.

        excl_zone = int(math.ceil(m / config.STUMPY_EXCL_ZONE_DENOM))

        l = n - m + 1
        # The start index of subsequences are: 0, 1, ..., l-1

        # If `l` is odd
        # Suppose `l == 5`. So, the start index of the subsequences
        # are: 0, 1, 2, 3, 4
        # The central subsequence is located at index position c=2, with two
        # farthest neighbors, one located at index 0, and the other is located
        # at index 4. In both cases, the relative (index-wise) distance is 2,
        # which is simply `5 // 2`. In general, it can be shown that the
        # (index-wise) distance from the central subsequence to its farthest
        # neighbor is `l // 2`.

        # If `l` is even
        # Suppose `l == 6`. So, the start index of the subsequences
        # are: 0, 1, 2, 3, 4, 5
        # There are two central-most subsequences, located at the index
        # positions c=2 and c=3. For the central-most subsequence at index
        # position c=2, its farthest neighbor will be located at index 5 (to the
        # right of c=2) and, for the central-most subsequence at index position
        # c=3, its farthest neighbor will be located at index 0 (to the left of
        # c=3). In both cases, the relative (index-wise) distance is 3,
        # which is simply `6 // 2`. In general, it can be shown that the
        # (index-wise) distance from the central-most subsequence to its
        # farthest neighbor is `l // 2`.

        # Therefore, regardless if `l` is even or odd, for the central
        # subsequence for any time series, the index location of its
        # farthest neighbor will always be `l // 2` index positions away.
        diff_to_farthest_idx = l // 2
        if diff_to_farthest_idx <= excl_zone:
            msg = (
                f"The window size, 'm = {m}', may be too large and could lead to "
                + "meaningless results. Consider reducing 'm' where necessary"
            )
            warnings.warn(msg)


@njit(fastmath=config.STUMPY_FASTMATH_TRUE)
def _sliding_dot_product(Q, T):
    """
    A Numba JIT-compiled implementation of the sliding window dot product.

    Parameters
    ----------
    Q : numpy.ndarray
        Query array or subsequence

    T : numpy.ndarray
        Time series or sequence

    Returns
    -------
    out : numpy.ndarray
        Sliding dot product between `Q` and `T`.
    """
    m = Q.shape[0]
    l = T.shape[0] - m + 1
    out = np.empty(l)
    for i in range(l):
        out[i] = np.dot(Q, T[i : i + m])

    return out


def sliding_dot_product(Q, T):
    """
    Use FFT convolution to calculate the sliding window dot product.

    Parameters
    ----------
    Q : numpy.ndarray
        Query array or subsequence

    T : numpy.ndarray
        Time series or sequence

    Returns
    -------
    output : numpy.ndarray
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
    QT = convolve(Qr, T)

    return QT.real[m - 1 : n]


@njit(
    # "f8[:](f8[:], i8, b1[:])",
    fastmath=config.STUMPY_FASTMATH_FLAGS
)
def _welford_nanvar(a, w, a_subseq_isfinite):
    """
    Compute the rolling variance for a 1-D array while ignoring NaNs using a modified
    version of Welford's algorithm but is much faster than using `np.nanstd` with stride
    tricks.

    Parameters
    ----------
    a : numpy.ndarray
        The input array

    w : int
        The rolling window size

    a_subseq_isfinite : numpy.ndarray
        A boolean array that describes whether each subequence of length `w` within `a`
        is finite.

    Returns
    -------
    all_variances : numpy.ndarray
        Rolling window nanvar
    """
    all_variances = np.empty(a.shape[0] - w + 1, dtype=np.float64)
    prev_mean = 0.0
    prev_var = 0.0

    for start_idx in range(a.shape[0] - w + 1):
        prev_start_idx = start_idx - 1
        stop_idx = start_idx + w  # Exclusive index value
        last_idx = start_idx + w - 1  # Last inclusive index value

        if (
            start_idx == 0
            or not a_subseq_isfinite[prev_start_idx]
            or not a_subseq_isfinite[start_idx]
        ):
            curr_mean = np.nanmean(a[start_idx:stop_idx])
            curr_var = np.nanvar(a[start_idx:stop_idx])
        else:
            curr_mean = prev_mean + (a[last_idx] - a[prev_start_idx]) / w
            curr_var = (
                prev_var
                + (a[last_idx] - a[prev_start_idx])
                * (a[last_idx] - curr_mean + a[prev_start_idx] - prev_mean)
                / w
            )

        all_variances[start_idx] = curr_var

        prev_mean = curr_mean
        prev_var = curr_var

    return all_variances


def welford_nanvar(a, w=None):
    """
    Compute the rolling variance for a 1-D array while ignoring NaNs using a modified
    version of Welford's algorithm but is much faster than using `np.nanstd` with stride
    tricks.

    This is a convenience wrapper around the `_welford_nanvar` function.

    Parameters
    ----------
    a : numpy.ndarray
        The input array

    w : numpy.ndarray, default None
        The rolling window size

    Returns
    -------
    output : numpy.ndarray
        Rolling window nanvar.
    """
    if w is None:
        w = a.shape[0]

    a_subseq_isfinite = rolling_isfinite(a, w)

    return _welford_nanvar(a, w, a_subseq_isfinite)


def welford_nanstd(a, w=None):
    """
    Compute the rolling standard deviation for a 1-D array while ignoring NaNs using
    a modified version of Welford's algorithm but is much faster than using `np.nanstd`
    with stride tricks.

    This a convenience wrapper around `welford_nanvar`.

    Parameters
    ----------
    a : numpy.ndarray
        The input array

    w : numpy.ndarray, default None
        The rolling window size

    Returns
    -------
    output : numpy.ndarray
        Rolling window nanstd.
    """
    if w is None:
        w = a.shape[0]

    return np.sqrt(np.clip(welford_nanvar(a, w), a_min=0, a_max=None))


@njit(parallel=True, fastmath=config.STUMPY_FASTMATH_FLAGS)
def _rolling_nanstd_1d(a, w):
    """
    A Numba JIT-compiled and parallelized function for computing the rolling standard
    deviation for 1-D array while ignoring NaN.

    Parameters
    ----------
    a : numpy.ndarray
        The input array

    w : int
        The rolling window size

    Returns
    -------
    out : numpy.ndarray
        This 1D array has the length of `a.shape[0]-w+1`. `out[i]`
        contains the stddev value of `a[i : i + w]`
    """
    n = a.shape[0] - w + 1
    out = np.empty(n, dtype=np.float64)
    for i in prange(n):
        out[i] = np.nanstd(a[i : i + w])

    return out


def rolling_nanstd(a, w, welford=False):
    """
    Compute the rolling standard deviation over the last axis of `a` while ignoring
    NaNs.

    This essentially replaces:
        `np.nanstd(rolling_window(a[..., start:stop], w), axis=a.ndim)`

    Parameters
    ----------
    a : numpy.ndarray
        The input array

    w : numpy.ndarray
        The rolling window size

    welford : bool, default False
        When False (default), the computation is parallelized and the stddev of
        each subsequence is calculated on its own. When `welford==True`, the
        welford method is used to reduce the computing time at the cost of slightly
        reduced precision.

    Returns
    -------
    out : numpy.ndarray
        Rolling window nanstd
    """
    axis = a.ndim - 1  # Account for rolling
    if welford:
        return np.apply_along_axis(
            lambda a_row, w: welford_nanstd(a_row, w), axis=axis, arr=a, w=w
        )
    else:
        return np.apply_along_axis(
            lambda a_row, w: _rolling_nanstd_1d(a_row, w), axis=axis, arr=a, w=w
        )


def _rolling_nanmin_1d(a, w=None):
    """
    Compute the rolling min for 1-D while ignoring NaNs.

    This essentially replaces:

        `np.nanmin(rolling_window(a[..., start:stop], w), axis=a.ndim)`

    Parameters
    ----------
    a : numpy.ndarray
        The input array

    w : numpy.ndarray, default None
        The rolling window size

    Returns
    -------
    output : numpy.ndarray
        Rolling window nanmin.
    """
    if w is None:
        w = a.shape[0]

    half_window_size = int(math.ceil((w - 1) / 2))
    return minimum_filter1d(a, size=w)[
        half_window_size : half_window_size + a.shape[0] - w + 1
    ]


def _rolling_nanmax_1d(a, w=None):
    """
    Compute the rolling max for 1-D while ignoring NaNs.

    This essentially replaces:

        `np.nanmax(rolling_window(a[..., start:stop], w), axis=a.ndim)`

    Parameters
    ----------
    a : numpy.ndarray
        The input array

    w : numpy.ndarray, default None
        The rolling window size

    Returns
    -------
    output : numpy.ndarray
        Rolling window nanmax.
    """
    if w is None:
        w = a.shape[0]

    half_window_size = int(math.ceil((w - 1) / 2))
    return maximum_filter1d(a, size=w)[
        half_window_size : half_window_size + a.shape[0] - w + 1
    ]


def rolling_nanmin(a, w):
    """
    Compute the rolling min for 1-D and 2-D arrays while ignoring NaNs.

    This a convenience wrapper around `_rolling_nanmin_1d`.

    This essentially replaces:

        `np.nanmin(rolling_window(a[..., start:stop], w), axis=a.ndim)`

    Parameters
    ----------
    a : numpy.ndarray
        The input array

    w : numpy.ndarray
        The rolling window size

    Returns
    -------
    output : numpy.ndarray
        Rolling window nanmin.
    """
    axis = a.ndim - 1  # Account for rolling
    return np.apply_along_axis(
        lambda a_row, w: _rolling_nanmin_1d(a_row, w), axis=axis, arr=a, w=w
    )


def rolling_nanmax(a, w):
    """
    Compute the rolling max for 1-D and 2-D arrays while ignoring NaNs.

    This a convenience wrapper around `_rolling_nanmax_1d`.

    This essentially replaces:

        `np.nanmax(rolling_window(a[..., start:stop], w), axis=a.ndim)`

    Parameters
    ----------
    a : numpy.ndarray
        The input array

    w : numpy.ndarray
        The rolling window size

    Returns
    -------
    output : numpy.ndarray
        Rolling window nanmax.
    """
    axis = a.ndim - 1  # Account for rolling
    return np.apply_along_axis(
        lambda a_row, w: _rolling_nanmax_1d(a_row, w), axis=axis, arr=a, w=w
    )


def compute_mean_std(T, m):
    """
    Compute the sliding mean and standard deviation for the array `T` with
    a window size of `m`

    Parameters
    ----------
    T : numpy.ndarray
        Time series or sequence

    m : int
        Window size

    Returns
    -------
    M_T : numpy.ndarray
        Sliding mean. All nan values are replaced with np.inf

    Σ_T : numpy.ndarray
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
                tmp_std = rolling_nanstd(T[..., start:stop], m)
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


@njit(
    # "f8(i8, f8, f8, f8, f8, f8)",
    fastmath=config.STUMPY_FASTMATH_FLAGS
)
def _calculate_squared_distance(
    m, QT, μ_Q, σ_Q, M_T, Σ_T, Q_subseq_isconstant, T_subseq_isconstant
):
    """
    Compute a single squared distance given all scalar inputs.

    Parameters
    ----------
    m : int
        Window size

    QT : float
        Pre-computed dot product between `Q` and the ith subsequence in `T`, each with
        length `m`

    μ_Q : float
        Mean of `Q`

    σ_Q : float
        Standard deviation of `Q`

    M_T : float
        Mean of the ith subsequence in `T`

    Σ_T : float
        Standard deviation of the ith subsequence in `T`

    Q_subseq_isconstant : bool
        A boolean value that indicates whether the subsequence `Q` is constant (True)

    T_subseq_isconstant : bool
        A boolean value that indicates whether the ith subsequence in `T` is
        constant (True)

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
    elif Q_subseq_isconstant and T_subseq_isconstant:
        D_squared = 0
    elif Q_subseq_isconstant or T_subseq_isconstant:
        D_squared = m
    else:
        denom = (σ_Q * Σ_T) * m
        denom = max(denom, config.STUMPY_DENOM_THRESHOLD)

        ρ = (QT - (μ_Q * M_T) * m) / denom
        ρ = min(ρ, 1.0)

        D_squared = np.abs(2 * m * (1.0 - ρ))

    return D_squared


@njit(
    # "f8[:](i8, f8[:], f8, f8, f8[:], f8[:])",
    fastmath=config.STUMPY_FASTMATH_FLAGS,
)
def _calculate_squared_distance_profile(
    m, QT, μ_Q, σ_Q, M_T, Σ_T, Q_subseq_isconstant, T_subseq_isconstant
):
    """
    Compute the squared distance profile

    Parameters
    ----------
    m : int
        Window size

    QT : numpy.ndarray
        Dot product between `Q` and `T`

    μ_Q : float
        Mean of `Q`

    σ_Q : float
        Standard deviation of `Q`

    M_T : numpy.ndarray
        Sliding mean of `T`

    Σ_T : numpy.ndarray
        Sliding standard deviation of `T`

    Q_subseq_isconstant : bool
        A boolean value that indicates whether the subsequence `Q` is constant (True)

    T_subseq_isconstant : numpy.ndarray
        A boolean array that indicates whether a subsequence in `T` is constant (True)

    Returns
    -------
    D_squared : numpy.ndarray
        Squared distance profile

    Notes
    -----
    `DOI: 10.1109/ICDM.2016.0179 \
    <https://www.cs.ucr.edu/~eamonn/PID4481997_extend_Matrix%20Profile_I.pdf>`__

    See Equation on Page 4
    """
    k = M_T.shape[0]
    D_squared = np.empty(k, dtype=np.float64)

    for i in range(k):
        D_squared[i] = _calculate_squared_distance(
            m,
            QT[i],
            μ_Q,
            σ_Q,
            M_T[i],
            Σ_T[i],
            Q_subseq_isconstant,
            T_subseq_isconstant[i],
        )

    return D_squared


@njit(
    # "f8[:](i8, f8[:], f8, f8, f8[:], f8[:])",
    fastmath=config.STUMPY_FASTMATH_FLAGS,
)
def calculate_distance_profile(
    m, QT, μ_Q, σ_Q, M_T, Σ_T, Q_subseq_isconstant, T_subseq_isconstant
):
    """
    Compute the distance profile

    Parameters
    ----------
    m : int
        Window size

    QT : numpy.ndarray
        Dot product between `Q` and `T`

    μ_Q : float
        Mean of `Q`

    σ_Q : float
        Standard deviation of `Q`

    M_T : numpy.ndarray
        Sliding mean of `T`

    Σ_T : numpy.ndarray
        Sliding standard deviation of `T`

    Q_subseq_isconstant : bool
        A boolean value that indicates whether the subsequence `Q` is constant (True)

    T_subseq_isconstant : numpy.ndarray
        A boolean array that indicates whether a subsequence in `T` is constant (True)

    Returns
    -------
    output : numpy.ndarray
        Distance profile

    Notes
    -----
    `DOI: 10.1109/ICDM.2016.0179 \
    <https://www.cs.ucr.edu/~eamonn/PID4481997_extend_Matrix%20Profile_I.pdf>`__

    See Equation on Page 4
    """
    D_squared = _calculate_squared_distance_profile(
        m, QT, μ_Q, σ_Q, M_T, Σ_T, Q_subseq_isconstant, T_subseq_isconstant
    )

    return np.sqrt(D_squared)


@njit(fastmath=config.STUMPY_FASTMATH_TRUE)
def _p_norm_distance_profile(Q, T, p=2.0):
    """
    A Numba JIT-compiled and parallelized function for computing the p-normalized
    distance profile

    Parameters
    ----------
    Q : numpy.ndarray
        Query array or subsequence

    T : numpy.ndarray
        Time series or sequence

    p : float, default 2.0
        The p-norm to apply for computing the Minkowski distance.

    Returns
    -------
    output : numpy.ndarray
        p-normalized distance profile between `Q` and `T`

    Notes
    -----
    The special case `p==inf` is not supported.
    """
    m = Q.shape[0]
    l = T.shape[0] - m + 1
    p_norm_profile = np.empty(l, dtype=np.float64)

    if p == 2.0:
        Q_squared = np.sum(Q * Q)
        T_squared = np.empty(l, dtype=np.float64)
        T_squared[0] = np.sum(T[:m] * T[:m])
        for i in range(1, l):
            T_squared[i] = (
                T_squared[i - 1] - T[i - 1] * T[i - 1] + T[i + m - 1] * T[i + m - 1]
            )
        QT = _sliding_dot_product(Q, T)
        for i in range(l):
            p_norm_profile[i] = Q_squared + T_squared[i] - 2.0 * QT[i]
    else:
        for i in range(l):
            p_norm_profile[i] = np.sum(np.power(np.abs(Q - T[i : i + m]), p))

    return p_norm_profile


def _mass_absolute(Q, T, p=2.0):
    """
    A wrapper around `cdist` for computing the non-normalized distance profile

    Parameters
    ----------
    Q : numpy.ndarray
        Query array or subsequence

    T : numpy.ndarray
        Time series or sequence

    p : float, default 2.0
        The p-norm to apply for computing the Minkowski distance. Minkowski distance is
        typically used with `p` being 1 or 2, which correspond to the Manhattan distance
        and the Euclidean distance, respectively.

    Returns
    -------
    output : numpy.ndarray
        Non-normalized distance profile
    """
    m = Q.shape[0]

    return cdist(
        rolling_window(Q, m), rolling_window(T, m), metric="minkowski", p=p
    ).flatten()


def mass_absolute(Q, T, T_subseq_isfinite=None, p=2.0, query_idx=None):
    """
    Compute the non-normalized distance profile (i.e., without z-normalization) using
    the "MASS absolute" algorithm. This is a convenience wrapper around the Numba JIT
    compiled `_mass_absolute` function.

    Parameters
    ----------
    Q : numpy.ndarray
        Query array or subsequence

    T : numpy.ndarray
        Time series or sequence

    T_subseq_isfinite : numpy.ndarray, default None
        A boolean array that indicates whether a subsequence in `T` contains a
        `np.nan`/`np.inf` value (False)

    p : float, default 2.0
        The p-norm to apply for computing the Minkowski distance. Minkowski distance is
        typically used with `p` being 1 or 2, which correspond to the Manhattan distance
        and the Euclidean distance, respectively.

    query_idx : int, default None
        This is the index position along the time series, `T`, where the query
        subsequence, `Q`, is located. `query_idx` should be set to None if `Q`
        is not a subsequence of `T`. If `Q` is a subsequence of `T`, provding
        this argument is optional. If query_idx is provided, the distance between
        Q and T[query_idx : query_idx + m] will automatically be set to zero.

    Returns
    -------
    output : numpy.ndarray
        Unnormalized Distance profile

    Notes
    -----
    `See Mueen's Absolute Algorithm for Similarity Search \
    <https://www.cs.unm.edu/~mueen/MASS_absolute.m>`__
    """
    Q = _preprocess(Q)
    m = Q.shape[0]

    if Q.ndim == 2 and Q.shape[1] == 1:  # pragma: no cover
        warnings.warn("`Q` must be 1-dimensional and was automatically flattened")
        Q = Q.flatten()

    if Q.ndim != 1:  # pragma: no cover
        raise ValueError(f"`Q` is {Q.ndim}-dimensional and must be 1-dimensional. ")
    Q_isfinite = np.isfinite(Q)

    check_window_size(m, max_size=Q.shape[0])

    if query_idx is not None:  # pragma: no cover
        query_idx = int(query_idx)
        T_isfinite_idx = np.isfinite(T[query_idx : query_idx + m])
        if not np.all(Q_isfinite == T_isfinite_idx) or not np.allclose(
            Q[Q_isfinite], T[query_idx : query_idx + m][T_isfinite_idx]
        ):
            msg = (
                "Subsequences `Q` and `T[query_idx:query_idx+m]` are "
                + "different but were expected to be identical. Please "
                + "verify that `query_idx` is correct."
            )
            warnings.warn(msg)

    T = _preprocess(T)
    n = T.shape[0]

    if T.ndim == 2 and T.shape[1] == 1:  # pragma: no cover
        T = T.flatten()

    if T.ndim != 1:  # pragma: no cover
        raise ValueError(f"T is {T.ndim}-dimensional and must be 1-dimensional. ")

    if m > n:  # pragma: no cover
        raise ValueError(
            f"The length of `Q` ({len(Q)}) must be less than or equal to "
            f"the length of `T` ({len(T)}). "
        )

    distance_profile = np.empty(n - m + 1, dtype=np.float64)
    if np.any(~Q_isfinite):
        distance_profile[:] = np.inf
    else:
        if T_subseq_isfinite is None:
            T, T_subseq_isfinite = preprocess_non_normalized(T, m)
        distance_profile[:] = _mass_absolute(Q, T, p)
        if query_idx is not None:  # pragma: no cover
            distance_profile[query_idx] = 0.0

        distance_profile[~T_subseq_isfinite] = np.inf

    return distance_profile


def _mass_absolute_distance_matrix(Q, T, m, distance_matrix, p=2.0):
    """
    Compute the full non-normalized (i.e., without z-normalization) distance matrix
    between all of the subsequences of `Q` and `T` using the MASS absolute algorithm

    Parameters
    ----------
    Q : numpy.ndarray
        Query array

    T : numpy.ndarray
        Time series or sequence

    m : int
        Window size

    distance_matrix : numpy.ndarray
        The full output distance matrix. This is mandatory since it may be reused.

    p : float, default 2.0
        The p-norm to apply for computing the Minkowski distance. Minkowski distance is
        typically used with `p` being 1 or 2, which correspond to the Manhattan distance
        and the Euclidean distance, respectively.

    Returns
    -------
    None
    """
    cdist(
        rolling_window(Q, m),
        rolling_window(T, m),
        out=distance_matrix,
        metric="minkowski",
        p=p,
    )


def mueen_calculate_distance_profile(Q, T):
    """
    Compute the mueen distance profile

    Parameters
    ----------
    Q : numpy.ndarray
        Query array or subsequence

    T : numpy.ndarray
        Time series or sequence

    Returns
    -------
    output : numpy.ndarray
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

    cumsum_T = np.empty(len(T) + 1, dtype=np.float64)  # Add one element, fix off-by-one
    np.cumsum(T, out=cumsum_T[1:])  # store output in cumsum_T[1:]
    cumsum_T[0] = 0

    cumsum_T_squared = np.empty(len(T) + 1, dtype=np.float64)
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


@njit(
    # "f8[:](f8[:], f8[:], f8[:], f8, f8, f8[:], f8[:])",
    fastmath=config.STUMPY_FASTMATH_TRUE
)
def _mass(Q, T, QT, μ_Q, σ_Q, M_T, Σ_T, Q_subseq_isconstant, T_subseq_isconstant):
    """
    A Numba JIT compiled algorithm for computing the distance profile using the MASS
    algorithm.

    This private function assumes only finite numbers in your inputs and it is the
    responsibility of the user to pre-process and post-process their results if the
    original time series contains `np.nan`/`np.inf` values. Failure to do so will
    result in incorrect outputs. See `core.mass` for common pre-processing and
    post-processing procedures.

    Parameters
    ----------
    Q : numpy.ndarray
        Query array or subsequence

    T : numpy.ndarray
        Time series or sequence

    QT : numpy.ndarray
        The sliding dot product of Q and T

    μ_Q : float
        The scalar mean of Q

    σ_Q : float
        The scalar standard deviation of Q

    M_T : numpy.ndarray
        Sliding mean of `T`

    Σ_T : numpy.ndarray
        Sliding standard deviation of `T`

    Q_subseq_isconstant : bool
        A boolean value that indicates whether the subsequence `Q` is constant (True)

    T_subseq_isconstant : numpy.ndarray
        A boolean array that indicates whether a subsequence in `T` is constant (True)

    Returns
    -------
    output : numpy.ndarray
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
    m = Q.shape[0]

    return calculate_distance_profile(
        m, QT, μ_Q, σ_Q, M_T, Σ_T, Q_subseq_isconstant, T_subseq_isconstant
    )


@non_normalized(
    mass_absolute,
    exclude=[
        "normalize",
        "M_T",
        "Σ_T",
        "T_subseq_isfinite",
        "p",
        "T_subseq_isconstant",
        "Q_subseq_isconstant",
    ],
    replace={"M_T": "T_subseq_isfinite", "Σ_T": None},
)
def mass(
    Q,
    T,
    M_T=None,
    Σ_T=None,
    normalize=True,
    p=2.0,
    T_subseq_isfinite=None,
    T_subseq_isconstant=None,
    Q_subseq_isconstant=None,
    query_idx=None,
):
    """
    Compute the distance profile using the MASS algorithm

    This is a convenience wrapper around the Numba JIT compiled `_mass` function.

    Parameters
    ----------
    Q : numpy.ndarray
        Query array or subsequence.

    T : numpy.ndarray
        Time series or sequence.

    M_T : numpy.ndarray, default None
        Sliding mean of ``T``.

    Σ_T : numpy.ndarray, default None
        Sliding standard deviation of ``T``.

    normalize : bool, default True
        When set to ``True``, this z-normalizes subsequences prior to computing
        distances. Otherwise, this function gets re-routed to its complementary
        non-normalized equivalent set in the ``@core.non_normalized`` function
        decorator.

    p : float, default 2.0
        The p-norm to apply for computing the Minkowski distance. This parameter is
        ignored when ``normalize == True``.

    T_subseq_isfinite : numpy.ndarray, default None
        A boolean array that indicates whether a subsequence in ``T`` contains a
        ``np.nan``/``np.inf`` value (``False``). This parameter is ignored when
        ``normalize == True``.

    T_subseq_isconstant : numpy.ndarray or function, default None
        A boolean array that indicates whether a subsequence in ``T`` is constant
        (``True``). Alternatively, a custom, user-defined function that returns a
        boolean array that indicates whether a subsequence in ``T`` is constant
        (``True``). The function must only take two arguments, ``a``, a 1-D array,
        and ``w``, the window size, while additional arguments may be specified
        by currying the user-defined function using ``functools.partial``. Any
        subsequence with at least one ``np.nan``/``np.inf`` will automatically have
        its corresponding value set to ``False`` in this boolean array.

    Q_subseq_isconstant : numpy.ndarray or function, default None
        A boolean array that indicates whether the subsequence in ``Q`` is constant
        (``True``). Alternatively, a custom, user-defined function that returns a
        boolean array that indicates whether the subsequence in ``Q`` is constant
        (``True``). The function must only take two arguments, ``a``, a 1-D array,
        and ``w``, the window size, while additional arguments may be specified
        by currying the user-defined function using ``functools.partial``. Any
        subsequence with at least one ``np.nan``/``np.inf`` will automatically have
        its corresponding value set to ``False`` in this boolean array.

    query_idx : int, default None
        This is the index position along the time series, ``T``, where the query
        subsequence, ``Q``, is located. ``query_idx`` should be set to ``None`` if
        ``Q`` is not a subsequence of ``T``. If ``Q`` is a subsequence of ``T``,
        provding this argument is optional. If ``query_idx`` is provided, the distance
        between ``Q`` and ``T[query_idx : query_idx + m]`` will automatically be set to
        zero.

    Returns
    -------
    distance_profile : numpy.ndarray
        Distance profile.

    See Also
    --------
    stumpy.motifs : Discover the top motifs for time series ``T``
    stumpy.match : Find all matches of a query ``Q`` in a time series ``T``

    Notes
    -----
    `DOI: 10.1109/ICDM.2016.0179 \
    <https://www.cs.ucr.edu/~eamonn/PID4481997_extend_Matrix%20Profile_I.pdf>`__

    See Table II

    Note that ``Q``, ``T`` are not directly required to calculate ``D``

    Note: Unlike the Matrix Profile I paper, here, ``M_T``, ``Σ_T`` can be calculated
    once for all subsequences of ``T`` and passed in so the redundancy is removed

    Examples
    --------
    >>> import stumpy
    >>> import numpy as np
    >>> stumpy.mass(
    ...     np.array([-11.1, 23.4, 79.5, 1001.0]),
    ...     np.array([584., -11., 23., 79., 1001., 0., -19.]))
    array([3.18792463e+00, 1.11297393e-03, 3.23874018e+00, 3.34470195e+00])
    """
    Q = _preprocess(Q)
    m = Q.shape[0]

    if Q.ndim == 2 and Q.shape[1] == 1:  # pragma: no cover
        warnings.warn("`Q` must be 1-dimensional and was automatically flattened")
        Q = Q.flatten()

    if Q.ndim != 1:  # pragma: no cover
        raise ValueError(f"Q is {Q.ndim}-dimensional and must be 1-dimensional. ")
    Q_isfinite = np.isfinite(Q)

    check_window_size(m, max_size=Q.shape[0])

    if query_idx is not None:
        query_idx = int(query_idx)
        T_isfinite_idx = np.isfinite(T[query_idx : query_idx + m])
        if not np.all(Q_isfinite == T_isfinite_idx) or not np.allclose(
            Q[Q_isfinite], T[query_idx : query_idx + m][T_isfinite_idx]
        ):  # pragma: no cover
            msg = (
                "Subsequences `Q` and `T[query_idx:query_idx+m]` are "
                + "different but were expected to be identical. Please "
                + "verify that `query_idx` is correct."
            )
            warnings.warn(msg)

    T = _preprocess(T)
    n = T.shape[0]

    if T.ndim == 2 and T.shape[1] == 1:  # pragma: no cover
        T = T.flatten()

    if T.ndim != 1:  # pragma: no cover
        raise ValueError(f"T is {T.ndim}-dimensional and must be 1-dimensional. ")

    if m > n:  # pragma: no cover
        raise ValueError(
            f"The length of `Q` ({len(Q)}) must be less than or equal to "
            f"the length of `T` ({len(T)}). "
        )

    distance_profile = np.empty(n - m + 1, dtype=np.float64)
    if np.any(~Q_isfinite):
        distance_profile[:] = np.inf
    else:
        T, M_T, Σ_T, T_subseq_isconstant = preprocess(
            T,
            m,
            copy=False,
            M_T=M_T,
            Σ_T=Σ_T,
            T_subseq_isconstant=T_subseq_isconstant,
        )

        QT = sliding_dot_product(Q, T)
        Q, μ_Q, σ_Q, Q_subseq_isconstant = preprocess(
            Q,
            m,
            copy=False,
            T_subseq_isconstant=Q_subseq_isconstant,
        )

        distance_profile[:] = _mass(
            Q,
            T,
            QT,
            μ_Q[0],
            σ_Q[0],
            M_T,
            Σ_T,
            Q_subseq_isconstant[0],
            T_subseq_isconstant,
        )

        if query_idx is not None:
            distance_profile[query_idx] = 0

    return distance_profile


def _mass_distance_matrix(
    Q,
    T,
    m,
    distance_matrix,
    μ_Q,
    σ_Q,
    M_T,
    Σ_T,
    Q_subseq_isconstant,
    T_subseq_isconstant,
    query_idx=None,
):
    """
    Compute the full distance matrix between all of the subsequences of `Q` and `T`
    using the MASS algorithm

    Parameters
    ----------
    Q : numpy.ndarray
        Query array

    T : numpy.ndarray
        Time series or sequence

    m : int
        Window size

    distance_matrix : numpy.ndarray
        The full output distance matrix. This is mandatory since it may be reused.

    μ_Q : numpy.ndarray
        Sliding mean of `Q`

    σ_Q : numpy.ndarray
        Sliding standard deviation of `Q`

    M_T : numpy.ndarray
        Sliding mean of `T`

    Σ_T : numpy.ndarray
        Sliding standard deviation of `T`

    Q_subseq_isconstant : numpy.ndarray
        A boolean array that indicates whether the subsequence in `Q` is constant (True)

    T_subseq_isconstant : numpy.ndarray
        A boolean array that indicates whether a subsequence in `T` is constant (True)

    query_idx : int, default None
        This is the index position along the time series, `T`, where the query
        subsequence, `Q`, is located. `query_idx` should be set to None if `Q`
        is not a subsequence of `T`. If `Q` is a subsequence of `T`, provding
        this argument is optional. If provided, the precision of computation
        can be slightly improved.

    Returns
    -------
        None
    """
    if query_idx is not None:
        query_idx = int(query_idx)

    for i in range(distance_matrix.shape[0]):
        if np.any(~np.isfinite(Q[i : i + m])):  # pragma: no cover
            distance_matrix[i, :] = np.inf
        else:
            QT = _sliding_dot_product(Q[i : i + m], T)
            distance_matrix[i, :] = _mass(
                Q[i : i + m],
                T,
                QT,
                μ_Q[i],
                σ_Q[i],
                M_T,
                Σ_T,
                Q_subseq_isconstant[i],
                T_subseq_isconstant,
            )

            # this is to fix slight loss-of-precision
            if query_idx is not None:
                distance_matrix[i, query_idx + i] = 0.0


def mass_distance_matrix(
    Q,
    T,
    m,
    distance_matrix,
    M_T=None,
    Σ_T=None,
    T_subseq_isconstant=None,
    Q_subseq_isconstant=None,
):
    """
    Compute the full distance matrix between all of the subsequences of `Q` and `T`
    using the MASS algorithm

    Parameters
    ----------
    Q : numpy.ndarray
        Query array

    T : numpy.ndarray
        Time series or sequence

    m : int
        Window size

    distance_matrix : numpy.ndarray
        The full output distance matrix. This is mandatory since it may be reused.

    M_T : numpy.ndarray, default None
        Sliding mean of `T`

    Σ_T : numpy.ndarray, default None
        Sliding standard deviation of `T`

    T_subseq_isconstant : numpy.ndarray, function, default None
        A boolean array that indicates whether a subsequence in `T` is constant
        (True). Alternatively, a custom, user-defined function that returns a
        boolean array that indicates whether a subsequence in `T` is constant
        (True). The function must only take two arguments, `a`, a 1-D array,
        and `w`, the window size, while additional arguments may be specified
        by currying the user-defined function using `functools.partial`. Any
        subsequence with at least one np.nan/np.inf will automatically have its
        corresponding value set to False in this boolean array.

    Q_subseq_isconstant : numpy.ndarray, function, default None
        A boolean array that indicates whether the subsequence in `Q` is constant
        (True). Alternatively, a custom, user-defined function that returns a
        boolean array that indicates whether the subsequence in `Q` is constant
        (True). The function must only take two arguments, `a`, a 1-D array,
        and `w`, the window size, while additional arguments may be specified
        by currying the user-defined function using `functools.partial`. Any
        subsequence with at least one np.nan/np.inf will automatically have its
        corresponding value set to False in this boolean array.

    Returns
    -------
        None
    """
    Q, μ_Q, σ_Q, Q_subseq_isconstant = preprocess(
        T=Q, m=m, copy=True, T_subseq_isconstant=Q_subseq_isconstant
    )

    T, M_T, Σ_T, T_subseq_isconstant = preprocess(
        T,
        m,
        copy=True,
        M_T=M_T,
        Σ_T=Σ_T,
        T_subseq_isconstant=T_subseq_isconstant,
    )

    check_window_size(m, max_size=min(Q.shape[0], T.shape[0]))

    return _mass_distance_matrix(
        Q,
        T,
        m,
        distance_matrix,
        μ_Q,
        σ_Q,
        M_T,
        Σ_T,
        Q_subseq_isconstant,
        T_subseq_isconstant,
    )


def _get_QT(start, T_A, T_B, m):
    """
    Compute the sliding dot product between the query, `T_B`, (from
    [start:start+m]) and the time series, `T_A`. Additionally, compute
    QT for the first window `T_A[:m]` and `T_B`.

    Parameters
    ----------
    start : int
        The window index for T_B from which to calculate the QT dot product

    T_A : numpy.ndarray
        The time series or sequence for which to compute the dot product

    T_B : numpy.ndarray
        The time series or sequence that will be used to annotate T_A. For every
        subsequence in T_A, its nearest neighbor in T_B will be recorded.

    m : int
        Window size

    Returns
    -------
    QT : numpy.ndarray
        Given `start`, return the corresponding QT

    QT_first : numpy.ndarray
         QT for the first window
    """
    QT = sliding_dot_product(T_B[start : start + m], T_A)
    QT_first = sliding_dot_product(T_A[:m], T_B)

    return QT, QT_first


@njit(
    # ["(f8[:], i8, i8)", "(f8[:, :], i8, i8)"],
    fastmath=config.STUMPY_FASTMATH_FLAGS
)
def _apply_exclusion_zone(a, idx, excl_zone, val):
    """
    Apply an exclusion zone to an array (inplace), i.e. set all values
    to `val` in a window around a given index.

    All values in a in [idx - excl_zone, idx + excl_zone] (endpoints included)
    will be set to `val`.

    Parameters
    ----------
    a : numpy.ndarray
        The array you want to apply the exclusion zone to

    idx : int
        The index around which the window should be centered

    excl_zone : int
        Size of the exclusion zone.

    val : float or bool
        The elements within the exclusion zone will be set to this value

    Returns
    -------
    None
    """
    zone_start = max(0, idx - excl_zone)
    zone_stop = min(a.shape[-1], idx + excl_zone)
    a[..., zone_start : zone_stop + 1] = val


def apply_exclusion_zone(a, idx, excl_zone, val):
    """
    Apply an exclusion zone to an array (inplace), i.e. set all values
    to `val` in a window around a given index.

    All values in a in [idx - excl_zone, idx + excl_zone] (endpoints included)
    will be set to `val`. This is a convenience wrapper around the Numba JIT-compiled
    `_apply_exclusion_zone` function.

    Parameters
    ----------
    a : numpy.ndarray
        The array you want to apply the exclusion zone to

    idx : int
        The index around which the window should be centered

    excl_zone : int
        Size of the exclusion zone.

    val : float or bool
        The elements within the exclusion zone will be set to this value

    Returns
    -------
    None
    """
    check_dtype(a, dtype=type(val))
    _apply_exclusion_zone(a, idx, excl_zone, val)


def _preprocess(T, copy=True):
    """
    Creates a copy of the time series when `copy` is True, transposes all dataframes,
    converts to `numpy.ndarray`, and checks the `dtype`

    Parameters
    ----------
    T : numpy.ndarray
        Time series or sequence

    copy : bool, default True
        A boolean value that indicates whether the process should be done on
        input `T` (False) or its copy (True).

    Returns
    -------
    T : numpy.ndarray
        Modified time series
    """
    if copy:
        try:
            T = T.copy()
        except AttributeError:  # Polars copy
            T = T.clone()

    T = transpose_dataframe(T)

    if "polars" in str(type(T)):
        T = T.to_numpy(writable=True)

    T = np.asarray(T)
    check_dtype(T)

    return T


def preprocess(
    T,
    m,
    copy=True,
    M_T=None,
    Σ_T=None,
    T_subseq_isconstant=None,
):
    """
    Creates a copy of the time series where all NaN and inf values
    are replaced with zero. Also computes mean and standard deviation
    for every subsequence. Every subsequence that contains at least
    one NaN or inf value, will have a mean of np.inf. For the standard
    deviation these values are ignored. If all values are illegal, the
    standard deviation will be 0 (see `core.compute_mean_std`). Also,
    compute the rolling isconstant, a boolean array that indicates if
    a subsequence is constant (True) or False. A subsequence is constant
    if it contains finite values that are identical.

    Parameters
    ----------
    T : numpy.ndarray
        Time series or sequence

    m : int
        Window size

    copy : bool, default True
        A boolean value that indicates whether the process should be done on
        input `T` (False) or its copy (True).

    M_T : numpy.ndarray, default None
        Rolling mean

    Σ_T : numpy.ndarray, default None
        Rolling standard deviation

    T_subseq_isconstant : numpy.ndarray or function, default None
        A boolean array that indicates whether a subsequence in `T` is constant
        (True). Alternatively, a custom, user-defined function that returns a
        boolean array that indicates whether a subsequence in `T` is constant
        (True). The function must only take two arguments, `a`, a 1-D array,
        and `w`, the window size, while additional arguments may be specified
        by currying the user-defined function using `functools.partial`. Any
        subsequence with at least one np.nan/np.inf will automatically have its
        corresponding value set to False in this boolean array.

    Returns
    -------
    T : numpy.ndarray
        Modified time series
    M_T : numpy.ndarray
        Rolling mean
    Σ_T : numpy.ndarray
        Rolling standard deviation
    T_subseq_isconstant : numpy.ndarray
        A boolean array that indicates whether a subsequence in `T`
        is constant (True)
    """
    T = _preprocess(T, copy)
    check_window_size(m, max_size=T.shape[-1])

    T[np.isinf(T)] = np.nan

    T_subseq_isconstant = process_isconstant(T, m, T_subseq_isconstant)
    if M_T is None or Σ_T is None:
        M_T, Σ_T = compute_mean_std(T, m)
    T[np.isnan(T)] = 0

    return T, M_T, Σ_T, T_subseq_isconstant


def preprocess_non_normalized(T, m, copy=True):
    """
    Preprocess a time series that is to be used when computing a non-normalized (i.e.,
    without z-normalization) distance matrix.

    Creates a copy of the time series where all NaN and inf values
    are replaced with zero. Every subsequence that contains at least
    one NaN or inf value will have a `False` value in its `T_subseq_isfinite` `bool`
    array.

    Parameters
    ----------
    T : numpy.ndarray
        Time series or sequence

    m : int
        Window size

    copy : bool, default True
        A boolean value that indicates whether the process should be done on
        input `T` (False) or its copy (True).

    Returns
    -------
    T : numpy.ndarray
        Modified time series

    T_subseq_isfinite : numpy.ndarray
        A boolean array that indicates whether a subsequence in `T` contains a
        `np.nan`/`np.inf` value (False)
    """
    T = _preprocess(T, copy)
    check_window_size(m, max_size=T.shape[-1])
    T_subseq_isfinite = rolling_isfinite(T, m)
    T[~np.isfinite(T)] = np.nan
    T[np.isnan(T)] = 0

    return T, T_subseq_isfinite


def preprocess_diagonal(
    T,
    m,
    T_subseq_isconstant=None,
    copy=True,
):
    """
    Preprocess a time series that is to be used when traversing the diagonals of a
    distance matrix.

    Creates a copy of the time series where all NaN and inf values are replaced
    with zero. Also computes means, `M_T` and `M_T_m_1`, for every subsequence
    using a window size of `m` and `m-1`, respectively, and the inverse standard
    deviation, `Σ_T_inverse`. Every subsequence that contains at least one NaN or
    inf value will have a `False` value in its `T_subseq_isfinite` `bool` array.
    Additionally, the inverse standard deviation, σ_inverse, will also be computed
    and returned. Finally, constant subsequences (i.e., subsequences with a standard
    deviation of zero), will have a corresponding `True` value in its
    `T_subseq_isconstant` array.

    Parameters
    ----------
    T : numpy.ndarray
        Time series or sequence

    m : int
        Window size

    T_subseq_isconstant : numpy.ndarray or function, default None
        A boolean array that indicates whether a subsequence in `T` is constant
        (True). Alternatively, a custom, user-defined function that returns a
        boolean array that indicates whether a subsequence in `T` is constant
        (True). The function must only take two arguments, `a`, a 1-D array,
        and `w`, the window size, while additional arguments may be specified
        by currying the user-defined function using `functools.partial`. Any
        subsequence with at least one np.nan/np.inf will automatically have its
        corresponding value set to False in this boolean array.

    copy : bool, default True
        A boolean value that indicates whether the process should be done on
        input `T` (False) or its copy (True).

    Returns
    -------
    T : numpy.ndarray
        Modified time series

    M_T : numpy.ndarray
        Rolling mean with a subsequence length of `m`

    Σ_T_inverse : numpy.ndarray
        Inverted rolling standard deviation

    M_T_m_1 : numpy.ndarray
        Rolling mean with a subsequence length of `m-1`

    T_subseq_isfinite : numpy.ndarray
        A boolean array that indicates whether a subsequence in `T` contains a
        `np.nan`/`np.inf` value (False)

    T_subseq_isconstant : numpy.ndarray
        A boolean array that indicates whether a subsequence in `T` is constant (True)
    """
    T = _preprocess(T, copy)
    check_window_size(m, max_size=T.shape[-1])
    T_subseq_isfinite = rolling_isfinite(T, m)
    T[~np.isfinite(T)] = np.nan
    T_subseq_isconstant = process_isconstant(T, m, T_subseq_isconstant)
    T[np.isnan(T)] = 0

    M_T, Σ_T = compute_mean_std(T, m)
    Σ_T[T_subseq_isconstant] = 1.0  # Avoid divide by zero in next inversion step
    Σ_T_inverse = 1.0 / Σ_T
    M_T_m_1, _ = compute_mean_std(T, m - 1)

    return T, M_T, Σ_T_inverse, M_T_m_1, T_subseq_isfinite, T_subseq_isconstant


def replace_distance(D, search_val, replace_val, epsilon=0.0):
    """
    Replace values in distance array inplace

    Parameters
    ----------
    D : numpy.ndarray
        Distance array

    search_val : float
        Value to search for

    replace_val : float
        Value to replace with

    epsilon : float, default 0.0
        Threshold below `search_val` in which to still allow for a replacement

    Returns
    -------
    None
    """
    D[D == search_val - epsilon] = replace_val


def array_to_temp_file(a):
    """
    Write an array to a file

    Parameters
    ----------
    a : numpy.ndarray
        A NumPy array to be written to a file

    Returns
    -------
    fname : str
        The output file name
    """
    fname = tempfile.NamedTemporaryFile(delete=False)
    fname = fname.name + ".npy"
    np.save(fname, a, allow_pickle=False)

    return fname


@njit(
    # "i8[:](i8[:], i8, i8, i8)",
    fastmath=config.STUMPY_FASTMATH_TRUE,
)
def _count_diagonal_ndist(diags, m, n_A, n_B):
    """
    Count the number of distances that would be computed for each diagonal index
    referenced in `diags`

    Parameters
    ----------
    diags : numpy.ndarray
        The diagonal indices of interest

    m : int
        Window size

    n_A : int
        The length of time series `T_A`

    n_B : int
        The length of time series `T_B`

    Returns
    -------
    diag_ndist_counts : numpy.ndarray
        Counts of distances computed along each diagonal of interest
    """
    diag_ndist_counts = np.zeros(diags.shape[0], dtype=np.int64)
    for diag_idx in range(diags.shape[0]):
        k = diags[diag_idx]
        if k >= 0:
            diag_ndist_counts[diag_idx] = min(n_B - m + 1 - k, n_A - m + 1)
        else:
            diag_ndist_counts[diag_idx] = min(n_B - m + 1, n_A - m + 1 + k)

    return diag_ndist_counts


@njit(
    # "i8[:, :](i8[:], i8, b1)"
    fastmath=config.STUMPY_FASTMATH_TRUE
)
def _get_array_ranges(a, n_chunks, truncate):
    """
    Given an input array, split it into `n_chunks`.

    Parameters
    ----------
    a : numpy.ndarray
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
    array_ranges : numpy.ndarray
        A two column array where each row consists of a start and (exclusive) stop index
        pair. The first column contains the start indices and the second column
        contains the stop indices.
    """
    array_ranges = np.zeros((n_chunks, 2), dtype=np.int64)
    if a.shape[0] > 0 and n_chunks > 0:
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


@njit(
    # "i8[:, :](i8, i8, b1)"
    fastmath=config.STUMPY_FASTMATH_TRUE
)
def _get_ranges(size, n_chunks, truncate):
    """
    Given a single input integer value, split an array of that length `size` evenly into
    `n_chunks`.

    This function is different from `_get_array_ranges` in that it does not take into
    account the contents of the array and, instead, assumes that we are chunking up
    `np.ones(size, dtype=np.int64)`. Additionally, the non-truncated sections may not
    all appear at the end of the returned array (i.e., they may be scattered throughout
    different rows of the array) but may be identified as having the same start and
    stop indices.

    Parameters
    ----------
    size : int
        The size or length of an array to chunk

    n_chunks : int
        Number of chunks to split the array into

    truncate : bool
        If `truncate=True`, truncate the rows of `array_ranges` if there are not enough
        elements in `a` to be chunked up into `n_chunks`.  Otherwise, if
        `truncate=False`, all extra chunks will have their start and stop indices set
        to `a.shape[0]`.

    Returns
    -------
    array_ranges : numpy.ndarray
        A two column array where each row consists of a start and (exclusive) stop index
        pair. The first column contains the start indices and the second column
        contains the stop indices.
    """
    a = np.ones(size, dtype=np.int64)
    array_ranges = np.zeros((n_chunks, 2), dtype=np.int64)
    if a.shape[0] > 0 and n_chunks > 0:
        cumsum = a.cumsum()
        insert = np.linspace(0, a.sum(), n_chunks + 1)[1:-1]
        idx = 1 + np.searchsorted(cumsum, insert)
        array_ranges[1:, 0] = idx  # Fill the first column with start indices
        array_ranges[:-1, 1] = idx  # Fill the second column with exclusive stop indices
        array_ranges[-1, 1] = a.shape[0]  # Handle the stop index for the final chunk

    if truncate:
        array_ranges = array_ranges[array_ranges[:, 0] != array_ranges[:, 1]]

    return array_ranges


def _rolling_isfinite_1d(a, w):
    """
    Determine if all elements in each rolling window `isfinite`

    Parameters
    ----------
    a : numpy.ndarray
        The input array

    w : int
        The length of the rolling window

    Returns
    -------
    output : numpy.ndarray
        A boolean array of length `a.shape[0] - w + 1` that records whether each
        rolling window subsequence contain all finite values
    """
    if a.dtype == np.dtype("bool"):
        a_isfinite = a.copy()
    else:
        a_isfinite = np.isfinite(a)
    a_subseq_isfinite = rolling_window(a_isfinite, w)

    # Process first subsequence
    a_first_subseq = ~a_isfinite[:w]
    if a_first_subseq.any():
        a_isfinite[: np.flatnonzero(a_first_subseq).max()] = False

    # Shift `a_isfinite` and fill forward by `w`
    a_subseq_isfinite[~a_isfinite[w - 1 :]] = False

    return a_isfinite[: a_isfinite.shape[0] - w + 1]


def rolling_isfinite(a, w):
    """
    Compute the rolling `isfinite` for 1-D and 2-D arrays.

    This a convenience wrapper around `_rolling_isfinite_1d`.

    Parameters
    ----------
    a : numpy.ndarray
        The input array

    w : numpy.ndarray
        The rolling window size

    Returns
    -------
    output : numpy.ndarray
        Rolling window nanmax.
    """
    axis = a.ndim - 1  # Account for rolling
    return np.apply_along_axis(
        lambda a_row, w: _rolling_isfinite_1d(a_row, w), axis=axis, arr=a, w=w
    )


@njit(parallel=True, fastmath=config.STUMPY_FASTMATH_FLAGS)
def _rolling_isconstant(a, w):
    """
    Compute the rolling isconstant for 1-D array.

    This is accomplished by comparing the min and max within each window and
    assigning `True` when the min and max are equal and `False` otherwise. If
    a subsequence contains at least one NaN, then the subsequence is not constant.

    Parameters
    ----------
    a : numpy.ndarray
        The input array

    w : numpy.ndarray
        The rolling window size

    Returns
    -------
    output : numpy.ndarray
        Rolling window isconstant.
    """
    l = a.shape[0] - w + 1
    out = np.empty(l)
    for i in prange(l):
        out[i] = np.ptp(a[i : i + w])

    return out == 0


def rolling_isconstant(a, w, a_subseq_isconstant=None):
    """
    Compute the rolling isconstant for 1-D and 2-D arrays.

    This is accomplished by comparing the min and max within each window and
    assigning `True` when the min and max are equal and `False` otherwise. If
    a subsequence contains at least one NaN, then the subsequence is not constant.

    Parameters
    ----------
    a : numpy.ndarray
        The input array

    w : numpy.ndarray
        The rolling window size

    a_subseq_isconstant : numpy.ndarray or function, default None
        A boolean array that indicates whether a subsequence in `a` is constant
        (True). Alternatively, a custom, user-defined function that returns a
        boolean array that indicates whether a subsequence in `a` is constant
        (True). The function must only take two arguments, `a`, a 1-D array,
        and `w`, the window size, while additional arguments may be specified
        by currying the user-defined function using `functools.partial`. When
        None, this defaults to `_rolling_isconstant`.

    Returns
    -------
    a_subseq_isconstant : numpy.ndarray
        Rolling window isconstant
    """
    if a_subseq_isconstant is None:
        a_subseq_isconstant = _rolling_isconstant

    isconstant_func = None
    if callable(a_subseq_isconstant):
        incomp_args = _find_incompatible_args(a_subseq_isconstant, ["a", "w"])
        if len(incomp_args) > 0:  # pragma: no cover
            msg = (
                f"Incompatible arguments {incomp_args} found in `T_subseq_isconstant`. "
                + "Please provide the custom function `T_subseq_isconstant` with "
                + "arguments `a`, a 1-D array, and `w`, the window size."
            )
            raise ValueError(msg)

        isconstant_func = a_subseq_isconstant

    elif isinstance(a_subseq_isconstant, np.ndarray):
        isconstant_func = None
        if a.ndim != a_subseq_isconstant.ndim:  # pragma: no cover
            msg = (
                "The arrays `a` and `a_subseq_isconstant` must have same "
                + "number of dimensions, {a.ndim} != {a_subseq_isconstant.ndim}"
            )
            raise ValueError(msg)

    else:  # pragma: no cover
        msg = (
            "`T_subseq_isconstant` must be of type `np.ndarray` or a callable "
            + f"function. Found {type(a_subseq_isconstant)} instead."
        )
        raise ValueError(msg)

    if isconstant_func is not None:
        axis = a.ndim - 1
        a_subseq_isconstant = np.apply_along_axis(
            lambda a_row, w: isconstant_func(a_row, w), axis=axis, arr=a, w=w
        )

    if not issubclass(a_subseq_isconstant.dtype.type, np.bool_):  # pragma: no cover
        msg = (
            f"The output dtype of `T_subseq_isconstant` is {a_subseq_isconstant.dtype} "
            + "but dtype `np.bool` was expected"
        )
        raise ValueError(msg)

    return a_subseq_isconstant


def fix_isconstant_isfinite_conflicts(
    T, m, T_subseq_isconstant, T_subseq_isfinite=None
):
    """
    Fix `T_subseq_isconstant` by setting its element to False if their
    corresponding value in `T_subseq_isfinite` is False.

    Parameters
    ----------
    T : numpy.ndarray
        Time series

    m : int
        Subsequence window size

    T_subseq_isconstant : numpy.ndarray
        A numpy array `dtype` of boolean that indicates whether a subsequence
        is constant (True)  or not (False).

    T_subseq_isfinite : numpy.ndarray, default None
        A boolean array that indicates whether a subsequence in `T` contains a
        `np.nan`/`np.inf` value (False)

    Returns
    -------
    fixed : numpy.ndarray
        The same as input `T_subseq_isconstant` but with indices set to False
        if their corresponding subsequence are not finite.
    """
    if T_subseq_isfinite is None:
        T_subseq_isfinite = rolling_isfinite(T, m)

    fixed = np.logical_and(T_subseq_isconstant, T_subseq_isfinite)

    conflicts = fixed != T_subseq_isconstant
    if np.any(conflicts):  # pragma: no cover
        msg = (
            f"Subsequences located at indices {np.nonzero(conflicts)} contain one "
            + "or more np.nan/np.inf and so their corresponding values in "
            + "`T_subseq_isconstant` have been automatically switched from True "
            + " to False."
        )
        warnings.warn(msg)

    return fixed


def _get_partial_mp_func(mp_func, client=None, device_id=None):
    """
    A convenience function for creating a `functools.partial` matrix profile function
    for single server (parallel CPU), multi-server with Dask distributed (parallel CPU),
    and multi-GPU implementations.

    Parameters
    ----------
    mp_func : function
        The matrix profile function to be used for computing a matrix profile

    client : client, default None
        A Dask or Ray Distributed client. Setting up a distributed cluster is beyond
        the scope of this library. Please refer to the Dask or Ray Distributed
        documentation.

    device_id : int or list, default None
        The (GPU) device number to use. The default value is `0`. A list of
        valid device ids (``int``) may also be provided for parallel GPU-STUMP
        computation. A list of all valid device ids can be obtained by
        executing `[device.id for device in numba.cuda.list_devices()]`.

    Returns
    -------
    partial_mp_func : functools.partial
        A generic matrix profile function that wraps the distributed `client` or GPU
        `device_id` into `functools.partial` function
    """
    if client is not None:
        partial_mp_func = functools.partial(mp_func, client)
    elif device_id is not None:
        partial_mp_func = functools.partial(mp_func, device_id=device_id)
    elif isinstance(mp_func, functools.partial):
        partial_mp_func = functools.partial(mp_func)
    else:
        partial_mp_func = mp_func

    return partial_mp_func


def _jagged_list_to_array(a, fill_value, dtype):
    """
    Fits a 2d jagged list into a 2d numpy array of the specified dtype.
    The resulting array will have a shape of (len(a), l), where l is the length
    of the longest list in a. All other lists will be padded with `fill_value`.

    Example:
    [[2, 1, 1], [0]] with a fill value of -1 will become
    np.array([[2, 1, 1], [0, -1, -1]])

    Parameters
    ----------
    a : list
        Jagged list (list-of-lists) to be converted into an array.

    fill_value : int or float
        Missing entries will be filled with this value.

    dtype : dtype
        The desired data-type for the array.

    Returns
    -------
    out : numpy.ndarray
        The resulting array of dtype `dtype`.
    """
    if not a:
        return np.array([[]])

    max_length = max([len(row) for row in a])

    out = np.full((len(a), max_length), fill_value, dtype=dtype)

    for i, row in enumerate(a):
        out[i, : row.size] = row

    return out


def _get_mask_slices(mask):
    """
    For a boolean vector mask, return the (inclusive) start and (exclusive) stop
    indices where the mask is `True`.

    Parameters
    ----------
    mask : numpy.ndarray
        A boolean 1D array

    Returns
    -------
    slices : numpy.ndarray
        slices of indices where the mask is True. Each slice has a size of two:
        The first number is the start index (inclusive)
        The second number is the end index (exclusive)

    """
    m1 = np.r_[0, mask]
    m2 = np.r_[mask, 0]

    (starts,) = np.where(~m1 & m2)
    (ends,) = np.where(m1 & ~m2)

    slices = np.c_[starts, ends]

    return slices


def _idx_to_mp(
    I, T, m, normalize=True, p=2.0, T_subseq_isconstant=None, check_neg=True
):
    """
    Convert a set of matrix profile indices (including left and right indices) to its
    corresponding matrix profile distances

    Parameters
    ----------
    I : numpy.ndarray
        A 1D array of matrix profile indices

    T : numpy.ndarray
        Time series

    m : int
        Subsequence window size

    normalize : bool, default True
        When set to `True`, this z-normalizes subsequences prior to computing distances

    p : float, default 2.0
        The p-norm to apply for computing the Minkowski distance. This parameter is
        ignored when `normalize == True`.

    T_subseq_isconstant : bool, default None
        A boolean value that indicates whether the ith subsequence in `T` is
        constant (True). When `None`, it is computed by `rolling_isconstant`

    check_neg : bool, default True
        Check for the existence of negative indices

    Returns
    -------
    P : numpy.ndarray
        Matrix profile distances
    """
    I = I.astype(np.int64)
    T = T.copy()

    if check_neg:
        neg_idx = np.where(I < 0)[0]
        if neg_idx.size > 0:  # pragma: no cover
            msg = f"A negative index value ({I[neg_idx[0]]}) was found "
            msg += f"at I[{neg_idx[0]}] where a positive index value was "
            msg += "expected (i.e., a negative index is considered null)."
            warnings.warn(msg)

    if normalize:
        T_subseq_isconstant = process_isconstant(T, m, T_subseq_isconstant)

    T_isfinite = np.isfinite(T)
    T_subseq_isfinite = np.all(rolling_window(T_isfinite, m), axis=1)

    T[~T_isfinite] = 0.0
    T_subseqs = rolling_window(T, m)
    nn_subseqs = T_subseqs[I]
    if normalize:
        P = linalg.norm(z_norm(T_subseqs, axis=1) - z_norm(nn_subseqs, axis=1), axis=1)
        nn_subseq_isconstant = T_subseq_isconstant[I]
        P[T_subseq_isconstant & nn_subseq_isconstant] = (
            0  # both subsequences are constant
        )
        P[np.logical_xor(T_subseq_isconstant, nn_subseq_isconstant)] = np.sqrt(
            m
        )  # only one subsequence is constant
    else:
        P = linalg.norm(T_subseqs - nn_subseqs, axis=1, ord=p)
    P[~T_subseq_isfinite] = np.inf
    P[I < 0] = np.inf

    return P


@njit(fastmath=config.STUMPY_FASTMATH_TRUE)
def _total_diagonal_ndists(tile_lower_diag, tile_upper_diag, tile_height, tile_width):
    """
    Count the total number of distances covered by a range of diagonals

    Parameters
    ----------
    tile_lower_diag : int
        The (inclusive) lower diagonal index

    tile_upper_diag : int
        The (exclusive) upper diagonal index

    tile_height : int
        The height of the tile

    tile_width : int
        The width of the tile

    Returns
    -------
    out : int
        The total number of distances

    Notes
    -----
    This function essentially uses the "shoelace formula" to determine the area
    of a simple polygon whose vertices are described by their Cartesian coordinates
    in a plane.
    """
    if tile_width < tile_height:
        # Transpose inputs, adjust for inclusive/exclusive diags
        tile_width, tile_height = tile_height, tile_width
        tile_lower_diag, tile_upper_diag = 1 - tile_upper_diag, 1 - tile_lower_diag

    if tile_lower_diag > tile_upper_diag:  # pragma: no cover
        # Swap diags
        tile_lower_diag, tile_upper_diag = tile_upper_diag, tile_lower_diag

    min_tile_diag = 1 - tile_height
    max_tile_diag = tile_width  # Exclusive

    if (
        tile_lower_diag < min_tile_diag
        or tile_upper_diag < min_tile_diag
        or tile_lower_diag > max_tile_diag
        or tile_upper_diag > max_tile_diag
    ):
        return 0

    if tile_lower_diag == min_tile_diag and tile_upper_diag == max_tile_diag:
        return tile_height * tile_width

    # Determine polygon shape and establish vertices
    if tile_lower_diag <= 0 and tile_upper_diag <= 0:
        # lower trapezoid/triangle
        lower_ndists = tile_height + tile_lower_diag
        upper_ndists = tile_height + tile_upper_diag
        vertices = np.array(
            [
                [tile_lower_diag, 0],
                [tile_upper_diag, 0],
                [-tile_height, upper_ndists],
                [-tile_height, lower_ndists],
            ]
        )
    elif tile_lower_diag <= 0 and 0 < tile_upper_diag <= tile_width - tile_height:
        # irregular hexagon/diamond
        lower_ndists = tile_height + tile_lower_diag
        upper_ndists = min(tile_height, tile_width - tile_upper_diag)
        vertices = np.array(
            [
                [tile_lower_diag, 0],
                [0, 0],
                [0, tile_upper_diag],
                [-upper_ndists, tile_upper_diag + upper_ndists],
                [-tile_height, lower_ndists],
            ]
        )
    elif tile_lower_diag <= 0 and tile_upper_diag >= tile_width - tile_height:
        # irregular hexagon/diamond
        lower_ndists = tile_height + tile_lower_diag
        upper_ndists = min(tile_height, tile_width - tile_upper_diag)
        vertices = np.array(
            [
                [tile_lower_diag, 0],
                [0, 0],
                [0, tile_upper_diag],
                [-upper_ndists, tile_upper_diag + upper_ndists],
                [-tile_height, tile_width],
                [-tile_height, lower_ndists],
            ]
        )
    elif (
        0 < tile_lower_diag <= tile_width - tile_height
        and 0 < tile_upper_diag <= tile_width - tile_height
    ):
        # parallelogram
        lower_ndists = min(tile_height, tile_width - tile_lower_diag)
        upper_ndists = min(tile_height, tile_width - tile_upper_diag)
        vertices = np.array(
            [
                [0, tile_lower_diag],
                [0, tile_upper_diag],
                [-upper_ndists, tile_upper_diag + upper_ndists],
                [-tile_height, tile_lower_diag + lower_ndists],
            ]
        )
    elif (
        0 < tile_lower_diag <= tile_width - tile_height
        and tile_upper_diag > tile_width - tile_height
    ):
        # upper diamond
        lower_ndists = min(tile_height, tile_width - tile_lower_diag)
        upper_ndists = tile_width - tile_upper_diag
        vertices = np.array(
            [
                [0, tile_lower_diag],
                [0, tile_upper_diag],
                [-upper_ndists, tile_upper_diag + upper_ndists],
                [-tile_height, tile_width],
                [-tile_height, tile_lower_diag + lower_ndists],
            ]
        )
    else:
        # tile_lower_diag > tile_width - tile_height and
        # tile_upper_diag > tile_width - tile_height
        # upper trapezoid
        lower_ndists = tile_width - tile_lower_diag
        upper_ndists = tile_width - tile_upper_diag
        vertices = np.array(
            [
                [0, tile_lower_diag],
                [0, tile_upper_diag],
                [-upper_ndists, tile_upper_diag + upper_ndists],
                [-lower_ndists, tile_width],
            ]
        )

    # Shoelace formula
    i = np.arange(vertices.shape[0])
    total_diagonal_ndists = np.abs(
        np.sum(
            vertices[i, 0] * vertices[i - 1, 1] - vertices[i, 1] * vertices[i - 1, 0]
        )
        / 2
    )
    # Account for over/under-counting due to jagged upper/lower diagonal shapes
    total_diagonal_ndists += lower_ndists / 2 - upper_ndists / 2

    return int(total_diagonal_ndists)


def _bfs_indices(n, fill_value=None):
    """
    Generate the level order indices from the implicit construction of a binary
    search tree followed by a breadth first (level order) search.

    Example:

    If `n = 10` then the corresponding (zero-based index) balanced binary tree is:

                5
               * *
              *   *
             *     *
            *       *
           *         *
          2           8
         * *         * *
        *   *       *   *
       *     *     *     *
      1       4   7       9
     *       *   *
    0       3   6

    And if we traverse the nodes at each level from left to right then the breadth
    first search indices would be `[5, 2, 8, 1, 4, 7, 9, 0, 3, 6]`. In this function,
    we avoid/skip the explicit construction of the binary tree and directly output
    the desired indices efficiently. Note that it is not always possible to reconstruct
    the position of the leaf nodes from the indices alone (i.e., a leaf node could be
    attached to any parent node from a previous layer). For example, if `n = 9` then
    the corresponding (zero-based index) balanced binary tree is:

                   4
                  * *
                 *   *
                *     *
               *       *
              *         *
             *           *
            *             *
           *               *
          2                 7
         * *               * *
        *   *             *   *
       *     *           *     *
      1       3         6       8
     *                 *
    0                 5

    And if we traverse the nodes at each level from left to right then the breadth
    first search indices would be `[4, 2, 7, 1, 3, 6, 8, 0, 5]`. Notice that the parent
    of index `5` is not index `1` or index `3`. Instead, it is index `6`! So, given only
    the indices, it is not possible to easily reconstruct the full tree (i.e., whether a
    given leaf index is a left or right child or which parent node that it should be
    attached to). Therefore, to reduce the abiguity, you can choose to fill in the
    missing leaf nodes by specifying a `fill_value = -1`, which will produce a modified
    set of indices, `[ 4,  2,  7,  1,  3,  6,  8,  0, -1, -1, -1,  5, -1, -1, -1]` and
    represents the following fully populated tree:

                   4
                  * *
                 *   *
                *     *
               *       *
              *         *
             *           *
            *             *
           *               *
          2                 7
         * *               * *
        *   *             *   *
       *     *           *     *
      1       3         6       8
     * *     * *       * *     * *
    0  -1  -1  -1     5  -1  -1  -1

    Parameters
    ----------
    n : int
        The number indices to generate the ordered indices for

    fill_value : int, default None
        The integer value to use to fill in any missing leaf nodes. A negative integer
        is recommended as the only allowable indices being returned will be positive
        whole numbers.

    Returns
    -------
    level_idx : numpy.ndarray
        The breadth first search (level order) indices
    """
    if n == 1:  # pragma: no cover
        return np.array([0], dtype=np.int64)

    nlevel = np.floor(np.log2(n) + 1).astype(np.int64)
    nindices = np.power(2, np.arange(nlevel))
    cumsum_nindices = np.cumsum(nindices)
    nindices[-1] = n - cumsum_nindices[np.searchsorted(cumsum_nindices, n) - 1]

    indices = np.empty((2, nindices.max()), dtype=np.int64)
    indices[0, 0] = 0
    indices[1, 0] = n
    tmp_indices = np.empty((2, 2 * nindices.max()), dtype=np.int64)

    out = np.empty(n, dtype=np.int64)
    out_idx = 0

    for nidx in nindices:
        level_indices = (indices[0, :nidx] + indices[1, :nidx]) // 2

        if out_idx + len(level_indices) < n:
            tmp_indices[0, 0 : 2 * nidx : 2] = indices[0, :nidx]
            tmp_indices[0, 1 : 2 * nidx : 2] = level_indices + 1
            tmp_indices[1, 0 : 2 * nidx : 2] = level_indices
            tmp_indices[1, 1 : 2 * nidx : 2] = indices[1, :nidx]

            mask = tmp_indices[0, : 2 * nidx] < tmp_indices[1, : 2 * nidx]
            mask_sum = np.count_nonzero(mask)
            indices[0, :mask_sum] = tmp_indices[0, : 2 * nidx][mask]
            indices[1, :mask_sum] = tmp_indices[1, : 2 * nidx][mask]

        # for level_idx in level_indices:
        #     yield level_idx

        out[out_idx : out_idx + len(level_indices)] = level_indices
        out_idx += len(level_indices)

    if fill_value is not None and nindices[-1] < np.power(2, nlevel):
        fill_value = int(fill_value)

        out = out[: -nindices[-1]]
        last_row = np.empty(np.power(2, nlevel - 1), dtype=np.int64)
        last_row[0::2] = out[-nindices[-2] :] - 1
        last_row[1::2] = out[-nindices[-2] :] + 1

        mask = np.isin(last_row, out[: nindices[-2]])
        last_row[mask] = fill_value
        last_row[last_row >= n] = fill_value
        out = np.concatenate([out, last_row])

    return out


def _contrast_pan(pan, threshold, bfs_indices, n_processed):
    """
    Center the pan matrix profile (inplace) around the desired distance threshold
    in order to increase the contrast

    Parameters
    ----------
    pan : numpy.ndarray
        The pan matrix profile

    threshold : float
        The distance threshold value in which to center the pan matrix profile around

    bfs_indices : numpy.ndarray
        The breadth-first-search indices

    n_processed : numpy.ndarray
        The number of breadth-first-search indices to apply contrast to

    Returns
    -------
    None
    """
    idx = bfs_indices[:n_processed]
    l = n_processed * pan.shape[1]
    tmp = pan[idx].argsort(kind="mergesort", axis=None)
    ranks = np.empty(l, dtype=np.int64)
    ranks[tmp] = np.arange(l).astype(np.int64)

    percentile = np.full(ranks.shape, np.nan)
    percentile[:l] = np.linspace(0, 1, l)
    percentile = percentile[ranks].reshape(pan[idx].shape)
    pan[idx] = 1.0 / (1.0 + np.exp(-10 * (percentile - threshold)))


def _binarize_pan(pan, threshold, bfs_indices, n_processed):
    """
    Binarize the pan matrix profile (inplace) such all values below the `threshold`
    are set to `0.0` and all values above the `threshold` are set to `1.0`.

    Parameters
    ----------
    pan : numpy.ndarray
        The pan matrix profile

    threshold : float
        The distance threshold value in which to center the pan matrix profile around

    bfs_indices : numpy.ndarray
        The breadth-first-search indices

    n_processed : numpy.ndarray
        The number of breadth-first-search indices to binarize

    Returns
    -------
    None
    """
    idx = bfs_indices[:n_processed]
    pan[idx] = np.where(pan[idx] <= threshold, 0.0, 1.0)


def _select_P_ABBA_value(P_ABBA, k, custom_func=None):
    """
    A convenience function for returning the `k`th smallest value from the `P_ABBA`
    array or use a custom function to specify what `P_ABBA` value to return.

    The MPdist distance measure considers two time series to be similar if they share
    many subsequences, regardless of the order of matching subsequences. MPdist
    concatenates the output of an AB-join and a BA-join and returns the `k`th smallest
    value as the reported distance. Note that MPdist is a measure and not a metric.
    Therefore, it does not obey the triangular inequality but the method is highly
    scalable.

    Parameters
    ----------
    P_ABBA : numpy.ndarray
        An unsorted array resulting from the concatenation of the outputs from an
        AB-joinand BA-join for two time series, `T_A` and `T_B`

    k : int
        Specify the `k`th value in the concatenated matrix profiles to return. This
        parameter is ignored when `custom_func` is not None.

    custom_func : function, default None
        A custom user defined function for selecting the desired value from the
        unsorted `P_ABBA` array. This function may need to leverage `functools.partial`
        and should take `P_ABBA` as its only input parameter and return a single
        `MPdist` value. The `percentage` and `k` parameters are ignored when
        `custom_func` is not None.

    Returns
    -------
    MPdist : float
        The matrix profile distance
    """
    k = min(int(k), P_ABBA.shape[0] - 1)
    if custom_func is not None:
        MPdist = custom_func(P_ABBA)
    else:
        partition = np.partition(P_ABBA, k)
        MPdist = partition[k]
        if ~np.isfinite(MPdist):
            partition[:k].sort()
            k = max(0, np.count_nonzero(np.isfinite(partition[:k])) - 1)
            MPdist = partition[k]

    return MPdist


@njit(fastmath=config.STUMPY_FASTMATH_FLAGS)
def _merge_topk_PI(PA, PB, IA, IB):
    """
    Merge two top-k matrix profiles `PA` and `PB`, and update `PA` (in place).
    When the inputs are 1D arrays, PA[i] is updated if it is greater than PB[i] and
    IA[i] != IB[i]. In such case, PA[i] and IA[i] are replaced with PB[i] and IB[i],
    respectively. (Note that it might happen that IA[i]==IB[i] but PA[i] != PB[i].
    This situation can occur if there is slight imprecision in numerical calculations.
    In that case, we do not update PA[i] and IA[i]. While updating PA[i] and IA[i]
    is harmless in this case, we avoid doing that so to be consistent with the merging
    process when the inputs are 2D arrays)
    When the inputs are 2D arrays, we always prioritize the values of `PA` over the
    values of `PB` in case of ties. (i.e., values from `PB` are always inserted to
    the right of values from `PA`). Also, update `IA` accordingly. In case of
    overlapping values between two arrays IA[i] and IB[i], the ones in IB[i] (and
    their corresponding values in PB[i]) are ignored throughout the updating process
    of IA[i] (and PA[i]).

    Unlike `_merge_topk_ρI`, where `top-k` largest values are kept, this function
    keeps `top-k` smallest values.

    Parameters
    ----------
    PA : numpy.ndarray
        A (top-k) matrix profile where values in each row are sorted in ascending
        order. `PA` must be 1- or 2-dimensional.

    PB : numpy.ndarray
        A (top-k) matrix profile where values in each row are sorted in ascending
        order. `PB` must have the same shape as `PA`.

    IA : numpy.ndarray
        A (top-k) matrix profile indices corresponding to `PA`

    IB : numpy.ndarray
        A (top-k) matrix profile indices corresponding to `PB`

    Returns
    -------
    None
    """
    if PA.ndim == 1:
        mask = (PB < PA) & (IB != IA)
        PA[mask] = PB[mask]
        IA[mask] = IB[mask]
    else:
        k = PA.shape[1]
        tmp_P = np.empty(k, dtype=np.float64)
        tmp_I = np.empty(k, dtype=np.int64)
        for i in range(PA.shape[0]):
            overlap = set(IB[i]).intersection(set(IA[i]))
            aj, bj = 0, 0
            idx = 0
            # 2 * k iterations are required to traverse both A and B if needed.
            for _ in range(2 * k):
                if idx >= k:
                    break
                if bj < k and PB[i, bj] < PA[i, aj]:
                    if IB[i, bj] not in overlap:
                        tmp_P[idx] = PB[i, bj]
                        tmp_I[idx] = IB[i, bj]
                        idx += 1
                    bj += 1
                else:
                    tmp_P[idx] = PA[i, aj]
                    tmp_I[idx] = IA[i, aj]
                    idx += 1
                    aj += 1

            PA[i] = tmp_P
            IA[i] = tmp_I


@njit(fastmath=config.STUMPY_FASTMATH_FLAGS)
def _merge_topk_ρI(ρA, ρB, IA, IB):
    """
    Merge two top-k pearson profiles `ρA` and `ρB`, and update `ρA` (in place).
    When the inputs are 1D arrays, ρA[i] is updated if it is less than ρB[i] and
    IA[i] != IB[i]. In such case, ρA[i] and IA[i] are replaced with ρB[i] and IB[i],
    respectively. (Note that it might happen that IA[i]==IB[i] but ρA[i] != ρB[i].
    This situation can occur if there is slight imprecision in numerical calculations.
    In that case, we do not update ρA[i] and IA[i]. While updating ρA[i] and IA[i]
    is harmless in this case, we avoid doing that so to be consistent with the merging
    process when the inputs are 2D arrays)
    When the inputs are 2D arrays, we always prioritize the values of `ρA` over
    the values of `ρB` in case of ties. (i.e., values from `ρB` are always inserted
    to the left of values from `ρA`). Also, update `IA` accordingly. In case of
    overlapping values between two arrays IA[i] and IB[i], the ones in IB[i] (and
    their corresponding values in ρB[i]) are ignored throughout the updating process
    of IA[i] (and ρA[i]).

    Unlike `_merge_topk_PI`, where `top-k` smallest values are kept, this function
    keeps `top-k` largest values.

    Parameters
    ----------
    ρA : numpy.ndarray
        A (top-k) pearson profile where values in each row are sorted in ascending
        order. `ρA` must be 1- or 2-dimensional.

    ρB : numpy.ndarray
        A (top-k) pearson profile, where values in each row are sorted in ascending
        order. `ρB` must have the same shape as `ρA`.

    IA : numpy.ndarray
        A (top-k) matrix profile indices corresponding to `ρA`

    IB : numpy.ndarray
        A (top-k) matrix profile indices corresponding to `ρB`

    Returns
    -------
    None
    """
    if ρA.ndim == 1:
        mask = (ρB > ρA) & (IB != IA)
        ρA[mask] = ρB[mask]
        IA[mask] = IB[mask]
    else:
        k = ρA.shape[1]
        tmp_ρ = np.empty(k, dtype=np.float64)
        tmp_I = np.empty(k, dtype=np.int64)
        last_idx = k - 1
        for i in range(len(ρA)):
            overlap = set(IB[i]).intersection(set(IA[i]))
            aj, bj = last_idx, last_idx
            idx = last_idx
            # 2 * k iterations are required to traverse both A and B if needed.
            for _ in range(2 * k):
                if idx < 0:
                    break
                if bj >= 0 and ρB[i, bj] > ρA[i, aj]:
                    if IB[i, bj] not in overlap:
                        tmp_ρ[idx] = ρB[i, bj]
                        tmp_I[idx] = IB[i, bj]
                        idx -= 1
                    bj -= 1
                else:
                    tmp_ρ[idx] = ρA[i, aj]
                    tmp_I[idx] = IA[i, aj]
                    idx -= 1
                    aj -= 1

            ρA[i] = tmp_ρ
            IA[i] = tmp_I


@njit(fastmath=config.STUMPY_FASTMATH_FLAGS)
def _shift_insert_at_index(a, idx, v, shift="right"):
    """
    If `shift=right` (default), all elements in `a[idx:]` are shifted to the right by
    one element, `v` in inserted at index `idx` and the last element is discarded.
    If `shift=left`, all elements in `a[:idx]` are shifted to the left by one element,
    `v` in inserted at index `idx-1`, and the first element is discarded. In both cases,
    `a` is updated in place and its length remains unchanged.

    Note that all unrecognized `shift` inputs will default to `shift=right`.


    Parameters
    ----------
    a : numpy.ndarray
        A 1d array

    idx : int
        The index at which the value `v` should be inserted. This can be any
        integer number from `0` to `len(a)`. When `idx=len(a)` and `shift="right"`,
        OR when `idx=0` and `shift="left"`, then no change will occur on
        the input array `a`.

    v : float
        The value that should be inserted into array `a` at index `idx`

    shift : str, default "right"
        The value that indicates whether the shifting of elements should be towards
        the right or left. If `shift="right"` (default), all elements in `a[idx:]`
        are shifted to the right by one element. If `shift="left"`, all elements
        in `a[:idx]` are shifted to the left by one element.

    Returns
    -------
    None
    """
    if shift == "left":
        if 0 < idx <= len(a):
            a[: idx - 1] = a[1:idx]
            # elements were shifted to the left, thus the insertion index becomes
            # `idx-1`
            a[idx - 1] = v
    else:
        if 0 <= idx < len(a):
            a[idx + 1 :] = a[idx:-1]
            a[idx] = v


def _check_P(P, threshold=1e-6):
    """
    Check if the 1-dimensional matrix profile values are too small and
    log a warning the if true.

    Parameters
    ----------
    P : numpy.ndarray
        A 1-dimensional matrix profile

    threshold : float, default 1e-6
        A distance threshold

    Returns
    -------
        None
    """
    if P.ndim != 1:
        raise ValueError("`P` was {P.ndim}-dimensional and must be 1-dimensional")
    if are_distances_too_small(P, threshold=threshold):  # pragma: no cover
        msg = f"A large number of values in `P` are smaller than {threshold}.\n"
        msg += "For a self-join, try setting `ignore_trivial=True`."
        warnings.warn(msg)


def _find_matches(
    D, excl_zone, max_distance=None, max_matches=None, query_idx=None, atol=1e-8
):
    """
    Find all matches of a query `Q` whose distance profile with `T` is `D`.

    Parameters
    ----------
    D : numpy.ndarray
        The distance profile of `Q` with `T`. It is a 1D numpy array of size
        `len(T)-len(Q)+1`, where `D[i]` is the distance between query `Q` and
        `T[i : i + len(Q)]`.

    excl_zone : int
        Size of the exclusion zone. That is, after finding the next-best-match
        located at index `idx`, we ignore subsequences with start index in range
        (idx -  excl_zone, idx + excl_zone + 1).

    max_distance : float or function, default None
        Maximum distance between `Q` and a subsequence `S` for `S` to be considered a
        match.
        If a function, then it has to be a function of one argument `D`, which will be
        the distance profile of `Q` with `T` (a 1D numpy array of size `n-m+1`).
        If None, this defaults to
        `np.nanmax([np.nanmean(D) - 2 * np.nanstd(D), np.nanmin(D)])` (i.e. at
        least the closest match will be returned).

    max_matches : int, default None
        The maximum amount of similar occurrences to be returned. The resulting
        occurrences are sorted by distance, so a value of `10` means that the
        indices of the most similar `10` subsequences is returned. If `None`, then all
        occurrences are returned.

    query_idx : int, default None
        This is the index position along the time series, `T`, where the query
        subsequence, `Q`, is located.
        `query_idx` should only be used when the matrix profile is a self-join and
        should be set to `None` for matrix profiles computed from AB-joins.
        If `query_idx` is set to a specific integer value, then this will help ensure
        that the self-match will be returned first.

    atol : float, default 1e-8
        The absolute tolerance parameter. This value will be added to `max_distance`
        when comparing distances between subsequences.

    Returns
    -------
    out : numpy.ndarray
        The first column consists of values selected from `D`. These are the distances
        of subsequences of `T` whose distances to `Q` are less than or equal to
        `max_distance`, sorted by distance (lowest to highest). The second column
        consists of the corresponding indices in `D`. These are in fact the start index
        of susequences in `T` selected as the match of `Q`.

    """
    D = D.copy()
    if max_distance is None:

        def max_distance(D):
            D_copy = D.copy().astype(np.float64)
            D_copy[np.isinf(D_copy)] = np.nan
            return np.nanmax(
                [np.nanmean(D_copy) - 2.0 * np.nanstd(D_copy), np.nanmin(D_copy)]
            )

    if not isinstance(max_distance, float):
        max_distance = max_distance(D)

    if max_matches is None:
        max_matches = np.inf

    if query_idx is not None:
        candidate_idx = query_idx
    else:
        candidate_idx = np.argmin(D)

    matches = []
    for _ in range(len(D)):
        if (
            D[candidate_idx] > atol + max_distance
            or ~np.isfinite(D[candidate_idx])
            or len(matches) >= max_matches
        ):
            break

        matches.append([D[candidate_idx], candidate_idx])
        apply_exclusion_zone(D, candidate_idx, excl_zone, np.inf)
        candidate_idx = np.argmin(D)

    return np.array(matches, dtype=object)


@cuda.jit(device=True)
def _gpu_searchsorted_left(a, v, bfs, nlevel):
    """
    A device function, equivalent to numpy.searchsorted(a, v, side='left')

    Parameters
    ----------
    a : numpy.ndarray
        1-dim array sorted in ascending order.

    v : float
        Value to insert into array `a`

    bfs : numpy.ndarray
        The breadth-first-search indices where the missing leaves of its corresponding
        binary search tree are filled with -1.

    nlevel : int
        The number of levels in the binary search tree from which the array
        `bfs` is obtained.

    Returns
    -------
    idx : int
        The index of the insertion point
    """
    n = a.shape[0]
    idx = 0
    for level in range(nlevel):
        if v <= a[bfs[idx]]:
            next_idx = 2 * idx + 1
        else:
            next_idx = 2 * idx + 2

        if level == nlevel - 1 or bfs[next_idx] < 0:
            if v <= a[bfs[idx]]:
                idx = max(bfs[idx], 0)
            else:
                idx = min(bfs[idx] + 1, n)
            break
        idx = next_idx

    return idx


@cuda.jit(device=True)
def _gpu_searchsorted_right(a, v, bfs, nlevel):
    """
    A device function, equivalent to numpy.searchsorted(a, v, side='right')

    Parameters
    ----------
    a : numpy.ndarray
        1-dim array sorted in ascending order.

    v : float
        Value to insert into array `a`

    bfs : numpy.ndarray
        The breadth-first-search indices where the missing leaves of its corresponding
        binary search tree are filled with -1.

    nlevel : int
        The number of levels in the binary search tree from which the array
        `bfs` is obtained.

    Returns
    -------
    idx : int
        The index of the insertion point
    """
    n = a.shape[0]
    idx = 0
    for level in range(nlevel):
        if v < a[bfs[idx]]:
            next_idx = 2 * idx + 1
        else:
            next_idx = 2 * idx + 2

        if level == nlevel - 1 or bfs[next_idx] < 0:
            if v < a[bfs[idx]]:
                idx = max(bfs[idx], 0)
            else:
                idx = min(bfs[idx] + 1, n)
            break
        idx = next_idx

    return idx


def check_ignore_trivial(T_A, T_B, ignore_trivial):
    """
    Check inputs and verify the appropriateness for self-joins vs AB-joins and
    provides relevant warnings.

    Note that the warnings will output the first occurrence of matching warnings
    for each location (module + line number) where the warning is issued

    Parameters
    ----------
    T_A : numpy.ndarray
        The time series or sequence for which to compute the matrix profile

    T_B : numpy.ndarray
        The time series or sequence that will be used to annotate T_A. For every
        subsequence in T_A, its nearest neighbor in T_B will be recorded. Default is
        `None` which corresponds to a self-join.

    ignore_trivial : bool
        Set to `True` if this is a self-join. Otherwise, for AB-join, set this
        to `False`.

    Returns
    -------
    ignore_trivial : bool
        The (corrected) ignore_trivial value

    Notes
    -----
    These warnings may be supressed by using a context manager
    ```
    import stumpy
    import numpy as np
    import warnings

    T = np.random.rand(10_000)
    m = 50
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="Arrays T_A, T_B are equal")
        for _ in range(5):
            stumpy.stump(T, m, T, ignore_trivial=False)
    ```
    """
    if ignore_trivial is False and are_arrays_equal(T_A, T_B):  # pragma: no cover
        msg = "Arrays T_A, T_B are equal, which implies a self-join. "
        msg += "Try setting `ignore_trivial = True`."
        warnings.warn(msg)

    if ignore_trivial and are_arrays_equal(T_A, T_B) is False:  # pragma: no cover
        msg = "Arrays T_A, T_B are not equal, which implies an AB-join. "
        msg += "`ignore_trivial` has been automatically set to `False`."
        warnings.warn(msg)
        ignore_trivial = False

    return ignore_trivial


def _client_to_func(client):
    """
    Based on the client information and the parent function calling this
    function, infer the name of the client function to return

    For example, if the parent function calling `_client_to_func` is called
    `stumped` and the `client` is a Dask client, then `_dask_` will be
    prepended to the string `calling_func` and the resulting function
    called `_dask_stumped` will be returned. For a Ray client, the function
    caled `_ray_stumped` will be returned. Note that it is the responsibility
    of the caller to ensure that the resulting derived function exists. Otherwise,
    this will likely result in a `ModuleNotFoundError`.

    Parameters
    ----------
    client : client
        A Dask or Ray Distributed client. Setting up a distributed cluster is beyond
        the scope of this library. Please refer to the Dask or Ray Distributed
        documentation.

    Returns
    -------
    func : function
        The correct function for a client
    """
    if client.__class__.__name__.startswith("Client"):
        prefix = "_dask_"
    elif inspect.ismodule(client) and str(client).startswith(
        "<module 'ray'"
    ):  # pragma: no cover
        prefix = "_ray_"
    else:
        msg = f"Distributed client `{client}` is unrecognized or "
        msg += "has yet to be implemented"
        raise NotImplementedError(msg)

    calling_func = inspect.stack()[1].function
    module = __import__(
        calling_func,
        globals(),
        locals(),
        level=1,
        fromlist=[prefix + calling_func],
    )
    func = getattr(module, prefix + calling_func)

    return func


def _find_incompatible_args(func, required_args):
    """
    For a given `func` and `requried_args`, return non-default
    arguments in `func` that are not in `required_args`

    Parameters
    ----------
    func : function
        A function

    required_args : list
        A list containing the name of required arguments.

    Returns
    -------
    out : set
        A set of non-default arguments in `func` which are not in
        required_args
    """
    if not isinstance(required_args, list):  # pragma: no cover
        required_args = list(required_args)

    non_default_args = []
    for arg_name, arg in inspect.signature(func).parameters.items():
        # inspect.signature(functools.partial(func)) returns all arguments
        # including the ones with default values. the following if block
        # is to find non-default arguments.
        if arg.default == inspect.Parameter.empty:
            non_default_args.append(arg_name)

    return set(non_default_args).difference(set(required_args))


def _preprocess_include(include):
    """
    A utility function for processing the `include` input

    Parameters
    ----------
    include : numpy.ndarray
        A list of (zero-based) indices corresponding to the dimensions in `T` that
        must be included in the constrained multidimensional motif search.
        For more information, see Section IV D in:

        `DOI: 10.1109/ICDM.2017.66 \
        <https://www.cs.ucr.edu/~eamonn/Motif_Discovery_ICDM.pdf>`__

    Returns
    -------
    include : numpy.ndarray
        Process `include` and remove any redundant index values
    """
    include = np.asarray(include)
    _, idx = np.unique(include, return_index=True)
    if include.shape[0] != idx.shape[0]:  # pragma: no cover
        warnings.warn("Removed repeating indices in `include`")
        include = include[np.sort(idx)]

    return include


def _apply_include(
    D,
    include,
    restricted_indices=None,
    unrestricted_indices=None,
    mask=None,
    tmp_swap=None,
):
    """
    Apply a transformation to the multi-dimensional distance profile so that specific
    dimensions are always included. Essentially, it is swapping rows within the distance
    profile.

    Parameters
    ----------
    D : numpy.ndarray
        The multi-dimensional distance profile

    include : numpy.ndarray
        A list of (zero-based) indices corresponding to the dimensions in `T` that
        must be included in the constrained multidimensional motif search.
        For more information, see Section IV D in:

        `DOI: 10.1109/ICDM.2017.66 \
        <https://www.cs.ucr.edu/~eamonn/Motif_Discovery_ICDM.pdf>`__

    restricted_indices : numpy.ndarray, default None
        A list of indices specified in `include` that reside in the first
        `include.shape[0]` rows

    unrestricted_indices : numpy.ndarray, default None
        A list of indices specified in `include` that do not reside in the first
        `include.shape[0]` rows

    mask : numpy.ndarray, default None
        A boolean mask to select for unrestricted indices

    tmp_swap : numpy.ndarray, default None
        A reusable array to aid in array element swapping

    Returns
    -------
    None
    """
    include = _preprocess_include(include)

    if restricted_indices is None:
        restricted_indices = include[include < include.shape[0]]

    if unrestricted_indices is None:
        unrestricted_indices = include[include >= include.shape[0]]

    if mask is None:
        mask = np.ones(include.shape[0], dtype=bool)
        mask[restricted_indices] = False

    if tmp_swap is None:
        tmp_swap = D[: include.shape[0]].copy()
    else:
        tmp_swap[:] = D[: include.shape[0]]

    D[: include.shape[0]] = D[include]
    D[unrestricted_indices] = tmp_swap[mask]


def _subspace(D, k, include=None, discords=False):
    """
    Compute the k-dimensional matrix profile subspace for a given subsequence index and
    its nearest neighbor index

    Parameters
    ----------
    D : numpy.ndarray
        The multi-dimensional distance profile

    k : int
        The subset number of dimensions out of `D = T.shape[0]`-dimensions to return
        the subspace for. Note that zero-based indexing is used.

    include : numpy.ndarray, default None
        A list of (zero-based) indices corresponding to the dimensions in `T` that
        must be included in the constrained multidimensional motif search.
        For more information, see Section IV D in:

        `DOI: 10.1109/ICDM.2017.66 \
        <https://www.cs.ucr.edu/~eamonn/Motif_Discovery_ICDM.pdf>`__

    discords : bool, default False
        When set to `True`, this reverses the distance profile to favor discords rather
        than motifs. Note that indices in `include` are still maintained and respected.

    Returns
    -------
        S : numpy.ndarray
        An array that contains the `k`th-dimensional subspace for the subsequence. Note
        that `k+1` rows will be returned.
    """
    if discords:
        sorted_idx = D[::-1].argsort(axis=0, kind="mergesort")
    else:
        sorted_idx = D.argsort(axis=0, kind="mergesort")

    # `include` processing occur here since we are dealing with indices, not distances
    if include is not None:
        include = _preprocess_include(include)
        mask = np.in1d(sorted_idx, include)
        include_idx = mask.nonzero()[0]
        exclude_idx = (~mask).nonzero()[0]
        sorted_idx[: include_idx.shape[0]], sorted_idx[include_idx.shape[0] :] = (
            sorted_idx[include_idx],
            sorted_idx[exclude_idx],
        )

    S = sorted_idx[: k + 1]

    return S


def _mdl(disc_subseqs, disc_neighbors, S, n_bit=8):
    """
    Compute the number of bits needed to compress one array with another
    using the minimum description length (MDL)

    Parameters
    ----------
    disc_subseqs : numpy.ndarray
        The discretized array to be compressed

    disc_neighbors : numpy.ndarray
        The discretized array that will be used as a hypothesis for compression

    S : numpy.ndarray
        An array that contains the `k`th-dimensional subspace to be used

    n_bit : int, default 8
        The number of bits to use for computing the bit size

    Returns
    -------
    bit_size : float
        The total number of bits computed from MDL for representing both input arrays
    """
    ndim = disc_subseqs.shape[0]
    sub_dims, m = disc_subseqs[S].shape

    n_val = len(np.unique(disc_subseqs[S] - disc_neighbors[S]))
    bit_size = n_bit * (2 * ndim * m - sub_dims * m)
    bit_size = bit_size + sub_dims * m * np.log2(n_val) + n_val * n_bit

    return bit_size


@njit(
    # "(i8, i8, f8[:, :], f8[:], i8, f8[:, :], i8[:, :], f8)",
    fastmath=config.STUMPY_FASTMATH_FLAGS,
)
def _compute_multi_PI(d, idx, D, D_prime, range_start, P, I, p=2.0):
    """
    A Numba JIT-compiled version of mSTOMP for updating the matrix profile and matrix
    profile indices

    Parameters
    ----------
    d : int
        The total number of dimensions in `T`

    idx : int
        The subsequence index for the i-th time series, `T[i]`

    D : numpy.ndarray
        The distance profile

    D_prime : numpy.ndarray
        A reusable array for storing the column-wise cumulative sum of `D`

    range_start : int
        The starting index value along `T` for which to start the matrix
        profile calculation

    P : numpy.ndarray
        The matrix profile

    I : numpy.ndarray
        The matrix profile indices

    p : float, default 2.0
        The p-norm to apply for computing the Minkowski distance. Minkowski distance is
        typically used with `p` being 1 or 2, which correspond to the Manhattan distance
        and the Euclidean distance, respectively.

    Returns
    -------
    None
    """
    D_prime[:] = 0.0
    for i in range(d):
        D_prime[:] = D_prime + np.power(D[i], 1.0 / p)

        min_index = np.argmin(D_prime)
        pos = idx - range_start
        I[i, pos] = min_index
        P[i, pos] = D_prime[min_index] / (i + 1)
        if np.isinf(P[i, pos]):  # pragma nocover
            I[i, pos] = -1


def _compute_P_ABBA(
    T_A,
    T_B,
    m,
    P_ABBA,
    partial_mp_func,
    client=None,
    device_id=None,
):
    """
    A convenience function for computing the (unsorted) concatenated matrix profiles
    from an AB-join and BA-join for the two time series, `T_A` and `T_B`. This result
    can then be used to compute the matrix profile distance (MPdist) measure.

    The MPdist distance measure considers two time series to be similar if they share
    many subsequences, regardless of the order of matching subsequences. MPdist
    concatenates the output of an AB-join and a BA-join and returns the `k`th smallest
    value as the reported distance. Note that MPdist is a measure and not a metric.
    Therefore, it does not obey the triangular inequality but the method is highly
    scalable.

    Parameters
    ----------
    T_A : numpy.ndarray
        The first time series or sequence for which to compute the matrix profile

    T_B : numpy.ndarray
        The second time series or sequence for which to compute the matrix profile

    m : int
        Window size

    P_ABBA : numpy.ndarray
        The output array to write the concatenated AB-join and BA-join results to

    partial_mp_func : functools.partial
        A generic matrix profile function that wraps extra parameters into
        `functools.partial` function

    client : client, default None
        A Dask or Ray Distributed client. Setting up a distributed cluster is beyond
        the scope of this library. Please refer to the Dask or Ray Distributed
        documentation.

    device_id : int or list, default None
        The (GPU) device number to use. The default value is `0`. A list of
        valid device ids (``int``) may also be provided for parallel GPU-STUMP
        computation. A list of all valid device ids can be obtained by
        executing `[device.id for device in numba.cuda.list_devices()]`.

    Returns
    -------
    None

    Notes
    -----
    `DOI: 10.1109/ICDM.2018.00119 \
    <https://www.cs.ucr.edu/~eamonn/MPdist_Expanded.pdf>`__

    See Section III
    """
    partial_mp_func = _get_partial_mp_func(
        partial_mp_func, client=client, device_id=device_id
    )

    n_A = T_A.shape[0]
    if inspect.signature(partial_mp_func).parameters.get("normalize") is not None:
        # Normalized (stump-like)
        # Only normalized mp funcs can have a "normalize" parameter in its function
        # signature
        params = partial_mp_func.keywords
        T_A_subseq_isconstant = params.get("T_A_subseq_isconstant")
        T_B_subseq_isconstant = params.get("T_B_subseq_isconstant")
        P_ABBA[: n_A - m + 1] = partial_mp_func(
            T_A,
            m,
            T_B,
            ignore_trivial=False,
            T_A_subseq_isconstant=T_A_subseq_isconstant,
            T_B_subseq_isconstant=T_B_subseq_isconstant,
        )[:, 0]
        P_ABBA[n_A - m + 1 :] = partial_mp_func(
            T_B,
            m,
            T_A,
            ignore_trivial=False,
            T_A_subseq_isconstant=T_B_subseq_isconstant,
            T_B_subseq_isconstant=T_A_subseq_isconstant,
        )[:, 0]
    else:
        # Non-normalized (aamp-like)
        # Ignore/omit `T_A_subseq_isconstant` and `T_B_subseq_isconstant` parameters
        # for all non-normalized mp funcs
        P_ABBA[: n_A - m + 1] = partial_mp_func(T_A, m, T_B, ignore_trivial=False)[:, 0]
        P_ABBA[n_A - m + 1 :] = partial_mp_func(T_B, m, T_A, ignore_trivial=False)[:, 0]


def _mpdist(
    T_A,
    T_B,
    m,
    partial_mp_func,
    percentage=0.05,
    k=None,
    client=None,
    device_id=None,
    custom_func=None,
):
    """
    A convenience function for computing the matrix profile distance (MPdist) measure
    between any two time series.

    The MPdist distance measure considers two time series to be similar if they share
    many subsequences, regardless of the order of matching subsequences. MPdist
    concatenates the output of an AB-join and a BA-join and returns the `k`th smallest
    value as the reported distance. Note that MPdist is a measure and not a metric.
    Therefore, it does not obey the triangular inequality but the method is highly
    scalable.

    Parameters
    ----------
    T_A : numpy.ndarray
        The first time series or sequence for which to compute the matrix profile

    T_B : numpy.ndarray
        The second time series or sequence for which to compute the matrix profile

    m : int
        Window size

    partial_mp_func : functools.partial
        A generic matrix profile function that wraps extra parameters into
        `functools.partial` function.

    percentage : float, 0.05
       The percentage of distances that will be used to report `mpdist`. The value
        is between 0.0 and 1.0. This parameter is ignored when `k` is not `None` or when
        `k_func` is not None.

    k : int, default None
        Specify the `k`th value in the concatenated matrix profiles to return. When `k`
        is not `None`, then the `percentage` parameter is ignored. This parameter is
        ignored when `k_func` is not None.

    client : client, default None
        A Dask or Ray Distributed client. Setting up a distributed cluster is beyond
        the scope of this library. Please refer to the Dask or Ray Distributed
        documentation.

    device_id : int or list, default None
        The (GPU) device number to use. The default value is `0`. A list of
        valid device ids (``int``) may also be provided for parallel GPU-STUMP
        computation. A list of all valid device ids can be obtained by
        executing `[device.id for device in numba.cuda.list_devices()]`.

    custom_func : function, default None
        A custom user defined function for selecting the desired value from the
        unsorted `P_ABBA` array. This function may need to leverage `functools.partial`
        and should take `P_ABBA` as its only input parameter and return a single
        `MPdist` value. The `percentage` and `k` parameters are ignored when
        `custom_func` is not None.

    Returns
    -------
    MPdist : float
        The matrix profile distance

    Notes
    -----
    `DOI: 10.1109/ICDM.2018.00119 \
    <https://www.cs.ucr.edu/~eamonn/MPdist_Expanded.pdf>`__

    See Section III
    """
    n_A = T_A.shape[0]
    n_B = T_B.shape[0]
    P_ABBA = np.empty(n_A - m + 1 + n_B - m + 1, dtype=np.float64)

    _compute_P_ABBA(
        T_A,
        T_B,
        m,
        P_ABBA,
        partial_mp_func,
        client,
        device_id,
    )

    if k is not None:
        k = min(int(k), P_ABBA.shape[0] - 1)
    else:
        percentage = np.clip(percentage, 0.0, 1.0)
        k = min(math.ceil(percentage * (n_A + n_B)), n_A - m + 1 + n_B - m + 1 - 1)

    MPdist = _select_P_ABBA_value(P_ABBA, k, custom_func)

    return MPdist


def process_isconstant(T, m, T_subseq_isconstant=None, T_subseq_isfinite=None):
    """
    A convenience wrapper around the `rolling_isconstant` and
    `fix_isconstant_isfinite_conflicts`.

    It computes the rolling isconstant for 1-D and 2-D arrays. This is accomplished by
    comparing the min and max within each window and assigning `True` when the min and
    max are equal and `False` otherwise. If a subsequence contains at least one NaN,
    then the subsequence is not constant. If `T_subseq_isconstant` is provided as
    boolean array, its element will be set to False if their corresponding value in
    `T_subseq_isfinite` is False.

    Parameters
    ----------
    T : numpy.ndarray
        The input 1D or 2D array

    m : numpy.ndarray
        The rolling window size

    T_subseq_isconstant : numpy.ndarray, function, or list, default None
        A parameter that is used to show whether a subsequence of a time series in ``T``
        is constant (``True``) or not. ``T_subseq_isconstant`` can be a 1D or 2D
        boolean ``numpy.ndarray`` (depending on the dimension of ``T``) or a function
        that can be applied to each time series in ``T``. Alternatively, for  maximum
        flexibility, a list (with length equal to the total number of time series) may
        also be used. In this case, ``T_subseq_isconstant[i]`` corresponds to the
        ``i``-th time series ``T[i]`` and each element in the list can either be 1D
        boolean ``numpy.ndarray``, a function, or ``None``.

    T_subseq_isfinite : numpy.ndarray, default None
        A boolean array that indicates whether a subsequence in `T` contains a
        `np.nan`/`np.inf` value (False)

    Returns
    -------
    T_subseq_isconstant : numpy.ndarray
        Rolling window isconstant
    """
    if isinstance(T_subseq_isconstant, list):
        if T.ndim != 2:  # pragma: no cover
            msg = (
                "When `T_subseq_isconstant` is provided as a list, `T` "
                + f"must be a 2D array. Found {T.ndim} dimension instead."
            )
            raise ValueError(msg)

        if len(T_subseq_isconstant) != T.shape[0]:  # pragma: no cover
            msg = (
                "The length of the list `T_subseq_isconstant` must be "
                + "equal to the number of time series in `T`."
            )
            raise ValueError(msg)

        T_subseq_isconstant = np.array(
            [
                rolling_isconstant(T[i], m, T_subseq_isconstant[i])
                for i in range(T.shape[0])
            ]
        )
    else:
        T_subseq_isconstant = rolling_isconstant(T, m, T_subseq_isconstant)

    T_subseq_isconstant[...] = fix_isconstant_isfinite_conflicts(
        T, m, T_subseq_isconstant, T_subseq_isfinite
    )

    return T_subseq_isconstant


def deco_ray_tor(f):
    """
    Wraps a `numba` JIT-compiled function as a Python function.

    This "indirection" is required for Ray serialization to work.

    Parameters
    ----------
    f : function
        A `numba` JIT-compiled function

    Returns
    -------
    wrapper : function
        A Python function
    """

    def wrapper(*args):  # pragma: no cover
        return f(*args)

    return wrapper


def check_ray(ray_client):  # pragma: no cover
    """
    Check if Ray is initialized and, otherwise, raise an exception

    Due to the experimental nature of Ray support, a warning is
    also displayed.

    Parameters
    ----------
    ray_client : client
        A Ray client

    Returns
    -------
    None
    """
    if not ray_client.is_initialized():
        raise Exception("A Ray cluster could not be found!")

    ray_warning()


def ray_warning():
    """
    A generic warning for Ray support

    Parameters
    ----------
    None

    Returns
    -------
    None
    """
    msg = "Ray support is experimental and may be removed in the future.\n"
    msg += "Use at your own risk!"
    warnings.warn(msg)


def get_ray_nworkers(ray_client):
    """
    Return the total number of Ray workers in the cluster

    Parameters
    ----------
    ray_client : client
        A Ray client

    Returns
    -------
    nworkers : int
        Total number of Ray workers
    """
    return int(ray_client.cluster_resources().get("CPU"))


@njit(fastmath=config.STUMPY_FASTMATH_FLAGS)
def _update_incremental_PI(D, P, I, excl_zone, n_appended=0):
    """
    Given the 1D array distance profile, `D`, of the last subsequence of T,
    update (in-place) the (top-k) matrix profile, `P`, and the matrix profile
    index, I.

    Parameters
    ----------
    D : numpy.ndarray
        A 1D array (with dtype float) representing the distance profile of
        the last subsequence of T

    P : numpy.ndarray
        A 2D array representing the matrix profile of T,
        with shape (len(T) - m + 1, k), where `m` is the window size.
        P[-1, :] should be set to np.inf

    I : numpy.ndarray
        A 2D array representing the matrix profile index of T,
        with shape (len(T) - m + 1, k), where `m` is the window size
        I[-1, :] should be set to -1.

    excl_zone : int
        Size of the exclusion zone.

    n_appended : int
        Number of times the timeseries start point is shifted one to the right.
        See note below for more details.

    Returns
    -------
    None

    Note
    -----
    The `n_appended` parameter is used to indicate the number of times the timeseries
    start point is shifted one to the right. When `egress=False` (see stumpy.stumpi),
    the matrix profile and matrix profile index are updated in an incremental fashion
    while considering all historical data. `n_appended` must be set to 0 in such
    cases. However, when `egress=True`, the matrix profile and matrix profile index are
    updated in an incremental fashion and they represent the matrix profile and matrix
    profile index for the `l` most recent subsequences (where `l = len(T) - m + 1`).
    In this case, each subsequence is only compared against upto `l-1` left neighbors
    and upto `l-1` right neighbors.
    """
    _apply_exclusion_zone(D, D.shape[0] - 1, excl_zone, np.inf)

    update_idx = np.argwhere(D < P[:, -1]).flatten()
    for i in update_idx:
        idx = np.searchsorted(P[i], D[i], side="right")
        _shift_insert_at_index(P[i], idx, D[i])
        _shift_insert_at_index(I[i], idx, D.shape[0] + n_appended - 1)

    # Calculate the (top-k) matrix profile values/indidces
    # for the last subsequence
    P[-1] = np.inf
    I[-1] = -1
    for i, d in enumerate(D):
        if d < P[-1, -1]:
            idx = np.searchsorted(P[-1], d, side="right")
            _shift_insert_at_index(P[-1], idx, d)
            _shift_insert_at_index(I[-1], idx, i + n_appended)

    return
