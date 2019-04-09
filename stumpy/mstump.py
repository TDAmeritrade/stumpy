import numpy as np
from . import core, stamp
from . import _calculate_squared_distance_profile
from numba import njit, prange
import logging

logger = logging.getLogger(__name__)

def _get_first_mstump_profile(start, T, m, excl_zone, M_T, Σ_T, 
                             ignore_trivial):
    """
    multi-dimensional wrapper to compute the matrix profile, matrix profile index, 
    left matrix profile index, and right matrix profile index for given window 
    within the times series or sequence that is denote by the `start` index. 
    Essentially, this is a convenience wrapper around `stamp.multi_mass`

    Parameters
    ----------
    start : int
        The window index to calculate the first matrix profile, matrix profile
        index, left matrix profile index, and right matrix profile index for.

    T : ndarray
        The time series or sequence for which the matrix profile index will 
        be returned

    m : int
        Window size

    excl_zone : int
        The half width for the exclusion zone relative to the `start`.

    M_T : ndarray
        Sliding mean for `T`

    Σ_T : ndarray
        Sliding standard deviation for `T`

    ignore_trivial : bool
        `True` if this is a self join and `False` otherwise (i.e., AB-join).

    Returns
    -------
    P : float64
        Matrix profile for the window with index equal to `start`

    I : int64
        Matrix profile index for the window with index equal to `start`
    """
    ndims = T.shape[0]
    P = np.empty(ndims, dtype='float64')
    I = np.empty(ndims, dtype='int64')
    IL = np.empty(ndims, dtype='int64')
    IR = np.empty(ndims, dtype='int64')

    # Handle first subsequence, add exclusionary zone
    if ignore_trivial:
        P, I = stamp.multi_mass(T[:, start:start+m], T, M_T, Σ_T, 
                          start, excl_zone)
        PL, IL = stamp.multi_mass(T[:, start:start+m], T, M_T, Σ_T, 
                            start, excl_zone, left=True)        
        PR, IR = stamp.multi_mass(T[:, start:start+m], T, M_T, Σ_T, 
                            start, excl_zone, right=True)
    else:
        P, I = stamp.mass(T[:, start:start+m], T, M_T, Σ_T)
        # No left and right matrix profile available
        IL = -1
        IR = -1

    return P, np.array([I, IL, IR]).T

def _get_multi_QT(start, T, m):
    """
    Multi-dimensional wrapper to compute the sliding dot product between 
    the query, `T[:, start:start+m])` and the time series, `T`. 
    Additionally, compute QT for the first window.

    Parameters
    ----------
    start : int
        The window index for T_B from which to calculate the QT dot product

    T : ndarray
        The time series or sequence for which to compute the dot product

    m : int
        Window size

    Returns
    ------- 
    QT : ndarray
        Given `start`, return the corresponding QT

    QT_first : ndarray
         QT for the first window
    """

    ndims = T.shape[0]
    k = T.shape[1]-m+1

    QT = np.empty((ndims, k), dtype='float64')
    QT_first = np.empty((ndims, k), dtype='float64')

    for dim in range(ndims):
        QT[dim] = core.sliding_dot_product(T[dim, start:start+m], T[dim])
        QT_first[dim] = core.sliding_dot_product(T[dim, :m], T[dim])

    return QT, QT_first

@njit(parallel=True, fastmath=True) 
def _mstump(T_A, T_B, m, range_stop, excl_zone, 
           M_T, Σ_T, QT, QT_first, μ_Q, σ_Q, k, 
           ignore_trivial=True, range_start=1):
    """
    A Numba JIT-compiled version of STOMP for parallel computation of the
    matrix profile, matrix profile indices, left matrix profile indices, 
    and right matrix profile indices.

    Parameters
    ----------
    T_A : ndarray
        The time series or sequence for which to compute the matrix profile

    T_B : ndarray
        The time series or sequence that contain your query subsequences
        of interest

    m : int
        Window size

    range_stop : int
        The index value along T_B for which to stop the matrix profile 
        calculation. This parameter is here for consistency with the 
        distributed `stumped` algorithm.

    excl_zone : int
        The half width for the exclusion zone relative to the current
        sliding window

    M_T : ndarray
        Sliding mean of time series, `T`

    Σ_T : ndarray
        Sliding standard deviation of time series, `T`

    QT : ndarray
        Dot product between some query sequence,`Q`, and time series, `T`

    QT_first : ndarray
        QT for the first window relative to the current sliding window

    μ_Q : ndarray
        Mean of the query sequence, `Q`, relative to the current sliding window

    σ_Q : ndarray
        Standard deviation of the query sequence, `Q`, relative to the current
        sliding window

    k : int
        The total number of sliding windows to iterate over

    ignore_trivial : bool
        Set to `True` if this is a self-join. Otherwise, for AB-join, set this to
        `False`. Default is `True`.

    range_start : int
        The starting index value along T_B for which to start the matrix
        profile claculation. Default is 1.

    Returns
    -------
    profile : ndarray
        Matrix profile

    indices : ndarray
        The first column consists of the matrix profile indices, the second
        column consists of the left matrix profile indices, and the third
        column consists of the right matrix profile indices.

    Notes
    -----
    DOI: 10.1109/ICDM.2016.0085
    See Table II

    Timeseries, T_B, will be annotated with the distance location
    (or index) of all its subsequences in another times series, T_A.

    Return: For every subsequence, Q, in T_B, you will get a distance
    and index for the closest subsequence in T_A. Thus, the array
    returned will have length T_B.shape[0]-m+1. Additionally, the 
    left and right matrix profiles are also returned.

    Note: Unlike in the Table II where T_A.shape is expected to be equal 
    to T_B.shape, this implementation is generalized so that the shapes of 
    T_A and T_B can be different. In the case where T_A.shape == T_B.shape, 
    then our algorithm reduces down to the same algorithm found in Table II. 

    Additionally, unlike STAMP where the exclusion zone is m/2, the default 
    exclusion zone for STOMP is m/4 (See Definition 3 and Figure 3).

    For self-joins, set `ignore_trivial = True` in order to avoid the 
    trivial match. 

    Note that left and right matrix profiles are only available for self-joins.
    """

    QT_odd = QT.copy()
    QT_even = QT.copy()
    profile = np.empty((range_stop-range_start,))  # float64
    indices = np.empty((range_stop-range_start, 3))  # int64
    
    for i in range(range_start, range_stop):
        # Numba's prange requires incrementing a range by 1 so replace
        # `for j in range(k-1,0,-1)` with its incrementing compliment
        for rev_j in prange(1, k):
            j = k - rev_j
            # GPU Stomp Parallel Implementation with Numba
            # DOI: 10.1109/ICDM.2016.0085
            # See Figure 5
            if i % 2 == 0:
                # Even
                QT_even[j] = QT_odd[j-1] - T_B[i-1]*T_A[j-1] + T_B[i+m-1]*T_A[j+m-1]
            else:
                # Odd
                QT_odd[j] = QT_even[j-1] - T_B[i-1]*T_A[j-1] + T_B[i+m-1]*T_A[j+m-1]

        if i % 2 == 0:
            QT_even[0] = QT_first[i]
            D = _calculate_squared_distance_profile(m, QT_even, μ_Q[i], σ_Q[i], M_T, Σ_T)
        else:
            QT_odd[0] = QT_first[i]
            D = _calculate_squared_distance_profile(m, QT_odd, μ_Q[i], σ_Q[i], M_T, Σ_T)

        if ignore_trivial:
            zone_start = max(0, i-excl_zone)
            zone_stop = i+excl_zone+1
            D[zone_start:zone_stop] = np.inf
        I = np.argmin(D)
        P = np.sqrt(D[I])

        # Get left and right matrix profiles for self-joins
        if ignore_trivial and i > 0:
            IL = np.argmin(D[:i])
            if zone_start <= IL < zone_stop:
                IL = -1
        else:
            IL = -1

        if ignore_trivial and i+1 < D.shape[0]:
            IR = i + 1 + np.argmin(D[i+1:])
            if zone_start <= IR < zone_stop:
                IR = -1
        else:
            IR = -1

        # Only a part of the profile/indices array are passed
        profile[i-range_start] = P
        indices[i-range_start] = I, IL, IR
    
    return profile, indices

def mstump(T, m):
    """
    This is a convenience wrapper around the Numba JIT-compiled parallelized 
    `_stump` function which computes the matrix profile according to STOMP.

    Parameters
    ----------
    T : ndarray
        The time series or sequence for which to compute the matrix profile
    m : int
        Window size

    Returns
    -------
    out : ndarray
        The first column consists of the matrix profile, the second column 
        consists of the matrix profile indices, the third column consists of 
        the left matrix profile indices, and the fourth column consists of 
        the right matrix profile indices.

    Notes
    -----

    DOI: 10.1109/ICDM.2016.0085
    See Table II

    Timeseries, T_B, will be annotated with the distance location
    (or index) of all its subsequences in another times series, T_A.

    Return: For every subsequence, Q, in T_B, you will get a distance
    and index for the closest subsequence in T_A. Thus, the array
    returned will have length T_B.shape[0]-m+1. Additionally, the 
    left and right matrix profiles are also returned.

    Note: Unlike in the Table II where T_A.shape is expected to be equal 
    to T_B.shape, this implementation is generalized so that the shapes of 
    T_A and T_B can be different. In the case where T_A.shape == T_B.shape, 
    then our algorithm reduces down to the same algorithm found in Table II. 

    Additionally, unlike STAMP where the exclusion zone is m/2, the default 
    exclusion zone for STOMP is m/4 (See Definition 3 and Figure 3).

    For self-joins, set `ignore_trivial = True` in order to avoid the 
    trivial match. 

    Note that left and right matrix profiles are only available for self-joins.
    """

    ignore_trivial = True
    core.check_dtype(T)
    
    ndims = T.shape[0]
    n = T.shape[1]
    k = T.shape[1]-m+1
    l = n-m+1
    zone = int(np.ceil(m/4))  # See Definition 3 and Figure 3

    M_T, Σ_T = core.multi_compute_mean_std(T, m)
    μ_Q, σ_Q = core.multi_compute_mean_std(T, m)

    out = np.empty((ndims, l, 4), dtype=object)
    profile = np.empty((ndims, l,), dtype='float64')
    indices = np.empty((ndims, l, 3), dtype='int64')

    start = 0
    stop = l

    profile[:, start], indices[:, start, :] = \
        _get_first_mstump_profile(start, T, m, zone, M_T, 
                                 Σ_T, ignore_trivial)

    QT, QT_first = _get_multi_QT(start, T, m)

    for dim in range(ndims):
        profile[dim, start+1:stop], indices[dim, start+1:stop, :] = \
            _mstump(T[dim], T[dim], m, stop, zone, M_T[dim], Σ_T[dim], QT[dim], QT_first[dim], μ_Q[dim],
                   σ_Q[dim], k, ignore_trivial, start+1)

    out[:, :, 0] = profile
    out[:, :, 1:4] = indices
    
    threshold = 10e-6
    if core.are_distances_too_small(out[:, 0], threshold=threshold):  # pragma: no cover
        logger.warning(f"A large number of values are smaller than {threshold}.")
        logger.warning("For a self-join, try setting `ignore_trivial = True`.")
        
    return out
