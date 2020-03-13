# STUMPY
# Copyright 2019 TD Ameritrade. Released under the terms of the 3-Clause BSD license.
# STUMPY is a trademark of TD Ameritrade IP Company, Inc. All rights reserved.

import logging

import numpy as np
from numba import njit, prange

from . import core

logger = logging.getLogger(__name__)


def _multi_mass(Q, T, m, M_T, Σ_T):
    """
    A multi-dimensional wrapper around "Mueen's Algorithm for Similarity Search"
    (MASS) to compute multi-dimensional distance profile.

    Parameters
    ----------
    Q : ndarray
        Query array or subsequence

    T : ndarray
        Time series array or sequence

    m : int
        Window size

    M_T : ndarray
        Sliding mean for `T`

    Σ_T : ndarray
        Sliding standard deviation for `T`

    Returns
    -------
    D : ndarray
        Multi-dimensional distance profile
    """

    d, n = T.shape
    k = n - m + 1

    D = np.empty((d, k), dtype="float64")

    for i in range(d):
        D[i, :] = core.mass(Q[i], T[i], M_T[i], Σ_T[i])

    # Column-wise sort
    D = np.sort(D, axis=0)

    D_prime = np.zeros(k)
    for i in range(d):
        D_prime[:] = D_prime + D[i]
        D[i, :] = D_prime / (i + 1)

    return D


def _get_first_mstump_profile(start, T_A, T_B, m, excl_zone, M_T, Σ_T):
    """
    Multi-dimensional wrapper to compute the multi-dimensional matrix profile
    and multi-dimensional matrix profile index for a given window within the
    times series or sequence that is denote by the `start` index.
    Essentially, this is a convenience wrapper around `_multi_mass`

    Parameters
    ----------
    start : int
        The window index to calculate the first matrix profile, matrix profile
        index, left matrix profile index, and right matrix profile index for.

    T_A : ndarray
        The time series or sequence for which the matrix profile index will
        be returned

    T_B : ndarray
        The time series or sequence that contains your query subsequences

    m : int
        Window size

    excl_zone : int
        The half width for the exclusion zone relative to the `start`.

    M_T : ndarray
        Sliding mean for `T`

    Σ_T : ndarray
        Sliding standard deviation for `T`

    Returns
    -------
    P : ndarray
        Multi-dimensional matrix profile for the window with index equal to
        `start`
    I : ndarray
        Multi-dimensional matrix profile index for the window with index
        equal to `start`
    """

    d, n = T_A.shape
    D = _multi_mass(T_B[:, start : start + m], T_A, m, M_T, Σ_T)

    zone_start = max(0, start - excl_zone)
    zone_stop = min(n - m + 1, start + excl_zone)
    D[:, zone_start : zone_stop + 1] = np.inf

    P = np.full(d, np.inf, dtype="float64")
    I = np.ones(d, dtype="int64") * -1

    for i in range(d):
        min_index = np.argmin(D[i])
        I[i] = min_index
        P[i] = D[i, min_index]
        if np.isinf(P[i]):  # pragma nocover
            I[i] = -1

    return P, I


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
        Given `start`, return the corresponding multi-dimensional QT

    QT_first : ndarray
        Multi-dimensional QT for the first window
    """

    d = T.shape[0]
    k = T.shape[1] - m + 1

    QT = np.empty((d, k), dtype="float64")
    QT_first = np.empty((d, k), dtype="float64")

    for i in range(d):
        QT[i] = core.sliding_dot_product(T[i, start : start + m], T[i])
        QT_first[i] = core.sliding_dot_product(T[i, :m], T[i])

    return QT, QT_first


@njit(parallel=True, fastmath=True)
def _mstump(
    T, m, range_stop, excl_zone, M_T, Σ_T, QT, QT_first, μ_Q, σ_Q, k, range_start=1
):
    """
    A Numba JIT-compiled version of mSTOMP, a variant of mSTAMP, for parallel
    computation of the multi-dimensional matrix profile and multi-dimensional
    matrix profile indices. Note that only self-joins are supported.

    Parameters
    ----------
    T: ndarray
        The time series or sequence for which to compute the multi-dimensional
        matrix profile

    m : int
        Window size

    range_stop : int
        The index value along T for which to stop the matrix profile
        calculation. This parameter is here for consistency with the
        distributed `mstumped` algorithm.

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

    range_start : int
        The starting index value along T_B for which to start the matrix
        profile calculation. Default is 1.

    Returns
    -------
    P : ndarray
        The multi-dimensional matrix profile. Each row of the array corresponds
        to each matrix profile for a given dimension (i.e., the first row is the
        1-D matrix profile and the second row is the 2-D matrix profile).
    I : ndarray
        The multi-dimensional matrix profile index where each row of the array
        corresponds to each matrix profile index for a given dimension.

    Notes
    -----

    `DOI: 10.1109/ICDM.2017.66 \
    <https://www.cs.ucr.edu/~eamonn/Motif_Discovery_ICDM.pdf>`__

    See mSTAMP Algorithm
    """

    QT_odd = QT.copy()
    QT_even = QT.copy()
    d = T.shape[0]

    P = np.empty((d, range_stop - range_start))
    I = np.empty((d, range_stop - range_start))
    D = np.empty((d, k))
    D_prime = np.empty(k)

    for idx in range(range_start, range_stop):
        for i in range(d):
            # Numba's prange requires incrementing a range by 1 so replace
            # `for j in range(k-1,0,-1)` with its incrementing compliment
            for rev_j in prange(1, k):
                j = k - rev_j
                # GPU Stomp Parallel Implementation with Numba
                # DOI: 10.1109/ICDM.2016.0085
                # See Figure 5
                if idx % 2 == 0:
                    # Even
                    QT_even[i, j] = (
                        QT_odd[i, j - 1]
                        - T[i, idx - 1] * T[i, j - 1]
                        + T[i, idx + m - 1] * T[i, j + m - 1]
                    )
                else:
                    # Odd
                    QT_odd[i, j] = (
                        QT_even[i, j - 1]
                        - T[i, idx - 1] * T[i, j - 1]
                        + T[i, idx + m - 1] * T[i, j + m - 1]
                    )

            if idx % 2 == 0:
                QT_even[i, 0] = QT_first[i, idx]
                D[i] = core._calculate_squared_distance_profile(
                    m, QT_even[i], μ_Q[i, idx], σ_Q[i, idx], M_T[i], Σ_T[i]
                )
            else:
                QT_odd[i, 0] = QT_first[i, idx]
                D[i] = core._calculate_squared_distance_profile(
                    m, QT_odd[i], μ_Q[i, idx], σ_Q[i, idx], M_T[i], Σ_T[i]
                )

        zone_start = max(0, idx - excl_zone)
        zone_stop = min(k, idx + excl_zone)
        D[:, zone_start : zone_stop + 1] = np.inf

        D = np.sqrt(D)

        # Column-wise sort
        for col in prange(k):
            D[:, col] = np.sort(D[:, col])

        D_prime[:] = 0.0
        for i in range(d):
            D_prime = D_prime + D[i]

            min_index = np.argmin(D_prime)
            pos = idx - range_start
            I[i, pos] = min_index
            P[i, pos] = D_prime[min_index] / (i + 1)
            if np.isinf(P[i, pos]):  # pragma nocover
                I[i, pos] = -1

    return P, I


def mstump(T, m):
    """
    Compute the multi-dimensional matrix profile with parallelized mSTOMP

    This is a convenience wrapper around the Numba JIT-compiled parallelized
    `_mstump` function which computes the multi-dimensional matrix profile and
    multi-dimensional matrix profile index according to mSTOMP, a variant of
    mSTAMP. Note that only self-joins are supported.

    Parameters
    ----------
    T : ndarray
        The time series or sequence for which to compute the multi-dimensional
        matrix profile. Each row in `T` represents data from a different
        dimension while each column in `T` represents data from the same
        dimension.

    m : int
        Window size

    Returns
    -------
    P : ndarray
        The multi-dimensional matrix profile. Each column of the array corresponds
        to each matrix profile for a given dimension (i.e., the first column is
        the 1-D matrix profile and the second column is the 2-D matrix profile).

    I : ndarray
        The multi-dimensional matrix profile index where each column of the array
        corresponds to each matrix profile index for a given dimension.

    Notes
    -----

    `DOI: 10.1109/ICDM.2017.66 \
    <https://www.cs.ucr.edu/~eamonn/Motif_Discovery_ICDM.pdf>`__

    See mSTAMP Algorithm
    """

    T_A = np.asarray(core.transpose_dataframe(T)).copy()
    T_B = T_A.copy()

    T_A[np.isinf(T_A)] = np.nan
    T_B[np.isinf(T_B)] = np.nan

    core.check_dtype(T_A)
    if T_A.ndim <= 1:  # pragma: no cover
        err = f"T is {T_A.ndim}-dimensional and must be at least 1-dimensional"
        raise ValueError(f"{err}")

    core.check_window_size(m)

    d = T_A.shape[0]
    n = T_A.shape[1]
    k = n - m + 1
    excl_zone = int(np.ceil(m / 4))  # See Definition 3 and Figure 3

    M_T, Σ_T = core.compute_mean_std(T_A, m)
    μ_Q, σ_Q = core.compute_mean_std(T_B, m)

    T_A[np.isnan(T_A)] = 0

    P = np.empty((d, k), dtype="float64")
    I = np.empty((d, k), dtype="int64")

    start = 0
    stop = k

    P[:, start], I[:, start] = _get_first_mstump_profile(
        start, T_A, T_B, m, excl_zone, M_T, Σ_T
    )

    T_B[np.isnan(T_B)] = 0

    QT, QT_first = _get_multi_QT(start, T_A, m)

    P[:, start + 1 : stop], I[:, start + 1 : stop] = _mstump(
        T_A, m, stop, excl_zone, M_T, Σ_T, QT, QT_first, μ_Q, σ_Q, k, start + 1
    )

    return P.T, I.T
