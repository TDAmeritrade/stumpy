# STUMPY
# Copyright 2019 TD Ameritrade. Released under the terms of the 3-Clause BSD license.
# STUMPY is a trademark of TD Ameritrade IP Company, Inc. All rights reserved.

import logging

import numpy as np
from numba import njit, prange

from . import core

logger = logging.getLogger(__name__)


@njit(parallel=True, fastmath=True)
def _scrimp(T, m, μ, σ, orders, excl_zone, percentage=1.0):
    """
    A Numba JIT-compiled version of SCRIMP (self-join) for parallel computation
    of the matrix profile and matrix profile indices.

    Parameters
    ----------
    T : ndarray
        The time series or sequence for which to compute the matrix profile

    m : int
        Window size

    μ : ndarray
        Sliding window mean for T

    σ : ndarray
        Sliding window standard deviation for T

    orders : ndarray
        The order of diagonals to process and compute

    excl_zone : int
        The half width for the exclusion zone relative to the current
        sliding window

    percentage : float
        Approximate percentage completed. The value is between 0.0 and 1.0.

    Returns
    -------
    P : ndarray
        Matrix profile

    I : ndarray
        Matrix profile indices

    Notes
    -----
    `DOI: 10.1109/ICDM.2018.00099 \
    <https://www.cs.ucr.edu/~eamonn/SCRIMP_ICDM_camera_ready_updated.pdf>`__

    See Algorithm 1
    """

    n = len(T)
    l = n - m + 1
    P = np.empty((l,))
    I = np.empty((l,))

    P[:] = np.inf
    I[:] = -1

    n_dist_computed = 0

    for order in prange(orders.shape[0]):
        if n_dist_computed / (l * l) > percentage:  # pragma: no cover
            break
        k = orders[order]
        for i in range(0, n - m + 2 - k):
            if i == 0:
                QT = np.dot(T[i : i + m], T[i + k - 1 : i + k - 1 + m])
            else:
                QT = QT - T[i - 1] * T[i + k - 2] + T[i + m - 1] * T[i + k + m - 2]

            denom = m * σ[i] * σ[i + k - 1]
            if denom == 0:
                denom = 1e-10  # Avoid divide by zero
            D = np.sqrt(np.abs(2 * m * (1.0 - (QT - m * μ[i] * μ[i + k - 1]) / denom)))
            threshold = 1e-7
            if σ[i] < threshold:  # pragma: no cover
                D = np.sqrt(m)
            if σ[i + k - 1] < threshold:  # pragma: no cover
                D = np.sqrt(m)
            if σ[i] < threshold and σ[i + k - 1] < threshold:
                D = 0

            if i < i + k - 1 - excl_zone and D < P[i]:
                P[i] = D
                I[i] = i + k - 1

            if i < i + k - 1 - excl_zone and D < P[i + k - 1]:
                P[i + k - 1] = D
                I[i + k - 1] = i

            n_dist_computed = n_dist_computed + 1

    return P, I


def scrimp(T, m, percentage=1.0):
    """
    Compute the matrix profile with parallelized SCRIMP (self-join)

    This is a convenience wrapper around the Numba JIT-compiled parallelized
    `_scrimp` function which computes the matrix profile according to SCRIMP.

    Parameters
    ----------
    T : ndarray
        The time series or sequence for which to compute the matrix profile

    m : int
        Window size

    percentage : float
        Approximate percentage completed. The value is between 0.0 and 1.0.

    Returns
    -------
    out : ndarray
        Matrix profile and matrix profile indices

    Notes
    -----
    `DOI: 10.1109/ICDM.2018.00099 \
    <https://www.cs.ucr.edu/~eamonn/SCRIMP_ICDM_camera_ready_updated.pdf>`__

    See Algorithm 1
    """

    T = np.asarray(T)
    core.check_dtype(T)
    core.check_nan(T)

    if T.ndim != 1:  # pragma: no cover
        raise ValueError(f"T is {T.ndim}-dimensional and must be 1-dimensional. ")

    core.check_window_size(m)

    n = len(T)
    l = n - m + 1
    out = np.empty((l, 2), dtype=object)

    μ, σ = core.compute_mean_std(T, m)
    excl_zone = int(np.ceil(m / 4))
    orders = np.random.permutation(range(excl_zone + 1, n - m + 2))

    P, I = _scrimp(T, m, μ, σ, orders, excl_zone, percentage)

    out[:, 0] = P
    out[:, 1] = I

    threshold = 10e-6
    if core.are_distances_too_small(out[:, 0], threshold=threshold):  # pragma: no cover
        logger.warning(f"A large number of values are smaller than {threshold}.")
        logger.warning("For a self-join, try setting `ignore_trivial = True`.")

    return out
