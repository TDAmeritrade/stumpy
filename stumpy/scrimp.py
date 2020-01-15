# STUMPY
# Copyright 2019 TD Ameritrade. Released under the terms of the 3-Clause BSD license.
# STUMPY is a trademark of TD Ameritrade IP Company, Inc. All rights reserved.

import logging

import numpy as np
from numba import njit, prange

from . import core

logger = logging.getLogger(__name__)


@njit(parallel=True, fastmath=True)
def _scrimp(T, m, excl_zone):
    """
    A Numba JIT-compiled version of SCRIMP (self-join) for parallel computation
    of the matrix profile and matrix profile indices.

    Parameters
    ----------
    T : ndarray
        The time series or sequence for which to compute the matrix profile

    m : int
        Window size

    excl_zone : int
        The half width for the exclusion zone relative to the current
        sliding window

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
    P = np.ones(l, dtype=np.float64) * np.inf
    I = np.ones(l, dtype=np.int64) * -1
    orders = np.random.permutation(range(excl_zone + 1, n - m + 2))
    μ, σ = core.compute_mean_std(T, m)

    for order in prange(orders.shape[0]):
        k = orders[order]
        for i in range(0, n - m + 2 - k):
            if i == 0:
                QT = np.dot(T[i : i + m], T[i + k - 1 : i + k - 1 + m])
            else:
                QT = QT - T[i - 1] * T[i + k - 2] + T[i + m - 1] * T[i + k + m - 2]

            D = core.calculate_distance_profile(
                m,
                QT,
                μ[i],
                σ[i],
                np.atleast_2d(μ[i + k - 1]),
                np.atleast_2d(σ[i + k - 1]),
            )

            if D < P[i]:
                P[i] = D[0][0]
                I[i] = i + k - 1

            if i < i + k - 1 - excl_zone and D < P[i + k - 1]:
                P[i + k - 1] = D[0][0]
                I[i + k - 1] = i

    return P, I


def scrimp(T, m):
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

    excl_zone = int(np.ceil(m / 4))

    P, I = _scrimp(T, m, excl_zone)

    out[:, 0] = P
    out[:, 1] = I

    threshold = 10e-6
    if core.are_distances_too_small(out[:, 0], threshold=threshold):  # pragma: no cover
        logger.warning(f"A large number of values are smaller than {threshold}.")
        logger.warning("For a self-join, try setting `ignore_trivial = True`.")

    return out
