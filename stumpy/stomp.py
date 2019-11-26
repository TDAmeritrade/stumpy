# STUMPY
# Copyright 2019 TD Ameritrade. Released under the terms of the 3-Clause BSD license.
# STUMPY is a trademark of TD Ameritrade IP Company, Inc. All rights reserved.

import numpy as np
from . import core, stamp
import logging

logger = logging.getLogger(__name__)


def stomp(T_A, m, T_B=None, ignore_trivial=True):
    """
    Compute "Scalable Time series Ordered-search Matrix Profile" (STOMP)

    Parameters
    ----------
    T_A : ndarray
        The time series or sequence for which the matrix profile index will
        be returned

    T_B : ndarray
        The time series or sequence that contain your query subsequences

    m : int
        Window size

    ignore_trivial : bool
        `True` if this is a self join and `False` otherwise (i.e., AB-join).

    Returns
    -------
    out : ndarray
        A four column numpy array where the first column is the matrix profile,
        the second column is the matrix profile indices. The third and fourth
        columns are the left and right matrix profile indices, respectively.

    Notes
    -----
    `DOI: 10.1109/ICDM.2016.0085 \
    <https://www.cs.ucr.edu/~eamonn/STOMP_GPU_final_submission_camera_ready.pdf>`__

    See Table II

    Timeseries, T_B, will be annotated with the distance location
    (or index) of all its subsequences in another times series, T_A.

    For every subsequence, Q, in T_B, you will get a distance
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
    core.check_dtype(T_A)
    core.check_nan(T_A)
    if T_B is None:
        T_B = T_A
    core.check_dtype(T_B)
    core.check_nan(T_B)
    core.check_window_size(m)

    if ignore_trivial is False and core.are_arrays_equal(T_A, T_B):  # pragma: no cover
        logger.warning("Arrays T_A, T_B are equal, which implies a self-join.")
        logger.warning("Try setting `ignore_trivial = True`.")

    if ignore_trivial and core.are_arrays_equal(T_A, T_B) is False:  # pragma: no cover
        logger.warning("Arrays T_A, T_B are not equal, which implies an AB-join.")
        logger.warning("Try setting `ignore_trivial = False`.")

    n = T_B.shape[0]
    l = n - m + 1
    excl_zone = int(np.ceil(m / 4))  # See Definition 3 and Figure 3
    M_T, Σ_T = core.compute_mean_std(T_A, m)
    QT = core.sliding_dot_product(T_B[:m], T_A)
    QT_first = core.sliding_dot_product(T_A[:m], T_B)

    μ_Q, σ_Q = core.compute_mean_std(T_B, m)

    out = np.empty((l, 4), dtype=object)

    # Handle first subsequence, add exclusionary zone
    if ignore_trivial:
        P, I = stamp.mass(T_B[:m], T_A, M_T, Σ_T, 0, excl_zone)
        PR, IR = stamp.mass(T_B[:m], T_A, M_T, Σ_T, 0, excl_zone, right=True)
    else:
        P, I = stamp.mass(T_B[:m], T_A, M_T, Σ_T)
        IR = -1  # No left and right matrix profile available
    out[0] = P, I, -1, IR

    k = T_A.shape[0] - m + 1
    for i in range(1, l):
        QT[1:] = (
            QT[: k - 1] - T_B[i - 1] * T_A[: k - 1] + T_B[i - 1 + m] * T_A[-(k - 1) :]
        )
        QT[0] = QT_first[i]
        D = core.calculate_distance_profile(m, QT, μ_Q[i], σ_Q[i], M_T, Σ_T)
        if ignore_trivial:
            zone_start = max(0, i - excl_zone)
            zone_stop = min(k, i + excl_zone)
            D[zone_start:zone_stop] = np.inf
        I = np.argmin(D)
        P = D[I]

        # Get left and right matrix profiles for self-joins
        if ignore_trivial and i > 0:
            IL = np.argmin(D[:i])
            if zone_start <= IL < zone_stop:  # pragma: no cover
                IL = -1
        else:
            IL = -1

        if ignore_trivial and i + 1 < D.shape[0]:
            IR = i + 1 + np.argmin(D[i + 1 :])
            if zone_start <= IR < zone_stop:  # pragma: no cover
                IR = -1
        else:
            IR = -1

        out[i] = P, I, IL, IR

    threshold = 10e-6
    if core.are_distances_too_small(out[:, 0], threshold=threshold):  # pragma: no cover
        logger.warning(f"A large number of values are smaller than {threshold}.")
        logger.warning("For a self-join, try setting `ignore_trivial = True`.")

    return out
