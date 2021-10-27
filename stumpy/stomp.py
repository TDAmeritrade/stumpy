# STUMPY
# Copyright 2019 TD Ameritrade. Released under the terms of the 3-Clause BSD license.
# STUMPY is a trademark of TD Ameritrade IP Company, Inc. All rights reserved.

import logging

import numpy as np

from . import core, stamp, config

logger = logging.getLogger(__name__)


def _stomp(T_A, m, T_B=None, ignore_trivial=True):
    """
    Compute "Scalable Time series Ordered-search Matrix Profile" (STOMP)

    Parameters
    ----------
    T_A : numpy.ndarray
        The time series or sequence for which the matrix profile index will
        be returned

    m : int
        Window size

    T_B : numpy.ndarray, default None
        The time series or sequence that will be used to annotate T_A. For every
        subsequence in T_A, its nearest neighbor in T_B will be recorded.

    ignore_trivial : bool, default True
        `True` if this is a self join and `False` otherwise (i.e., AB-join).

    Returns
    -------
    out : numpy.ndarray
        A four column numpy array where the first column is the matrix profile,
        the second column is the matrix profile indices. The third and fourth
        columns are the left and right matrix profile indices, respectively.

    Notes
    -----
    `DOI: 10.1109/ICDM.2016.0085 \
    <https://www.cs.ucr.edu/~eamonn/STOMP_GPU_final_submission_camera_ready.pdf>`__

    See Table II

    Timeseries, T_A, will be annotated with the distance location
    (or index) of all its subsequences in another times series, T_B.

    For every subsequence, Q, in T_A, you will get a distance
    and index for the closest subsequence in T_A. Thus, the array
    returned will have length T_A.shape[0]-m+1. Additionally, the
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
    logger.warning(
        "stumpy.stomp._stomp is not supported and only provided for reference."
    )
    logger.warning(
        "Please use the Numba JIT-compiled stumpy.stump or stumpy.gpu_stump instead."
    )

    if T_B is None:
        T_B = T_A
        ignore_trivial = True

    T_A, μ_Q, σ_Q = core.preprocess(T_A, m)
    T_B, M_T, Σ_T = core.preprocess(T_B, m)

    if T_A.ndim != 1:  # pragma: no cover
        raise ValueError(f"T_A is {T_A.ndim}-dimensional and must be 1-dimensional. ")

    if T_B.ndim != 1:  # pragma: no cover
        raise ValueError(f"T_B is {T_B.ndim}-dimensional and must be 1-dimensional. ")

    core.check_window_size(m, max_size=min(T_A.shape[0], T_B.shape[0]))

    if ignore_trivial is False and core.are_arrays_equal(T_A, T_B):  # pragma: no cover
        logger.warning("Arrays T_A, T_B are equal, which implies a self-join.")
        logger.warning("Try setting `ignore_trivial = True`.")

    if ignore_trivial and core.are_arrays_equal(T_A, T_B) is False:  # pragma: no cover
        logger.warning("Arrays T_A, T_B are not equal, which implies an AB-join.")
        logger.warning("Try setting `ignore_trivial = False`.")

    n = T_A.shape[0]
    l = n - m + 1
    excl_zone = int(
        np.ceil(m / config.STUMPY_EXCL_ZONE_DENOM)
    )  # See Definition 3 and Figure 3

    out = np.empty((l, 4), dtype=object)

    # Handle first subsequence, add exclusionary zone
    if np.isinf(μ_Q[0]):
        P = np.inf
        I = -1
        IR = -1
    else:
        if ignore_trivial:
            P, I = stamp._mass_PI(T_A[:m], T_B, M_T, Σ_T, 0, excl_zone)
            PR, IR = stamp._mass_PI(T_A[:m], T_B, M_T, Σ_T, 0, excl_zone, right=True)
        else:
            P, I = stamp._mass_PI(T_A[:m], T_B, M_T, Σ_T)
            IR = -1  # No left and right matrix profile available

    out[0] = P, I, -1, IR

    QT = core.sliding_dot_product(T_A[:m], T_B)
    QT_first = core.sliding_dot_product(T_B[:m], T_A)

    k = T_B.shape[0] - m + 1
    for i in range(1, l):
        QT[1:] = (
            QT[: k - 1] - T_A[i - 1] * T_B[: k - 1] + T_A[i - 1 + m] * T_B[-(k - 1) :]
        )
        QT[0] = QT_first[i]

        D = core._calculate_squared_distance_profile(
            m, QT, μ_Q[i].item(0), σ_Q[i].item(0), M_T, Σ_T
        )
        if ignore_trivial:
            core.apply_exclusion_zone(D, i, excl_zone)

        I = np.argmin(D)
        P = np.sqrt(D[I])
        if P == np.inf:
            I = -1

        # Get left and right matrix profiles
        IL = -1
        PL = np.inf
        if ignore_trivial and i > 0:
            IL = np.argmin(D[:i])
            PL = D[IL]
        if PL == np.inf:
            IL = -1

        IR = -1
        PR = np.inf
        if ignore_trivial and i + 1 < D.shape[0]:
            IR = i + 1 + np.argmin(D[i + 1 :])
            PR = D[IR]
        if PR == np.inf:
            IR = -1

        out[i] = P, I, IL, IR

    threshold = 10e-6
    if core.are_distances_too_small(out[:, 0], threshold=threshold):  # pragma: no cover
        logger.warning(f"A large number of values are smaller than {threshold}.")
        logger.warning("For a self-join, try setting `ignore_trivial = True`.")

    return out
