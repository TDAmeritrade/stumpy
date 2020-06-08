# STUMPY
# Copyright 2019 TD Ameritrade. Released under the terms of the 3-Clause BSD license.
# STUMPY is a trademark of TD Ameritrade IP Company, Inc. All rights reserved.

import numpy as np
from stumpy import core
import stumpy


def stumpi_init(T, m):  # pragma: no cover
    """
    A helper function that generates the initial inputs needed for `stumpy.stumpi`

    Parameters
    ----------
    T : ndarray
        The time series or sequence for which the matrix profile and matrix profile
        indices will be returned

    m : int
        Window size

    Returns
    -------
    T : ndarray
        The time series or sequence for which the matrix profile and matrix profile
        indices will be returned

    m : int
        Window size

    P : ndarray
        The matrix profile for `T`

    I : ndarray
        The matrix profile indices for `T`

    QT : ndarray
        Dot product between some query sequence,`Q`, and time series, `T`

    M_T : ndarray
        Sliding mean for `T`

    Σ_T : ndarray
        Sliding standard deviation for `T`
    """
    T, M_T, Σ_T = core.preprocess(T, m)
    mp = stumpy.stump(T, m)
    P = mp[:, 0]
    I = mp[:, 1]
    Q = T[-m:]
    QT = core.sliding_dot_product(Q, T)

    return T, P, I, QT, M_T, Σ_T


def stumpi(t, T, m, P, I, QT, M_T, Σ_T):
    """
    Incremental STUMP for streaming data based on the on-line STOMPI and STAMPI
    algorithms.

    Parameters
    ----------
    t : float
        A new single data point following `T`

    T : ndarray
        The time series or sequence for which the matrix profile and matrix profile
        indices will be returned

    m : int
        Window size

    P : ndarray
        The matrix profile for `T`

    I : ndarray
        The matrix profile indices for `T`

    QT : ndarray
        Dot product between some query sequence,`Q`, and time series, `T`

    M_T : ndarray
        Sliding mean for `T`

    Σ_T : ndarray
        Sliding standard deviation for `T`

    Returns
    -------
    T_new : ndarray
        The updated time series

    P_new : ndarray
        The updated matrix profile

    I_new : ndarray
        The updated matrix profile indices

    QT_new : ndarray
        The updated dot product vector

    M_T_new : ndarray
        The updated sliding mean for `T`

    Σ_T_new : ndarray
        The updated sliding standard deviation for `T`

    Notes
    -----
    `DOI: 10.1007/s10618-017-0519-9 \
    <https://www.cs.ucr.edu/~eamonn/MP_journal.pdf>`__

    See Table V

    Note that line 11 is missing an important `sqrt` operation!
    """
    T = np.asarray(T)
    if np.any(np.isinf(T)) or np.any(np.isnan(T)):
        raise ValueError(
            "One or more NaN/inf values were found in the input time series, `T`"
        )

    if np.isinf(t) or np.isnan(t):
        raise ValueError("A NaN/inf value was found in the input data point, `t`")

    n = T.shape[0]
    l = n - m + 1
    T_new = np.append(T, t)
    QT_new = np.empty(QT.shape[0] + 1)
    S = T_new[l:]
    t_drop = T_new[l - 1]

    for j in range(l, 0, -1):
        QT_new[j] = QT[j - 1] - T_new[j - 1] * t_drop + T_new[j + m - 1] * t
    QT_new[0] = 0

    for j in range(m):
        QT_new[0] = QT_new[0] + T_new[j] * S[j]

    μ_Q = M_T[l - 1] + (t - t_drop) / m
    σ_Q = np.sqrt(
        Σ_T[l - 1] * Σ_T[l - 1]
        + M_T[l - 1] * M_T[l - 1]
        + (t * t - t_drop * t_drop) / m
        - μ_Q * μ_Q
    )

    M_T_new = np.append(M_T, μ_Q)
    Σ_T_new = np.append(Σ_T, σ_Q)
    D = core.calculate_distance_profile(m, QT_new, μ_Q, σ_Q, M_T_new, Σ_T_new)

    excl_zone = int(np.ceil(m / 4))
    core.apply_exclusion_zone(D, D.shape[0] - 1, excl_zone)

    for j in range(l):
        if D[j] < P[j]:
            I[j] = l
            P[j] = D[j]

    I_last = np.argmin(D)
    I_new = np.append(I, I_last)
    P_new = np.append(P, D[I_last])

    return T_new, P_new, I_new, QT_new, M_T_new, Σ_T_new
