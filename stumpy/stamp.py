# STUMPY
# Copyright 2019 TD Ameritrade. Released under the terms of the 3-Clause BSD license.
# STUMPY is a trademark of TD Ameritrade IP Company, Inc. All rights reserved.

import numpy as np
from . import core


def mass(Q, T, M_T, Σ_T, trivial_idx=None, excl_zone=0, left=False, right=False):
    """
    Compute "Mueen's Algorithm for Similarity Search" (MASS)

    Parameters
    ----------
    Q : ndarray
        Query array or subsequence

    T : ndarray
        Time series array or sequence

    M_T : ndarray
        Sliding mean for `T`

    Σ_T : ndarray
        Sliding standard deviation for `T`

    trivial_idx : int
        Index for the start of the trivial self-join

    excl_zone : int
        The half width for the exclusion zone relative to the `trivial_idx`.
        If the `trivial_idx` is `None` then this parameter is ignored.

    left : bool
        Return the left matrix profile indices if `True`. If `right` is True
        then this parameter is ignored.

    right : bool
        Return the right matrix profiles indices if `True`

    Returns
    -------
    P : ndarray
        Matrix profile

    I : ndarray
        Matrix profile indices
    """
    D = core.mass(Q, T, M_T, Σ_T)
    if trivial_idx is not None:
        zone_start = max(0, trivial_idx - excl_zone)
        zone_stop = min(T.shape[0] - Q.shape[0] + 1, trivial_idx + excl_zone)
        D[zone_start:zone_stop] = np.inf

        # Get left and right matrix profiles
        IL = -1
        PL = np.inf
        if D[:trivial_idx].size:
            IL = np.argmin(D[:trivial_idx])
            PL = D[IL]
        if zone_start <= IL < zone_stop:
            IL = -1

        IR = -1
        PR = np.inf
        if D[trivial_idx:].size:
            IR = trivial_idx + np.argmin(D[trivial_idx:])
            PR = D[IR]
        if zone_start <= IR < zone_stop:
            IR = -1

    # Element-wise Min
    I = np.argmin(D)
    P = D[I]

    if trivial_idx is not None and left:
        I = IL
        P = PL

    if trivial_idx is not None and right:
        I = IR
        P = PR

    return P, I


def stamp(T_A, T_B, m, ignore_trivial=False):
    """
    Compute matrix profile and indices using the "Scalable Time series
    Anytime Matrix Profile" (STAMP) algorithm and MASS (2017 - with FFT).

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
        Two column numpy array where the first column is the matrix profile
        and the second column is the matrix profile indices

    Notes
    -----
    `DOI: 10.1109/ICDM.2016.0179 \
    <https://www.cs.ucr.edu/~eamonn/PID4481997_extend_Matrix%20Profile_I.pdf>`__

    See Table III

    Timeseries, T_B, will be annotated with the distance location
    (or index) of all its subsequences in another times series, T_A.

    For every subsequence, Q, in T_B, you will get a distance and index for
    the closest subsequence in T_A. Thus, the array returned will have length
    T_B.shape[0]-m+1
    """

    core.check_dtype(T_A)
    core.check_nan(T_A)
    core.check_dtype(T_B)
    core.check_nan(T_B)
    core.check_window_size(m)
    subseq_T_B = core.rolling_window(T_B, m)
    excl_zone = int(np.ceil(m / 2))
    M_T, Σ_T = core.compute_mean_std(T_A, m)

    # Add exclusionary zone
    if ignore_trivial:
        out = [
            mass(subseq, T_A, M_T, Σ_T, i, excl_zone)
            for i, subseq in enumerate(subseq_T_B)
        ]
    else:
        out = [mass(subseq, T_A, M_T, Σ_T) for subseq in subseq_T_B]
    out = np.array(out, dtype=object)

    return out
