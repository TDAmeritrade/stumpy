# STUMPY
# Copyright 2019 TD Ameritrade. Released under the terms of the 3-Clause BSD license.
# STUMPY is a trademark of TD Ameritrade IP Company, Inc. All rights reserved.

import numpy as np

from . import core


def _mass_PI(Q, T, M_T, Σ_T, trivial_idx=None, excl_zone=0, left=False, right=False):
    """
    Compute "Mueen's Algorithm for Similarity Search" (MASS)

    Parameters
    ----------
    Q : numpy.ndarray
        Query array or subsequence

    T : numpy.ndarray
        Time series array or sequence

    M_T : numpy.ndarray
        Sliding mean for `T`

    Σ_T : numpy.ndarray
        Sliding standard deviation for `T`

    trivial_idx : int, default None
        Index for the start of the trivial self-join

    excl_zone : int, default 0
        The half width for the exclusion zone relative to the `trivial_idx`.
        If the `trivial_idx` is `None` then this parameter is ignored.

    left : bool, default False
        Return the left matrix profile indices if `True`. If `right` is True
        then this parameter is ignored.

    right : bool, default False
        Return the right matrix profiles indices if `True`

    Returns
    -------
    P : numpy.ndarray
        Matrix profile

    I : numpy.ndarray
        Matrix profile indices
    """
    D = core.mass(Q, T, M_T, Σ_T)

    if trivial_idx is not None:
        core.apply_exclusion_zone(D, trivial_idx, excl_zone, np.inf)

        # Get left and right matrix profiles
        IL = -1
        PL = np.inf
        if D[:trivial_idx].size:
            IL = np.argmin(D[:trivial_idx])
            PL = D[IL]
        if PL == np.inf:
            IL = -1

        IR = -1
        PR = np.inf
        if D[trivial_idx + 1 :].size:
            IR = trivial_idx + 1 + np.argmin(D[trivial_idx + 1 :])
            PR = D[IR]
        if PR == np.inf:
            IR = -1

    # Element-wise Min
    I = np.argmin(D)
    P = D[I]
    if P == np.inf:
        I = -1

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
    T_A : numpy.ndarray
        The time series or sequence for which the matrix profile will be returned

    T_B : numpy.ndarray
        The time series or sequence that will be used to annotate T_A. For every
        subsequence in T_A, its nearest neighbor in T_B will be recorded.

    m : int
        Window size

    ignore_trivial : bool, default False
        `True` if this is a self join and `False` otherwise (i.e., AB-join).

    Returns
    -------
    out : numpy.ndarray
        Two column numpy array where the first column is the matrix profile
        and the second column is the matrix profile indices

    Notes
    -----
    `DOI: 10.1109/ICDM.2016.0179 \
    <https://www.cs.ucr.edu/~eamonn/PID4481997_extend_Matrix%20Profile_I.pdf>`__

    See Table III

    Timeseries, T_A, will be annotated with the distance location
    (or index) of all its subsequences in another times series, T_B.

    For every subsequence, Q, in T_A, you will get a distance and index for
    the closest subsequence in T_B. Thus, the array returned will have length
    T_A.shape[0]-m+1
    """
    T_B, M_T, Σ_T = core.preprocess(T_B, m)
    T_A = T_A.copy()
    T_A[np.isinf(T_A)] = np.nan

    if T_A.ndim != 1:  # pragma: no cover
        raise ValueError(f"T_A is {T_A.ndim}-dimensional and must be 1-dimensional. ")

    if T_B.ndim != 1:  # pragma: no cover
        raise ValueError(f"T_B is {T_B.ndim}-dimensional and must be 1-dimensional. ")

    core.check_window_size(m, max_size=min(T_A.shape[0], T_B.shape[0]))
    subseq_T_A = core.rolling_window(T_A, m)
    excl_zone = int(np.ceil(m / 2))

    # Add exclusionary zone
    if ignore_trivial:
        out = [
            _mass_PI(subseq, T_B, M_T, Σ_T, i, excl_zone)
            for i, subseq in enumerate(subseq_T_A)
        ]
    else:
        out = [_mass_PI(subseq, T_B, M_T, Σ_T) for subseq in subseq_T_A]
    out = np.array(out, dtype=object)

    return out
