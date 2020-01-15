# STUMPY
# Copyright 2019 TD Ameritrade. Released under the terms of the 3-Clause BSD license.  # noqa: E501
# STUMPY is a trademark of TD Ameritrade IP Company, Inc. All rights reserved.

import copy

import numpy as np
import scipy.stats

from . import core


def _nnmark(I):
    """
    Count the number of nearest neighbor overhead crossings or arcs.

    Parameters
    ----------
    I : ndarray
        Matrix profile indices

    Returns
    -------
    nnmark : ndarray
        Counts of nearest neighbor overheard crossings or arcs.

    Notes
    -----
    DOI: 10.1109/ICDM.2017.21 <https://www.cs.ucr.edu/~eamonn/Segmentation_ICDM.pdf>`__

    See Table I

    This is a fast and vectorized implementation of the nnmark algorithm.
    """
    I = I.astype(int)

    # Replace index values that are less than zero with its own positional index
    idx = np.argwhere(I < 0).flatten()
    I[idx] = idx

    k = I.shape[0]
    i = np.arange(k)

    nnmark = np.bincount(np.minimum(i, I), minlength=k)
    nnmark -= np.bincount(np.maximum(i, I), minlength=k)

    return nnmark.cumsum()


def _iac(
    width, bidirectional=True, n_iter=1000, n_samples=1000, seed=0
):  # pragma: no cover
    """
    Compute the bidirectional idealized arc curve (IAC). This is based
    on a beta distribution that is scaled with a width that is identical
    to the length of the matrix profile index. The height of the idealized
    parabolic curve is assumed to be exactly half the width.

    If `bidirectional=False` then the 1-dimensional IAC is computed instead.

    Parameters
    ----------
    width : int
        The width of the bidirectional idealized arc curve. This is equal
        to the length of the matrix profile index.

    bidirectional : bool
        Flag for computing a bidirectional (`True`) or 1-dimensional (`False`)
        idealized arc curve

    n_iter : int
        Number of iterations to average over when determining the parameters for
        beta distribution

    n_samples : int
        Number of distribution samples to draw during each iteration

    seed : int
        NumPy random seed used in sampling the beta distribution. Set this to your
        desired value for reproducibility purposes. The default value is set to `0`.

    Returns
    -------
    IAC : ndarray
        Idealized arc curve (IAC)
    """

    np.random.seed(seed)

    I = np.random.randint(0, width, size=width)
    if bidirectional is False:  # Idealized 1-dimensional matrix profile index
        I[:-1] = width
        for i in range(width - 1):
            I[i] = np.random.randint(i + 1, width)

    target_AC = _nnmark(I)

    params = np.empty((n_iter, 2))
    for i in range(n_iter):
        hist_dist = scipy.stats.rv_histogram(
            (target_AC, np.append(np.arange(width), width))
        )
        data = hist_dist.rvs(size=n_samples)
        a, b, c, d = scipy.stats.beta.fit(data, floc=0, fscale=width)

        params[i, 0] = a
        params[i, 1] = b

    a_mean = np.round(np.mean(params[:, 0]), 2)
    b_mean = np.round(np.mean(params[:, 1]), 2)

    IAC = scipy.stats.beta.pdf(np.arange(width), a_mean, b_mean, loc=0, scale=width)
    slope, _, _, _ = np.linalg.lstsq(IAC.reshape(-1, 1), target_AC, rcond=None)

    IAC *= slope

    return IAC


def _cac(I, L, bidirectional=True, excl_factor=5, custom_iac=None, seed=0):
    """
    Compute the corrected arc curve (CAC)

    Parameters
    ----------
    I : ndarray
        The matrix profile indices for the time series of interest

    L : int
        The subsequence length that is set roughly to be one period length.
        This is likely to be the same value as the window size, `m`, used
        to compute the matrix profile and matrix profile index but it can
        be different since this is only used to manage edge effects
        and has no bearing on any of the IAC or CAC core calculations.

    bidirectional : bool
        Flag for normalizing the arc curve with a bidirectional (`True`) or
        1-dimensional (`False`) idealized arc curve. If a `custom_iac` is
        specified then this flag is ignored.

    excl_factor : int
        The multiplying factor for the first and last regime exclusion zones

    custom_iac : np.array
        A custom idealized arc curve (IAC) that will used for correcting the
        arc curve

    seed : int
        NumPy random seed used in sampling the `iac` beta distribution. Set this
        to your desired value for reproducibility purposes. The default value is
        set to `0`.

    Returns
    -------
    output : ndarray
        A corrected arc curve (CAC)

    Notes
    -----
    DOI: 10.1109/ICDM.2017.21 <https://www.cs.ucr.edu/~eamonn/Segmentation_ICDM.pdf>`__

    See Table I

    This is the implementation for the corrected arc curve (CAC).
    """
    k = I.shape[0]
    AC = _nnmark(I)
    CAC = np.zeros(k)

    if custom_iac is None:
        IAC = _iac(k, bidirectional, seed=seed)
    else:
        IAC = custom_iac
    IAC[IAC == 0.0] = 10 ** -10  # Avoid divide by zero
    CAC[:] = AC / IAC
    CAC[CAC > 1.0] = 1.0  # Equivalent to min

    if excl_factor > 0:
        CAC[: L * excl_factor] = 1.0
        CAC[-L * excl_factor :] = 1.0

    return CAC


def _rea(cac, n_regimes, L, excl_factor=5):
    """
    Find the location of the regimes using the regime extracting
    algorithm (REA)

    Parameters
    ----------
    cac : ndarray
        The corrected arc curve

    n_regimes : int
        The number of regimes to search for. This is one more than the
        number of regime changes as denoted in the original paper.

    L : int
        The subsequence length that is set roughly to be one period length.
        This is likely to be the same value as the window size, `m`, used
        to compute the matrix profile and matrix profile index but it can
        be different since this is only used to manage edge effects
        and has no bearing on any of the IAC or CAC core calculations.

    excl_factor : int
        The multiplying factor for the regime exclusion zone

    Returns
    -------
    regime_locs : ndarray
        The locations of the regimes

    Notes
    -----
    DOI: 10.1109/ICDM.2017.21 <https://www.cs.ucr.edu/~eamonn/Segmentation_ICDM.pdf>`__

    See Table II

    This is the implementation for the regime extracting algorithm (REA).
    """
    regime_locs = np.empty(n_regimes - 1, dtype=np.int)
    tmp_cac = copy.deepcopy(cac)
    for i in range(n_regimes - 1):
        regime_locs[i] = np.argmin(tmp_cac)
        excl_start = max(regime_locs[i] - excl_factor * L, 0)
        excl_stop = min(regime_locs[i] + excl_factor * L, cac.shape[0])
        tmp_cac[excl_start:excl_stop] = 1.0

    return regime_locs


def fluss(I, L, n_regimes, excl_factor=5, custom_iac=None):
    """
    Compute the Fast Low-cost Unipotent Semantic Segmentation (FLUSS)
    for static data.

    Essentially, this is a wrapper to compute the corrected arc curve and
    regime locations.

    Parameters
    ----------
    I : ndarray
        The matrix profile indices for the time series of interest

    L : int
        The subsequence length that is set roughly to be one period length.
        This is likely to be the same value as the window size, `m`, used
        to compute the matrix profile and matrix profile index but it can
        be different since this is only used to manage edge effects
        and has no bearing on any of the IAC or CAC core calculations.

    n_regimes : int
        The number of regimes to search for. This is one more than the
        number of regime changes as denoted in the original paper.

    m : int
        The subsequence length. This is expected to be the same value as the
        window size used to compute the matrix profile and matrix
        profile index.

    excl_factor : int
        The multiplying factor for the regime exclusion zone

    custom_iac : np.array
        A custom idealized arc curve (IAC) that will used for correcting the
        arc curve

    Returns
    -------
    cac : ndarray
        A corrected arc curve (CAC)

    regime_locs : ndarray
        The locations of the regimes

    Notes
    -----
    `DOI: 10.1109/ICDM.2017.21 <https://www.cs.ucr.edu/~eamonn/Segmentation_ICDM.pdf>`__

    See Section A

    This is the implementation for Fast Low-cost Unipotent Semantic
    Segmentation (FLUSS).
    """
    cac = _cac(I, L, bidirectional=True, excl_factor=excl_factor, custom_iac=custom_iac)
    regime_locs = _rea(cac, n_regimes, L, excl_factor=excl_factor)

    return cac, regime_locs


def floss(
    mp,
    old_data,
    add_data,
    m,
    L,
    excl_factor=5,
    n_iter=1000,
    n_samples=1000,
    skip=0,
    custom_iac=None,
):
    """
    Compute the Fast Low-cost Online Semantic Segmentation (FLOSS) for
    streaming data.

    Parameters
    ----------
    mp : ndarray
        The first column consists of the matrix profile, the second column
        consists of the matrix profile indices, the third column consists of
        the left matrix profile indices, and the fourth column consists of
        the right matrix profile indices.

    old_data : ndarray
        A 1-D array of old time series data used to generate the matrix
        profile and matrix profile indices found in `mp`. Note that the
        the right matrix profile index is used and the right matrix profile
        is intelligently recomputed on the fly from `old_data` instead of
        using the bidirectional matrix profile.

    add_data : ndarray
        A 1-D array of additional time series data that is likely arriving
        from a stream.

    m : int
        The window size for computing sliding window mass. This is identical
        to the window size used in the matrix profile calculation. For managing
        edge effects, see the `L` parameter.

    L : int
        The subsequence length that is set roughly to be one period length.
        This is likely to be the same value as the window size, `m`, used
        to compute the matrix profile and matrix profile index but it can
        be different since this is only used to manage edge effects
        and has no bearing on any of the IAC or CAC core calculations.

    excl_factor : int
        The multiplying factor for the regime exclusion zone. Note that this
        is unrelated to the `excl_zone` used in to compute the matrix profile.

    n_iter : int
        Number of iterations to average over when determining the parameters for
        the IAC beta distribution

    n_samples : int
        Number of distribution samples to draw during each iteration when
        computing the IAC

    skip : int
        Number of windows to skip before yielding the first CAC. For example, setting
        `skip=49` will skip the first 49 windows, yield the 50th window, skip the
        next 49 windows, yield the 100th window, etc. This is useful for displaying
        FLOSS in an animation.

    custom_iac : np.array
        A custom idealized arc curve (IAC) that will used for correcting the
        arc curve

    Returns
    -------
    cac_out : ndarray
        A corrected arc curve (CAC) updated as a result of ingressing a single
        new data point and egressing a single old data point.

    mp_out : ndarray
        The first column consists of the matrix profile and the second column
        consists of the matrix profile indices. This updated matrix profile and
        matrix profile indices are a result of ingressing a single new data
        point and egressing a single old data point. Note that this array does
        not contain the left and right matrix profile indices.

    idx_out : int
        The total length of all of the aggregated old data and the additional data.

    T : ndarray
        The last `n` data points from the sliding window after concatenating
        `old_data` and `new_data` as a result of ingressing a single new
        data point and egressing a single old data point

    Notes
    -----
    DOI: 10.1109/ICDM.2017.21 <https://www.cs.ucr.edu/~eamonn/Segmentation_ICDM.pdf>`__

    See Section C

    This is the implementation for Fast Low-cost Online Semantic
    Segmentation (FLOSS).
    """

    n = old_data.shape[0]
    k = mp.shape[0]
    last_idx = n - m + 1  # This is dependent on the changing length of `old_data`

    if custom_iac is None:  # pragma: no cover
        custom_iac = _iac(k, bidirectional=False, n_iter=n_iter, n_samples=n_samples)
    cac_out = np.ones(k)
    mp_out = copy.deepcopy(mp)
    right_nn = np.zeros((k, m))

    old_data = np.asarray(old_data)
    add_data = np.asarray(add_data)

    concat_data = np.concatenate((old_data, add_data))

    # Disable the bidirectional matrix profile indices and left indices
    mp_out[:, 1] = -1
    mp_out[:, 2] = -1

    # Update matrix profile distance to be right mp distance and not bidirectional.
    # Use right indices to perform direct distance calculations
    # Note that any -1 indices must have a np.inf matrix profile value
    right_indices = [np.arange(IR, IR + m) for IR in mp_out[:, 3].tolist()]
    right_nn[:] = old_data[np.array(right_indices)]
    mp_out[:, 0] = np.linalg.norm(
        core.z_norm(core.rolling_window(old_data, m), 1) - core.z_norm(right_nn, 1),
        axis=1,
    )
    inf_indices = np.argwhere(mp_out[:, 3] < 0).flatten()
    mp_out[inf_indices, 0] = np.inf
    mp_out[inf_indices, 3] = inf_indices

    rolling_Qs = core.rolling_window(concat_data[k:], m)
    rolling_Ts = core.rolling_window(concat_data[1:], n)
    M_T = np.zeros(n - m + 1)
    Σ_T = np.zeros(n - m + 1)
    D = np.zeros(n - m + 1)
    excl_zone = int(np.ceil(m / 4))
    # Note that the start of the exclusion zone is relative to
    # the unchanging length of the matrix profile index
    zone_start = max(0, k - excl_zone)

    for i, (Q, T) in enumerate(zip(rolling_Qs, rolling_Ts)):
        # Egress
        # Remove the first element in the matrix profile index
        # Shift mp up by one and replace the last row with new values
        mp_out[:] = np.roll(mp_out, -1, axis=0)
        mp_out[-1, 0] = np.inf
        mp_out[-1, 3] = last_idx + i

        # Ingress
        M_T[:], Σ_T[:] = core.compute_mean_std(T, m)

        D[:] = core.mass(Q, T, M_T, Σ_T)
        D[zone_start:] = np.inf

        # Update nearest neighbor for old data if any old subsequences
        # are closer to the newly arrived subsequence
        update_idx = np.argwhere(D < mp_out[:, 0]).flatten()
        mp_out[update_idx, 0] = D[update_idx]
        mp_out[update_idx, 3] = last_idx + i

        if i % (skip + 1) == 0:
            cac_out[:] = _cac(
                mp_out[:, 3] - i - 1,
                L,
                bidirectional=False,
                excl_factor=excl_factor,
                custom_iac=custom_iac,
            )

            yield cac_out, mp_out, T
