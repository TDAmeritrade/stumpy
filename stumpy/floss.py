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

    bidirectional : bool, default True
        Flag for computing a bidirectional (`True`) or 1-dimensional (`False`)
        idealized arc curve

    n_iter : int, default 1000
        Number of iterations to average over when determining the parameters for
        beta distribution

    n_samples : int, default 1000
        Number of distribution samples to draw during each iteration

    seed : int, default 0
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

    bidirectional : bool, default True
        Flag for normalizing the arc curve with a bidirectional (`True`) or
        1-dimensional (`False`) idealized arc curve. If a `custom_iac` is
        specified then this flag is ignored.

    excl_factor : int, default 5
        The multiplying factor for the first and last regime exclusion zones

    custom_iac : ndarray, default None
        A custom idealized arc curve (IAC) that will used for correcting the
        arc curve

    seed : int, default 0
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

    excl_factor : int, default 5
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
    for static data (i.e., batch processing)

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

    excl_factor : int, default 5
        The multiplying factor for the regime exclusion zone

    custom_iac : ndarray, default None
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


class floss(object):
    """
    Compute the Fast Low-cost Online Semantic Segmentation (FLOSS) for
    streaming data

    Parameters
    ----------
    mp : ndarray
        The first column consists of the matrix profile, the second column
        consists of the matrix profile indices, the third column consists of
        the left matrix profile indices, and the fourth column consists of
        the right matrix profile indices.

    T : ndarray
        A 1-D time series data used to generate the matrix profile and matrix profile
        indices found in `mp`. Note that the the right matrix profile index is used and
        the right matrix profile is intelligently recomputed on the fly from `T` instead
        of using the bidirectional matrix profile.

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

    excl_factor : int, default 5
        The multiplying factor for the regime exclusion zone. Note that this
        is unrelated to the `excl_zone` used in to compute the matrix profile.

    n_iter : int, default 1000
        Number of iterations to average over when determining the parameters for
        the IAC beta distribution

    n_samples : int, default 1000
        Number of distribution samples to draw during each iteration when
        computing the IAC

    custom_iac : ndarray, default None
        A custom idealized arc curve (IAC) that will used for correcting the
        arc curve

    Attributes
    ----------
    cac_1d_ : ndarray
        A 1-dimensional corrected arc curve (CAC) updated as a result of ingressing a
        single new data point and egressing a single old data point.

    P_ : ndarray
        The matrix profile updated as a result of ingressing a single new data
        point and egressing a single old data point.

    I_ : ndarray
        The (right) matrix profile indices updated as a result of ingressing a single
        new data point and egressing a single old data point.

    T_ : ndarray
        The updated time series, `T`

    Methods
    -------
    update(t)
        Ingress a new data point, `t`, onto the time series, `T`, followed by egressing
        the oldest single data point from `T`. Then, update the 1-dimensional corrected
        arc curve (CAC_1D) and the matrix profile.

    Notes
    -----
    DOI: 10.1109/ICDM.2017.21 <https://www.cs.ucr.edu/~eamonn/Segmentation_ICDM.pdf>`__

    See Section C

    This is the implementation for Fast Low-cost Online Semantic
    Segmentation (FLOSS).
    """

    def __init__(
        self, mp, T, m, L, excl_factor=5, n_iter=1000, n_samples=1000, custom_iac=None
    ):
        """
        Initialize the FLOSS object

        Parameters
        ----------
        mp : ndarray
            The first column consists of the matrix profile, the second column
            consists of the matrix profile indices, the third column consists of
            the left matrix profile indices, and the fourth column consists of
            the right matrix profile indices.

        T : ndarray
            A 1-D time series data used to generate the matrix profile and matrix
            profile indices found in `mp`. Note that the the right matrix profile index
            is used and the right matrix profile is intelligently recomputed on-the-fly
            from `T` instead of using the bidirectional matrix profile.

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

        excl_factor : int, default 5
            The multiplying factor for the regime exclusion zone. Note that this
            is unrelated to the `excl_zone` used in to compute the matrix profile.

        n_iter : int, default 1000
            Number of iterations to average over when determining the parameters for
            the IAC beta distribution

        n_samples : int, default 1000
            Number of distribution samples to draw during each iteration when
            computing the IAC

        custom_iac : ndarray, default None
            A custom idealized arc curve (IAC) that will used for correcting the
            arc curve
        """
        self._mp = copy.deepcopy(np.asarray(mp))
        self._T = copy.deepcopy(np.asarray(T))
        self._m = m
        self._L = L
        self._excl_factor = excl_factor
        self._n_iter = n_iter
        self._n_samples = n_samples
        self._custom_iac = custom_iac
        self._k = self._mp.shape[0]
        self._n = self._T.shape[0]
        self._last_idx = self._n - self._m + 1  # Depends on the changing length of `T`
        self._n_appended = 0
        self._T_isfinite = np.isfinite(self._T)
        self._finite_T = self._T.copy()
        self._finite_T[~np.isfinite(self._finite_T)] = 0.0
        self._finite_Q = self._finite_T[-self._m :].copy()

        if self._custom_iac is None:  # pragma: no cover
            self._custom_iac = _iac(
                self._k,
                bidirectional=False,
                n_iter=self._n_iter,
                n_samples=self._n_samples,
            )

        right_nn = np.zeros((self._k, self._m))

        # Disable the bidirectional matrix profile indices and left indices
        self._mp[:, 1] = -1
        self._mp[:, 2] = -1

        # Update matrix profile distance to be right mp distance and not bidirectional.
        # Use right indices to perform direct distance calculations
        # Note that any -1 indices must have a np.inf matrix profile value
        right_indices = [np.arange(IR, IR + self._m) for IR in self._mp[:, 3].tolist()]
        right_nn[:] = self._T[np.array(right_indices)]
        self._mp[:, 0] = np.linalg.norm(
            core.z_norm(core.rolling_window(self._T, self._m), 1)
            - core.z_norm(right_nn, 1),
            axis=1,
        )
        inf_indices = np.argwhere(self._mp[:, 3] < 0).flatten()
        self._mp[inf_indices, 0] = np.inf
        self._mp[inf_indices, 3] = inf_indices

        self._cac = np.ones(self._k) * -1

    def update(self, t):
        """
        Ingress a new data point, `t`, onto the time series, `T`, followed by egressing
        the oldest single data point from `T`. Then, update the 1-dimensional corrected
        arc curve (CAC_1D) and the matrix profile.

        Parameters
        ----------
        t : float
            A single new data point to be appended to `T`

        Notes
        -----
        DOI: 10.1109/ICDM.2017.21 \
        <https://www.cs.ucr.edu/~eamonn/Segmentation_ICDM.pdf>`__

        See Section C

        This is the implementation for Fast Low-cost Online Semantic
        Segmentation (FLOSS).
        """
        self._T[:-1] = self._T[1:]
        self._T_isfinite[:-1] = self._T_isfinite[1:]
        self._finite_T[:-1] = self._finite_T[1:]
        self._finite_Q[:-1] = self._finite_Q[1:]
        self._T[-1] = t
        self._T_isfinite[-1] = np.isfinite(t)
        self._finite_T[-1] = t
        if not np.isfinite(t):
            self._finite_T[-1] = 0.0
        self._finite_Q[-1] = self._finite_T[-1]
        excl_zone = int(np.ceil(self._m / 4))
        # Note that the start of the exclusion zone is relative to
        # the unchanging length of the matrix profile index
        zone_start = max(0, self._k - excl_zone)

        # Egress
        # Remove the first element in the matrix profile index
        # Shift mp up by one and replace the last row with new values
        self._mp[:-1, :] = self._mp[1:, :]
        self._mp[-1, 0] = np.inf
        self._mp[-1, 3] = self._last_idx

        # Ingress
        M_T, Σ_T = core.compute_mean_std(self._T, self._m)

        D = core.mass(self._finite_Q, self._finite_T, M_T, Σ_T)
        D[zone_start:] = np.inf

        T_subseq_isfinite = core.rolling_isfinite(self._T_isfinite, self._m)

        D[~T_subseq_isfinite] = np.inf
        if not T_subseq_isfinite[-1]:
            D[:] = np.inf

        # Update nearest neighbor for old data if any old subsequences
        # are closer to the newly arrived subsequence
        update_idx = np.argwhere(D < self._mp[:, 0]).flatten()
        self._mp[update_idx, 0] = D[update_idx]
        self._mp[update_idx, 3] = self._last_idx

        self._cac[:] = _cac(
            self._mp[:, 3] - self._n_appended - 1,
            self._L,
            bidirectional=False,
            excl_factor=self._excl_factor,
            custom_iac=self._custom_iac,
        )

        self._last_idx += 1
        self._n_appended += 1

    @property
    def cac_1d_(self):
        """
        Get the updated 1-dimensional corrected arc curve (CAC_1D)
        """
        return self._cac.astype(np.float)

    @property
    def P_(self):
        """
        Get the updated matrix profile
        """
        return self._mp[:, 0].astype(np.float)

    @property
    def I_(self):
        """
        Get the updated (right) matrix profile indices
        """
        return self._mp[:, 3].astype(np.int)

    @property
    def T_(self):
        """
        Get the updated time series, `T`
        """
        return self._T.astype(np.float)
