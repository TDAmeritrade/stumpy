# STUMPY
# Copyright 2019 TD Ameritrade. Released under the terms of the 3-Clause BSD license.
# STUMPY is a trademark of TD Ameritrade IP Company, Inc. All rights reserved.

import numpy as np
from . import core, config
from .aamp import aamp


class aampi:
    """
    Compute an incremental non-normalized (i.e., without z-normalization) matrix profile
    for streaming data

    Parameters
    ----------
    T : numpy.ndarray
        The time series or sequence for which the non-normalized matrix profile and
        matrix profile indices will be returned

    m : int
        Window size

    egress : bool, default True
        If set to `True`, the oldest data point in the time series is removed and
        the time series length remains constant rather than forever increasing

    p : float, default 2.0
        The p-norm to apply for computing the Minkowski distance.

    Attributes
    ----------
    P_ : numpy.ndarray
        The updated matrix profile for `T`

    I_ : numpy.ndarray
        The updated matrix profile indices for `T`

    left_P_ : numpy.ndarray
        The updated left matrix profile for `T`

    left_I_ : numpy.ndarray
        The updated left matrix profile indices for `T`

    T_ : numpy.ndarray
        The updated time series or sequence for which the matrix profile and matrix
        profile indices are computed

    Methods
    -------
    update(t)
        Append a single new data point, `t`, to the time series, `T`, and update the
        matrix profile

    Notes
    -----
    `arXiv:1901.05708 \
    <https://arxiv.org/pdf/1901.05708.pdf>`__

    See Algorithm 1

    Note that we have extended this algorithm for AB-joins as well.
    """

    def __init__(self, T, m, egress=True, p=2.0):
        """
        Initialize the `stumpi` object

        Parameters
        ----------
        T : numpy.ndarray
            The time series or sequence for which the unnormalized matrix profile and
            matrix profile indices will be returned

        m : int
            Window size

        egress : bool, default True
            If set to `True`, the oldest data point in the time series is removed and
            the time series length remains constant rather than forever increasing

        p : float, default 2.0
            The p-norm to apply for computing the Minkowski distance.
        """
        self._T = core._preprocess(T)
        core.check_window_size(m, max_size=self._T.shape[-1])
        self._m = m
        self._n = self._T.shape[0]
        self._excl_zone = int(np.ceil(self._m / config.STUMPY_EXCL_ZONE_DENOM))
        self._egress = egress
        self._p = p

        mp = aamp(self._T, self._m, p=self._p)
        self._P = mp[:, 0].astype(np.float64)
        self._I = mp[:, 1].astype(np.int64)
        self._left_I = mp[:, 2].astype(np.int64)
        self._left_P = np.empty(self._P.shape, dtype=np.float64)
        self._left_P[:] = np.inf

        self._T_isfinite = np.isfinite(self._T)
        self._T, self._T_subseq_isfinite = core.preprocess_non_normalized(
            self._T, self._m
        )

        # Retrieve the left matrix profile values
        for i, j in enumerate(self._left_I):
            if j >= 0:
                self._left_P[i] = np.linalg.norm(
                    self._T[i : i + self._m] - self._T[j : j + self._m], ord=self._p
                )

        Q = self._T[-m:]
        self._p_norm = core.mass_absolute(Q, self._T, p=self._p) ** self._p
        if self._egress:
            self._p_norm_new = np.empty(self._p_norm.shape[0], dtype=np.float64)
            self._n_appended = 0

    def update(self, t):
        """
        Append a single new data point, `t`, to the existing time series `T` and update
        the non-normalized (i.e., without z-normalization) matrix profile and matrix
        profile indices.

        Parameters
        ----------
        t : float
            A single new data point to be appended to `T`

        Notes
        -----
        `arXiv:1901.05708 \
        <https://arxiv.org/pdf/1901.05708.pdf>`__

        See Algorithm 1

        Note that we have extended this algorithm for AB-joins as well.
        """
        if self._egress:
            self._update_egress(t)
        else:
            self._update(t)

    def _update_egress(self, t):
        """
        Ingress a new data point, egress the oldest data point, and update the matrix
        profile and matrix profile indices
        """
        self._n = self._T.shape[0]
        l = self._n - self._m + 1 - 1  # Subtract 1 due to egress
        self._T[:-1] = self._T[1:]
        self._T[-1] = t
        self._n_appended += 1
        self._p_norm[:-1] = self._p_norm[1:]
        S = self._T[l:]
        t_drop = self._T[l - 1]
        self._T_isfinite[:-1] = self._T_isfinite[1:]
        self._T_subseq_isfinite[:-1] = self._T_subseq_isfinite[1:]

        self._I[:-1] = self._I[1:]
        self._P[:-1] = self._P[1:]
        self._left_I[:-1] = self._left_I[1:]
        self._left_P[:-1] = self._left_P[1:]

        if np.isfinite(t):
            self._T_isfinite[-1] = True
        else:
            self._T_isfinite[-1] = False
            t = 0
            self._T[-1] = 0
            S[-1] = 0

        self._T_subseq_isfinite[-1] = np.all(self._T_isfinite[-self._m :])

        self._p_norm_new[1:] = (
            self._p_norm[:l]
            - np.power(abs(self._T[:l] - t_drop), self._p)
            + np.power(abs(self._T[self._m :] - t), self._p)
        )
        self._p_norm_new[0] = (
            np.linalg.norm(self._T[: self._m] - S[: self._m], ord=self._p) ** self._p
        )

        D = np.power(self._p_norm_new, 1.0 / self._p)
        D[~self._T_subseq_isfinite] = np.inf
        if np.any(~self._T_isfinite[-self._m :]):
            D[:] = np.inf

        core.apply_exclusion_zone(D, D.shape[0] - 1, self._excl_zone, np.inf)

        update_idx = np.argwhere(D < self._P).flatten()
        self._I[update_idx] = D.shape[0] + self._n_appended - 1  # D.shape[0] is base-1
        self._P[update_idx] = D[update_idx]

        I_last = np.argmin(D)

        if np.isinf(D[I_last]):
            self._I[-1] = -1
            self._P[-1] = np.inf
        else:
            self._I[-1] = I_last + self._n_appended
            self._P[-1] = D[I_last]

        self._left_I[-1] = I_last + self._n_appended
        self._left_P[-1] = D[I_last]

        self._p_norm[:] = self._p_norm_new

    def _update(self, t):
        """
        Ingress a new data point and update the matrix profile and matrix profile
        indices without egressing the oldest data point
        """
        self._n = self._T.shape[0]
        l = self._n - self._m + 1
        T_new = np.append(self._T, t)
        p_norm_new = np.empty(self._p_norm.shape[0] + 1, dtype=np.float64)
        S = T_new[l:]
        t_drop = T_new[l - 1]

        if np.isfinite(t):
            self._T_isfinite = np.append(self._T_isfinite, True)
        else:
            self._T_isfinite = np.append(self._T_isfinite, False)
            t = 0
            T_new[-1] = 0
            S[-1] = 0

        self._T_subseq_isfinite = np.append(
            self._T_subseq_isfinite, np.all(self._T_isfinite[-self._m :])
        )

        p_norm_new[1:] = (
            self._p_norm[:l]
            - np.power(abs(T_new[:l] - t_drop), self._p)
            + np.power(abs(T_new[self._m :] - t), self._p)
        )
        p_norm_new[0] = (
            np.linalg.norm(T_new[: self._m] - S[: self._m], ord=self._p) ** self._p
        )

        D = np.power(p_norm_new, 1.0 / self._p)
        D[~self._T_subseq_isfinite] = np.inf
        if np.any(~self._T_isfinite[-self._m :]):
            D[:] = np.inf

        core.apply_exclusion_zone(D, D.shape[0] - 1, self._excl_zone, np.inf)

        update_idx = np.argwhere(D[:l] < self._P[:l]).flatten()
        self._I[update_idx] = l
        self._P[update_idx] = D[update_idx]

        I_last = np.argmin(D)
        if np.isinf(D[I_last]):
            I_new = np.append(self._I, -1)
            P_new = np.append(self._P, np.inf)
        else:
            I_new = np.append(self._I, I_last)
            P_new = np.append(self._P, D[I_last])
        left_I_new = np.append(self._left_I, I_last)
        left_P_new = np.append(self._left_P, D[I_last])

        self._T = T_new
        self._P = P_new
        self._I = I_new
        self._left_I = left_I_new
        self._left_P = left_P_new
        self._p_norm = p_norm_new

    @property
    def P_(self):
        """
        Get the matrix profile
        """
        return self._P.astype(np.float64)

    @property
    def I_(self):
        """
        Get the matrix profile indices
        """
        return self._I.astype(np.int64)

    @property
    def left_P_(self):
        """
        Get the left matrix profile
        """
        return self._left_P.astype(np.float64)

    @property
    def left_I_(self):
        """
        Get the left matrix profile indices
        """
        return self._left_I.astype(np.int64)

    @property
    def T_(self):
        """
        Get the time series
        """
        return self._T
