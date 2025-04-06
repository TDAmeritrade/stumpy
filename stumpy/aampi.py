# STUMPY
# Copyright 2019 TD Ameritrade. Released under the terms of the 3-Clause BSD license.
# STUMPY is a trademark of TD Ameritrade IP Company, Inc. All rights reserved.

import numpy as np

from . import config, core
from .aamp import aamp


class aampi:
    # needs to be enhanced to support top-k matrix profile
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

    k : int, default 1
        The number of top `k` smallest distances used to construct the matrix profile.
        Note that this will increase the total computational time and memory usage
        when k > 1.

    mp : numpy.ndarray, default None
        A pre-computed matrix profile (and corresponding matrix profile indices).
        This is a 2D array of shape `(len(T) - m + 1, 2 * k + 2)`, where the first `k`
        columns are top-k matrix profile, and the next `k` columns are their
        corresponding indices. The last two columns correspond to the top-1 left and
        top-1 right matrix profile indices. When None (default), this array is computed
        internally using `stumpy.aamp`.

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

    def __init__(self, T, m, egress=True, p=2.0, k=1, mp=None):
        """
        Initialize the `aampi` object

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

        k : int, default 1
            The number of top `k` smallest distances used to construct the matrix
            profile. Note that this will increase the total computational time and
            memory usage when k > 1.

        mp : numpy.ndarray, default None
            A pre-computed matrix profile (and corresponding matrix profile indices).
            This is a 2D array of shape `(len(T) - m + 1, 2 * k + 2)`, where the first
            `k` columns are top-k matrix profile, and the next `k` columns are their
            corresponding indices. The last two columns correspond to the top-1 left
            and top-1 right matrix profile indices. When None (default), this array is
            computed internally using `stumpy.aamp`.
        """
        self._T = core._preprocess(T)
        core.check_window_size(m, max_size=self._T.shape[0])
        self._m = m
        self._n = self._T.shape[0]
        self._excl_zone = int(np.ceil(self._m / config.STUMPY_EXCL_ZONE_DENOM))
        self._egress = egress
        self._p = p
        self._k = k

        if mp is None:
            mp = aamp(self._T, self._m, p=self._p, k=self._k)
        else:
            mp = mp.copy()

        if mp.shape != (
            len(self._T) - self._m + 1,
            2 * self._k + 2,
        ):  # pragma: no cover
            msg = (
                f"The shape of `mp` must match ({len(T) - m + 1}, {2 * k + 2}) but "
                + f"found {mp.shape} instead."
            )
            raise ValueError(msg)

        self._P = mp[:, : self._k].astype(np.float64)
        self._I = mp[:, self._k : 2 * self._k].astype(np.int64)
        self._left_I = mp[:, 2 * self._k].astype(np.int64)
        self._left_P = np.full_like(self._left_I, np.inf, dtype=np.float64)
        self._left_P[:] = np.inf

        self._T_isfinite = np.isfinite(self._T)
        self._T, self._T_subseq_isfinite = core.preprocess_non_normalized(
            self._T, self._m
        )

        # Retrieve the left matrix profile values

        # Since each matrix profile value is the minimum between the left and right
        # matrix profile values, we can save time by re-computing only the left matrix
        # profile value when the matrix profile index is equal to the right matrix
        # profile index.
        mask = self._left_I == self._I[:, 0]
        self._left_P[mask] = self._P[mask, 0]

        # Only re-compute the `i`-th left matrix profile value, `self._left_P[i]`,
        # when `self._I[i] != self._left_I[i]`
        for i in np.flatnonzero(self._left_I >= 0 & ~mask):
            j = self._left_I[i]
            if j >= 0:
                self._left_P[i] = np.linalg.norm(
                    self._T[i : i + self._m] - self._T[j : j + self._m], ord=self._p
                )

        Q = self._T[-self._m :]
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

        Parameters
        ----------
        t : float
            A single new data point to be appended to `T`
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

        mask = self._p_norm_new < config.STUMPY_P_NORM_THRESHOLD
        self._p_norm_new[mask] = 0

        D = np.power(self._p_norm_new, 1.0 / self._p)
        D[~self._T_subseq_isfinite] = np.inf
        if np.any(~self._T_isfinite[-self._m :]):
            D[:] = np.inf

        core._update_incremental_PI(
            D, self._P, self._I, self._excl_zone, n_appended=self._n_appended
        )

        # All neighbors of the last subsequence are on its left. So, its (top-1)
        # matrix profile value/index and its left matrix profile value/index must
        # be equal.
        self._left_P[-1] = self._P[-1, 0]
        self._left_I[-1] = self._I[-1, 0]

        self._p_norm[:] = self._p_norm_new

    def _update(self, t):
        """
        Ingress a new data point and update the (top-k) matrix profile and matrix
        profile indices without egressing the oldest data point

        Parameters
        ----------
        t : float
            A single new data point to be appended to `T`
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

        mask = p_norm_new < config.STUMPY_P_NORM_THRESHOLD
        p_norm_new[mask] = 0

        D = np.power(p_norm_new, 1.0 / self._p)
        D[~self._T_subseq_isfinite] = np.inf
        if np.any(~self._T_isfinite[-self._m :]):
            D[:] = np.inf

        P_new = np.full(self._k, np.inf, dtype=np.float64)
        I_new = np.full(self._k, -1, dtype=np.int64)
        self._P = np.append(self._P, P_new.reshape(1, -1), axis=0)
        self._I = np.append(self._I, I_new.reshape(1, -1), axis=0)

        core._update_incremental_PI(D, self._P, self._I, self._excl_zone, n_appended=0)

        left_I_new = self._I[-1, 0]
        left_P_new = self._P[-1, 0]

        self._T = T_new
        self._left_P = np.append(self._left_P, left_P_new)
        self._left_I = np.append(self._left_I, left_I_new)
        self._p_norm = p_norm_new

    @property
    def P_(self):
        """
        Get the (top-k) matrix profile. When `k=1` (default), the output is
        a 1D array consisting of the matrix profile. When `k > 1`, the
        output is a 2D array that has exactly `k` columns and it consists of the
        top-k matrix profile.

        Parameters
        ----------
        None
        """
        if self._k == 1:
            return self._P.flatten().astype(np.float64)
        else:
            return self._P.astype(np.float64)

    @property
    def I_(self):
        """
        Get the (top-k) matrix profile indices. When `k=1` (default), the output is
        a 1D array consisting of the matrix profile indices. When `k > 1`, the
        output is a 2D array that has exactly `k` columns and it consists of the
        top-k matrix profile indices.

        Parameters
        ----------
        None
        """
        if self._k == 1:
            return self._I.flatten().astype(np.int64)
        else:
            return self._I.astype(np.int64)

    @property
    def left_P_(self):
        """
        Get the (top-1) left matrix profile

        Parameters
        ----------
        None
        """
        return self._left_P.astype(np.float64)

    @property
    def left_I_(self):
        """
        Get the (top-1) left matrix profile indices

        Parameters
        ----------
        None
        """
        return self._left_I.astype(np.int64)

    @property
    def T_(self):
        """
        Get the time series

        Parameters
        ----------
        None
        """
        return self._T
