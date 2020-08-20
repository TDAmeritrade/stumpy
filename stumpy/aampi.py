# STUMPY
# Copyright 2019 TD Ameritrade. Released under the terms of the 3-Clause BSD license.
# STUMPY is a trademark of TD Ameritrade IP Company, Inc. All rights reserved.

import numpy as np
from stumpy import core
import stumpy


class aampi(object):
    """
    Compute an incremental non-normalized (i.e., without z-normalization) matrix profile
    for streaming data. This is based on the aamp, on-line STOMPI, and STAMPI
    algorithms.

    Parameters
    ----------
    T : ndarray
        The time series or sequence for which the non-normalized matrix profile and
        matrix profile indices will be returned

    m : int
        Window size

    excl_zone : int
        The half width for the exclusion zone relative to the current
        sliding window

    Attributes
    ----------
    P_ : ndarray
        The updated matrix profile for `T`

    I_ : ndarray
        The updated matrix profile indices for `T`

    left_P_ : ndarray
        The updated left matrix profile for `T`

    left_I_ : ndarray
        The updated left matrix profile indices for `T`

    T_ : ndarray
        The updated time series or sequence for which the matrix profile and matrix
        profile indices are computed

    Methods
    -------
    update(t)
        Append a single new data point, `t`, to the time series, `T`, and update the
        matrix profile

    Notes
    -----
    `DOI: 10.1007/s10618-017-0519-9 \
    <https://www.cs.ucr.edu/~eamonn/MP_journal.pdf>`__

    See Table V

    Note that line 11 is missing an important `sqrt` operation!
    """

    def __init__(self, T, m, excl_zone=None):
        """
        Initialize the `stumpi` object

        Parameters
        ----------
        T : ndarray
            The time series or sequence for which the unnormalized matrix profile and
            matrix profile indices will be returned

        m : int
            Window size

        excl_zone : int
            The half width for the exclusion zone relative to the current
            sliding window
        """
        self._T = T
        self._m = m
        if excl_zone is not None:  # pragma: no cover
            self._excl_zone = excl_zone
        else:
            self._excl_zone = int(np.ceil(self._m / 4))

        mp = stumpy.aamp(self._T, self._m)
        self._P = mp[:, 0]
        self._I = mp[:, 1]
        self._left_I = mp[:, 2]
        self._left_P = np.empty(self._P.shape)
        self._left_P[:] = np.inf

        self._T = np.asarray(self._T)
        self._T = self._T.copy()
        self._T_isfinite = np.isfinite(self._T)
        self._T[np.isinf(self._T)] = np.nan
        self._T_subseq_isfinite = np.all(
            np.isfinite(core.rolling_window(self._T, m)), axis=1
        )
        self._T[np.isnan(self._T)] = 0

        # Retrieve the left matrix profile values
        for i, j in enumerate(self._left_I):
            if j >= 0:
                D = core.mass_absolute(
                    self._T[i : i + self._m], self._T[j : j + self._m]
                )
                self._left_P[i] = D[0]

        Q = self._T[-m:]
        self._QT = core.sliding_dot_product(Q, self._T)

    def update(self, t):
        """
        Append a single new data point, `t`, to the existing time series `T` and update
        the unnormalized matrix profile and matrix profile indices.

        Parameters
        ----------
        t : float
            A single new data point to be appended to `T`

        Notes
        -----
        `DOI: 10.1007/s10618-017-0519-9 \
        <https://www.cs.ucr.edu/~eamonn/MP_journal.pdf>`__

        See Table V

        Note that line 11 is missing an important `sqrt` operation!
        """
        n = self._T.shape[0]
        l = n - self._m + 1
        T_new = np.append(self._T, t)
        QT_new = np.empty(self._QT.shape[0] + 1)
        S = T_new[l:]
        t_drop = T_new[l - 1]

        if np.isfinite(t):
            self._T_isfinite = np.append(self._T_isfinite, True)
        else:
            self._T_isfinite = np.append(self._T_isfinite, False)
            t = 0
            T_new[-1] = 0
            S[-1] = 0

        T_subseq_isfinite = np.all(
            core.rolling_window(self._T_isfinite, self._m), axis=1
        )

        for j in range(l, 0, -1):
            QT_new[j] = (
                self._QT[j - 1] - T_new[j - 1] * t_drop + T_new[j + self._m - 1] * t
            )
        QT_new[0] = 0

        for j in range(self._m):
            QT_new[0] = QT_new[0] + T_new[j] * S[j]

        Q_squared = np.sum(S * S)
        T_squared = np.sum(core.rolling_window(T_new * T_new, self._m), axis=1)
        D = core._mass_absolute(Q_squared, T_squared, QT_new)
        D[~T_subseq_isfinite] = np.inf
        if np.any(~self._T_isfinite[-self._m :]):
            D[:] = np.inf

        core.apply_exclusion_zone(D, D.shape[0] - 1, self._excl_zone)

        for j in range(l):
            if D[j] < self._P[j]:
                self._I[j] = l
                self._P[j] = D[j]

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
        self._QT = QT_new

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
