# STUMPY
# Copyright 2019 TD Ameritrade. Released under the terms of the 3-Clause BSD license.
# STUMPY is a trademark of TD Ameritrade IP Company, Inc. All rights reserved.

import numpy as np
from stumpy import core
import stumpy


class stumpi(object):
    """
    Compute an incremental matrix profile for streaming data. This is based on the
    on-line STOMPI and STAMPI algorithms.

    Attributes
    ----------
    P : ndarray
        The matrix profile for `T`

    I : ndarray
        The matrix profile indices for `T`

    T : ndarray
        The time series or sequence for which the matrix profile and matrix profile
        indices will be computed

    Methods
    -------
    add(t)
        Append a single new data point to the time series and update the matrix profile
    """

    def __init__(self, T, m, excl_zone=None):
        """
        Initialize the `stumpi` object

        Parameters
        ----------
        T : ndarray
            The time series or sequence for which the matrix profile and matrix profile
            indices will be returned

        m : int
            Window size
        """
        self._T = T
        self._m = m
        if excl_zone is not None:  # pragma: no cover
            self._excl_zone = excl_zone
        else:
            self._excl_zone = int(np.ceil(self._m / 4))

        mp = stumpy.stump(self._T, self._m)
        self._P = mp[:, 0]
        self._I = mp[:, 1]

        self._T, self._M_T, self._Σ_T = core.preprocess(self._T, self._m)

        Q = self._T[-m:]
        self._QT = core.sliding_dot_product(Q, self._T)

    def add(self, t):
        """
        Append a single new data point to `T` and update the matrix profile.

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

        for j in range(l, 0, -1):
            QT_new[j] = (
                self._QT[j - 1] - T_new[j - 1] * t_drop + T_new[j + self._m - 1] * t
            )
        QT_new[0] = 0

        for j in range(self._m):
            QT_new[0] = QT_new[0] + T_new[j] * S[j]

        μ_Q = self._M_T[l - 1] + (t - t_drop) / self._m
        σ_Q = np.sqrt(
            self._Σ_T[l - 1] * self._Σ_T[l - 1]
            + self._M_T[l - 1] * self._M_T[l - 1]
            + (t * t - t_drop * t_drop) / self._m
            - μ_Q * μ_Q
        )

        M_T_new = np.append(self._M_T, μ_Q)
        Σ_T_new = np.append(self._Σ_T, σ_Q)
        D = core.calculate_distance_profile(self._m, QT_new, μ_Q, σ_Q, M_T_new, Σ_T_new)

        core.apply_exclusion_zone(D, D.shape[0] - 1, self._excl_zone)

        for j in range(l):
            if D[j] < self._P[j]:
                self._I[j] = l
                self._P[j] = D[j]

        I_last = np.argmin(D)
        I_new = np.append(self._I, I_last)
        P_new = np.append(self._P, D[I_last])

        self._T = T_new
        self._P = P_new
        self._I = I_new
        self._QT = QT_new
        self._M_T = M_T_new
        self._Σ_T = Σ_T_new

    @property
    def P(self):
        """
        Get the matrix profile
        """
        return self._P

    @property
    def I(self):
        """
        Get the matrix profile indices
        """
        return self._I

    @property
    def T(self):
        """
        Get the time series
        """
        return self._T
