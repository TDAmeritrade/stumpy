# STUMPY
# Copyright 2019 TD Ameritrade. Released under the terms of the 3-Clause BSD license.
# STUMPY is a trademark of TD Ameritrade IP Company, Inc. All rights reserved.

import numpy as np

from . import config, core, stump
from .aampi import aampi


@core.non_normalized(
    aampi,
    exclude=[
        "normalize",
        "T_subseq_isconstant_func",
    ],
)
class stumpi:
    """
    A class to compute an incremental z-normalized matrix profile for streaming data

    This is based on the on-line STOMPI and STAMPI algorithms.

    Parameters
    ----------
    T : numpy.ndarray
        The time series or sequence for which the matrix profile and matrix profile
        indices will be returned.

    m : int
        Window size.

    egress : bool, default True
        If set to ``True``, the oldest data point in the time series is removed and
        the time series length remains constant rather than forever increasing

    normalize : bool, default True
        When set to ``True``, this z-normalizes subsequences prior to computing
        distances. Otherwise, this class gets re-routed to its complementary
        non-normalized equivalent set in the ``@core.non_normalized`` class decorator.

    p : float, default 2.0
        The p-norm to apply for computing the Minkowski distance. This parameter is
        ignored when ``normalize == True``.

    k : int, default 1
        The number of top ``k`` smallest distances used to construct the matrix profile.
        Note that this will increase the total computational time and memory usage
        when ``k > 1``.

    mp : numpy.ndarray, default None
        A pre-computed matrix profile (and corresponding matrix profile indices).
        This is a 2D array of shape ``(len(T) - m + 1, 2 * k + 2)``, where the first
        ``k`` columns are top-k matrix profile, and the next ``k`` columns are their
        corresponding indices. The last two columns correspond to the top-1 left and
        top-1 right matrix profile indices. When ``None`` (default), this array is
        computed internally using ``stumpy.stump``.

    T_subseq_isconstant_func : function, default None
        A custom, user-defined function that returns a boolean array that indicates
        whether a subsequence in ``T`` is constant (``True``). The function must only
        take two arguments, ``a``, a 1-D array, and ``w``, the window size, while
        additional arguments may be specified by currying the user-defined function
        using ``functools.partial``. Any subsequence with at least one
        ``np.nan``/``np.inf`` will automatically have its corresponding value set to
        ``False`` in this boolean array.

    Attributes
    ----------
    P_ : numpy.ndarray
        The updated (top-k) matrix profile for ``T``. When ``k = 1`` (default), the
        first (and only) column in this 2D array consists of the matrix profile. When
        ``k > 1``, the output has exactly ``k`` columns consisting of the top-k matrix
        profile.

    I_ : numpy.ndarray
        The updated (top-k) matrix profile indices for ``T``. When ``k = 1`` (default),
        the first (and only) column in this 2D array consists of the matrix profile
        indices. When ``k > 1``, the output has exactly ``k`` columns consisting of the
        top-k matrix profile indices.

    left_P_ : numpy.ndarray
        The updated left (top-1) matrix profile for ``T``.

    left_I_ : numpy.ndarray
        The updated left (top-1) matrix profile indices for ``T``.

    T_ : numpy.ndarray
        The updated time series or sequence for which the matrix profile and matrix
        profile indices are computed.

    Methods
    -------
    update(t)
        Append a single new data point, ``t``, to the time series, ``T``, and update
        the matrix profile.

    Notes
    -----
    `DOI: 10.1007/s10618-017-0519-9 \
    <https://www.cs.ucr.edu/~eamonn/MP_journal.pdf>`__

    See Table V

    Note that line 11 is missing an important ``sqrt`` operation!

    Examples
    --------
    >>> import stumpy
    >>> import numpy as np
    >>> stream = stumpy.stumpi(
    ...     np.array([584., -11., 23., 79., 1001., 0.]),
    ...     m=3)
    >>> stream.update(-19.0)
    >>> stream.left_P_
    array([       inf, 3.00009263, 2.69407392, 3.05656417])
    >>> stream.left_I_
    array([-1,  0,  1,  2])
    """

    def __init__(
        self,
        T,
        m,
        egress=True,
        normalize=True,
        p=2.0,
        k=1,
        mp=None,
        T_subseq_isconstant_func=None,
    ):
        """
        Initialize the `stumpi` object

        Parameters
        ----------
        T : numpy.ndarray
            The time series or sequence for which the matrix profile and matrix profile
            indices will be returned

        m : int
            Window size

        egress : bool, default True
            If set to `True`, the oldest data point in the time series is removed and
            the time series length remains constant rather than forever increasing

        normalize : bool, default True
            When set to `True`, this z-normalizes subsequences prior to computing
            distances. Otherwise, this class gets re-routed to its complementary
            non-normalized equivalent set in the `@core.non_normalized` class decorator.

        p : float, default 2.0
            The p-norm to apply for computing the Minkowski distance. Minkowski distance
            is  typically used with `p` being 1 or 2, which correspond to the Manhattan
            distance and the Euclidean distance, respectively.This parameter is ignored
            when `normalize == True`.

        k : int, default 1
            The number of top `k` smallest distances used to construct the matrix
            profile. Note that this will increase the total computational time and
            memory usage when `k > 1`.

        mp : numpy.ndarray, default None
            A pre-computed matrix profile (and corresponding matrix profile indices).
            This is a 2D array of shape `(len(T) - m + 1, 2 * k + 2)`, where the first
            `k` columns are top-k matrix profile, and the next `k` columns are their
            corresponding indices. The last two columns correspond to the top-1 left
            and top-1 right matrix profile indices. When None (default), this array is
            computed internally using `stumpy.stump`.

        T_subseq_isconstant_func : function, default None
            A custom, user-defined function that returns a boolean array that indicates
            whether a subsequence in `T` is constant (True). The function must only take
            two arguments, `a`, a 1-D array, and `w`, the window size, while additional
            arguments may be specified by currying the user-defined function using
            `functools.partial`. Any subsequence with at least one np.nan/np.inf will
            automatically have its corresponding value set to False in this boolean
            array.
        """
        self._T = core._preprocess(T)
        core.check_window_size(m, max_size=self._T.shape[0])
        self._m = m
        self._k = k

        if T_subseq_isconstant_func is None:
            T_subseq_isconstant_func = core._rolling_isconstant
        if not callable(T_subseq_isconstant_func):  # pragma: no cover
            msg = (
                "`T_subseq_isconstant_func` was expected to be a callable function "
                + f"but {type(T_subseq_isconstant_func)} was found."
            )
            raise ValueError(msg)
        self._T_subseq_isconstant_func = T_subseq_isconstant_func

        self._n = self._T.shape[0]
        self._excl_zone = int(np.ceil(self._m / config.STUMPY_EXCL_ZONE_DENOM))
        self._T_isfinite = np.isfinite(self._T)
        self._egress = egress

        self._T_subseq_isconstant = core.process_isconstant(
            self._T, self._m, self._T_subseq_isconstant_func
        )

        if mp is None:
            mp = stump(
                self._T,
                self._m,
                k=self._k,
                T_A_subseq_isconstant=self._T_subseq_isconstant,
            )
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

        self._T, self._M_T, self._Σ_T, self._T_subseq_isconstant = core.preprocess(
            self._T, self._m, T_subseq_isconstant=self._T_subseq_isconstant
        )
        # Retrieve the left matrix profile values

        # Since each (top-1) matrix profile value is the minimum between the left
        # and right matrix profile values, we can save time by re-computing only
        # the left matrix profile value when the (top-1) matrix profile index is
        # equal to the right matrix profile index.
        mask = self._left_I == self._I[:, 0]
        self._left_P[mask] = self._P[mask, 0]

        # Only re-compute the `i`-th left matrix profile value, `self._left_P[i]`,
        # when `self._left_I[i] != self._I[i, 0]`
        for i in np.flatnonzero(self._left_I >= 0 & ~mask):
            j = self._left_I[i]
            QT = np.dot(self._T[i : i + self._m], self._T[j : j + self._m])
            D_square = core._calculate_squared_distance(
                self._m,
                QT,
                self._M_T[i],
                self._Σ_T[i],
                self._M_T[j],
                self._Σ_T[j],
                self._T_subseq_isconstant[i],
                self._T_subseq_isconstant[j],
            )
            self._left_P[i] = np.sqrt(D_square)

        Q = self._T[-self._m :]
        self._QT = core.sliding_dot_product(Q, self._T)
        if self._egress:
            self._QT_new = np.empty(self._QT.shape[0], dtype=np.float64)
            self._n_appended = 0

    def update(self, t):
        """
        Append a single new data point, `t`, to the existing time series `T` and update
        the (top-k) matrix profile and matrix profile indices.

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
        if self._egress:
            self._update_egress(t)
        else:
            self._update(t)

    def _update_egress(self, t):
        """
        Ingress a new data point, egress the oldest data point, and update the (top-k)
        matrix profile and matrix profile indices

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
        self._QT[:-1] = self._QT[1:]
        S = self._T[l:]
        t_drop = self._T[l - 1]
        self._T_isfinite[:-1] = self._T_isfinite[1:]

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

        if np.any(~self._T_isfinite[-self._m :]):
            μ_Q = np.inf
            σ_Q = np.nan
            Q_subseq_isconstant = False
        else:
            Q_subseq_isconstant = core.process_isconstant(
                S, self._m, self._T_subseq_isconstant_func
            )[0]
            μ_Q, σ_Q = [arr[0] for arr in core.compute_mean_std(S, self._m)]

        self._M_T[:-1] = self._M_T[1:]
        self._Σ_T[:-1] = self._Σ_T[1:]
        self._T_subseq_isconstant[:-1] = self._T_subseq_isconstant[1:]

        self._M_T[-1] = μ_Q
        self._Σ_T[-1] = σ_Q
        self._T_subseq_isconstant[-1] = Q_subseq_isconstant

        self._QT_new[1:] = self._QT[:l] - self._T[:l] * t_drop + self._T[self._m :] * t
        self._QT_new[0] = np.sum(self._T[: self._m] * S[: self._m])

        D = core.calculate_distance_profile(
            self._m,
            self._QT_new,
            μ_Q,
            σ_Q,
            self._M_T,
            self._Σ_T,
            Q_subseq_isconstant,
            self._T_subseq_isconstant,
        )
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

        self._QT[:] = self._QT_new

    def _update(self, t):
        """
        Ingress a new data point and update the (top-k) matrix profile and matrix
        profile indices without egressing the oldest data point

        Parameters
        ----------
        t : float
            A single new data point to be appended to `T`
        """
        n = self._T.shape[0]
        l = n - self._m + 1
        T_new = np.append(self._T, t)
        QT_new = np.empty(self._QT.shape[0] + 1, dtype=np.float64)
        S = T_new[l:]
        t_drop = T_new[l - 1]

        if np.isfinite(t):
            self._T_isfinite = np.append(self._T_isfinite, True)
        else:
            self._T_isfinite = np.append(self._T_isfinite, False)
            t = 0
            T_new[-1] = 0
            S[-1] = 0

        if np.any(~self._T_isfinite[-self._m :]):
            μ_Q = np.inf
            σ_Q = np.nan
            Q_subseq_isconstant = False
        else:
            Q_subseq_isconstant = core.process_isconstant(
                S, self._m, self._T_subseq_isconstant_func
            )[0]
            μ_Q, σ_Q = [arr[0] for arr in core.compute_mean_std(S, self._m)]

        M_T_new = np.append(self._M_T, μ_Q)
        Σ_T_new = np.append(self._Σ_T, σ_Q)
        T_subseq_isconstant_new = np.append(
            self._T_subseq_isconstant, Q_subseq_isconstant
        )

        QT_new[1:] = self._QT[:l] - T_new[:l] * t_drop + T_new[self._m :] * t
        QT_new[0] = np.sum(T_new[: self._m] * S[: self._m])

        D = core.calculate_distance_profile(
            self._m,
            QT_new,
            μ_Q,
            σ_Q,
            M_T_new,
            Σ_T_new,
            Q_subseq_isconstant,
            T_subseq_isconstant_new,
        )
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
        self._QT = QT_new
        self._M_T = M_T_new
        self._Σ_T = Σ_T_new
        self._T_subseq_isconstant = T_subseq_isconstant_new

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

        Returns
        -------
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

        Returns
        -------
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

        Returns
        -------
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

        Returns
        -------
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

        Returns
        -------
        None
        """
        return self._T
