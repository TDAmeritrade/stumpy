# STUMPY
# Copyright 2019 TD Ameritrade. Released under the terms of the 3-Clause BSD license.
# STUMPY is a trademark of TD Ameritrade IP Company, Inc. All rights reserved.

import logging

import numpy as np
from numba import njit, prange
import numba

from . import core, scraamp, config
from .stump import _stump

logger = logging.getLogger(__name__)


@njit(fastmath=True)
def _compute_PI(
    T_A,
    T_B,
    m,
    M_T,
    Σ_T,
    μ_Q,
    σ_Q,
    indices,
    start,
    stop,
    thread_idx,
    s,
    P_squared,
    I,
    excl_zone=None,
):
    """
    Compute (Numba JIT-compiled) and update the squared matrix profile distance
    and matrix profile indces according to the preSCRIMP algorithm

    Parameters
    ----------
    T_A : numpy.ndarray
        The time series or sequence for which to compute the matrix profile

    T_B : numpy.ndarray
        The time series or sequence that will be used to annotate T_A. For every
        subsequence in T_A, its nearest neighbor in T_B will be recorded.

    m : int
        Window size

    M_T : numpy.ndarray
        Sliding window mean for T_A

    Σ_T : numpy.ndarray
        Sliding window standard deviation for T_A

    μ_Q : numpy.ndarray
        Mean of the query sequence, `Q`, relative to the current sliding window in `T_B`

    σ_Q : numpy.ndarray
        Standard deviation of the query sequence, `Q`, relative to the current
        sliding window in `T_B`

    indices : numpy.ndarray
        The subsequence indices to compute `prescrump` for

    start : int
        The (inclusive) start index for `indices`

    stop : int
        The (exclusive) stop index for `indices`

    thread_idx : int
        The thread index

    s : int
        The sampling interval that defaults to
        `int(np.ceil(m / config.STUMPY_EXCL_ZONE_DENOM))`

    P_squared : numpy.ndarray
        The squared matrix profile

    I : numpy.ndarray
        The matrix profile indices

    excl_zone : int
        The half width for the exclusion zone relative to the `i`.

    Notes
    -----
    `DOI: 10.1109/ICDM.2018.00099 \
    <https://www.cs.ucr.edu/~eamonn/SCRIMP_ICDM_camera_ready_updated.pdf>`__

    See Algorithm 2
    """
    l = T_B.shape[0] - m + 1
    squared_distance_profile = np.empty(l)
    QT = np.empty(l, dtype=np.float64)
    for i in indices[start:stop]:
        Q = T_A[i : i + m]
        QT[:] = core._sliding_dot_product(Q, T_B)
        # Update P[i] relative to all T[j : j + m]
        squared_distance_profile[:] = core._mass(Q, T_B, QT, μ_Q[i], σ_Q[i], M_T, Σ_T)
        squared_distance_profile[:] = np.square(squared_distance_profile)
        if excl_zone is not None:
            zone_start = max(0, i - excl_zone)
            zone_stop = min(l, i + excl_zone)
            squared_distance_profile[zone_start : zone_stop + 1] = np.inf
        I[thread_idx, i] = np.argmin(squared_distance_profile)
        P_squared[thread_idx, i] = squared_distance_profile[I[thread_idx, i]]
        if P_squared[thread_idx, i] == np.inf:  # pragma: no cover
            I[thread_idx, i] = -1
        else:
            j = I[thread_idx, i]
            # Given the squared distance, work backwards and compute QT
            QT_j = (m - P_squared[thread_idx, i] / 2.0) * (Σ_T[j] * σ_Q[i]) + (
                m * M_T[j] * μ_Q[i]
            )
            QT_j_prime = QT_j
            for k in range(1, min(s, l - max(i, j))):
                QT_j = (
                    QT_j
                    - T_B[i + k - 1] * T_A[j + k - 1]
                    + T_B[i + k + m - 1] * T_A[j + k + m - 1]
                )
                D_squared = core._calculate_squared_distance(
                    m,
                    QT_j,
                    M_T[i + k],
                    Σ_T[i + k],
                    μ_Q[j + k],
                    σ_Q[j + k],
                )
                if D_squared < P_squared[thread_idx, i + k]:
                    P_squared[thread_idx, i + k] = D_squared
                    I[thread_idx, i + k] = j + k
                if D_squared < P_squared[thread_idx, j + k]:
                    P_squared[thread_idx, j + k] = D_squared
                    I[thread_idx, j + k] = i + k
            QT_j = QT_j_prime
            for k in range(1, min(s, i + 1, j + 1)):
                QT_j = QT_j - T_B[i - k + m] * T_A[j - k + m] + T_B[i - k] * T_A[j - k]
                D_squared = core._calculate_squared_distance(
                    m,
                    QT_j,
                    M_T[i - k],
                    Σ_T[i - k],
                    μ_Q[j - k],
                    σ_Q[j - k],
                )
                if D_squared < P_squared[thread_idx, i - k]:
                    P_squared[thread_idx, i - k] = D_squared
                    I[thread_idx, i - k] = j - k
                if D_squared < P_squared[thread_idx, j - k]:
                    P_squared[thread_idx, j - k] = D_squared
                    I[thread_idx, j - k] = i - k


@njit(
    # "(f8[:], f8[:], i8, f8[:], f8[:], f8[:], f8[:], f8[:], i8, i8, f8[:], f8[:],"
    # "i8[:], optional(i8))",
    parallel=True,
    fastmath=True,
)
def _prescrump(
    T_A,
    T_B,
    m,
    M_T,
    Σ_T,
    μ_Q,
    σ_Q,
    indices,
    s,
    excl_zone=None,
):
    """
    A Numba JIT-compiled implementation of the preSCRIMP algorithm.

    Parameters
    ----------
    T_A : numpy.ndarray
        The time series or sequence for which to compute the matrix profile

    T_B : numpy.ndarray
        The time series or sequence that will be used to annotate T_A. For every
        subsequence in T_A, its nearest neighbor in T_B will be recorded.

    m : int
        Window size

    M_T : numpy.ndarray
        Sliding window mean for T_A

    Σ_T : numpy.ndarray
        Sliding window standard deviation for T_A

    μ_Q : numpy.ndarray
        Mean of the query sequence, `Q`, relative to the current sliding window in `T_B`

    σ_Q : numpy.ndarray
        Standard deviation of the query sequence, `Q`, relative to the current
        sliding window in `T_B`

    indices : numpy.ndarray
        The subsequence indices to compute `prescrump` for

    idx_ranges : numpy.ndarray
        The (inclusive) start indices and (exclusive) stop indices referenced
        in the `indices` array

    s : int
        The sampling interval that defaults to
        `int(np.ceil(m / config.STUMPY_EXCL_ZONE_DENOM))`

    P_squared : numpy.ndarray
        The squared matrix profile

    I : numpy.ndarray
        The matrix profile indices

    excl_zone : int
        The half width for the exclusion zone relative to the `i`.

    Notes
    -----
    `DOI: 10.1109/ICDM.2018.00099 \
    <https://www.cs.ucr.edu/~eamonn/SCRIMP_ICDM_camera_ready_updated.pdf>`__

    See Algorithm 2
    """
    n_threads = numba.config.NUMBA_NUM_THREADS
    l = T_A.shape[0] - m + 1
    P_squared = np.full((n_threads, l), np.inf, dtype=np.float64)
    I = np.full((n_threads, l), -1, dtype=np.int64)

    idx_ranges = core._get_ranges(len(indices), n_threads, truncate=False)
    for thread_idx in prange(n_threads):
        _compute_PI(
            T_A,
            T_B,
            m,
            M_T,
            Σ_T,
            μ_Q,
            σ_Q,
            indices,
            idx_ranges[thread_idx, 0],
            idx_ranges[thread_idx, 1],
            thread_idx,
            s,
            P_squared,
            I,
            excl_zone,
        )

    for thread_idx in range(1, n_threads):
        for i in range(l):
            if P_squared[thread_idx, i] < P_squared[0, i]:
                P_squared[0, i] = P_squared[thread_idx, i]
                I[0, i] = I[thread_idx, i]

    return np.sqrt(P_squared[0]), I[0]


@core.non_normalized(scraamp.prescraamp)
def prescrump(T_A, m, T_B=None, s=None, normalize=True, p=2.0):
    """
    A convenience wrapper around the Numba JIT-compiled parallelized `_prescrump`
    function which computes the approximate matrix profile according to the preSCRIMP
    algorithm

    Parameters
    ----------
    T_A : numpy.ndarray
        The time series or sequence for which to compute the matrix profile

    m : int
        Window size

    T_B : numpy.ndarray, default None
        The time series or sequence that will be used to annotate T_A. For every
        subsequence in T_A, its nearest neighbor in T_B will be recorded.

    s : int, default None
        The sampling interval that defaults to
        `int(np.ceil(m / config.STUMPY_EXCL_ZONE_DENOM))`

    normalize : bool, default True
        When set to `True`, this z-normalizes subsequences prior to computing distances.
        Otherwise, this function gets re-routed to its complementary non-normalized
        equivalent set in the `@core.non_normalized` function decorator.

    p : float, default 2.0
        The p-norm to apply for computing the Minkowski distance. This parameter is
        ignored when `normalize == True`.

    Returns
    -------
    P : numpy.ndarray
        Matrix profile

    I : numpy.ndarray
        Matrix profile indices

    Notes
    -----
    `DOI: 10.1109/ICDM.2018.00099 \
    <https://www.cs.ucr.edu/~eamonn/SCRIMP_ICDM_camera_ready_updated.pdf>`__

    See Algorithm 2
    """
    if T_B is None:
        T_B = T_A
        excl_zone = int(np.ceil(m / config.STUMPY_EXCL_ZONE_DENOM))
    else:
        excl_zone = None

    T_A, μ_Q, σ_Q = core.preprocess(T_A, m)
    T_B, M_T, Σ_T = core.preprocess(T_B, m)

    n_A = T_A.shape[0]
    l = n_A - m + 1

    if s is None:  # pragma: no cover
        s = excl_zone

    indices = np.random.permutation(range(0, l, s)).astype(np.int64)
    P, I = _prescrump(
        T_A,
        T_B,
        m,
        M_T,
        Σ_T,
        μ_Q,
        σ_Q,
        indices,
        s,
        excl_zone,
    )

    return P, I


@core.non_normalized(
    scraamp.scraamp,
    exclude=["normalize", "pre_scrump", "pre_scraamp", "p"],
    replace={"pre_scrump": "pre_scraamp"},
)
class scrump:
    """
    Compute an approximate z-normalized matrix profile

    This is a convenience wrapper around the Numba JIT-compiled parallelized
    `_stump` function which computes the matrix profile according to SCRIMP.

    Parameters
    ----------
    T_A : numpy.ndarray
        The time series or sequence for which to compute the matrix profile

    T_B : numpy.ndarray
        The time series or sequence that will be used to annotate T_A. For every
        subsequence in T_A, its nearest neighbor in T_B will be recorded.

    m : int
        Window size

    ignore_trivial : bool
        Set to `True` if this is a self-join. Otherwise, for AB-join, set this to
        `False`. Default is `True`.

    percentage : float
        Approximate percentage completed. The value is between 0.0 and 1.0.

    pre_scrump : bool
        A flag for whether or not to perform the PreSCRIMP calculation prior to
        computing SCRIMP. If set to `True`, this is equivalent to computing
        SCRIMP++ and may lead to faster convergence

    s : int
        The size of the PreSCRIMP fixed interval. If `pre_scrump=True` and `s=None`,
        then `s` will automatically be set to
        `s=int(np.ceil(m / config.STUMPY_EXCL_ZONE_DENOM))`, the size of the exclusion
        zone.

    normalize : bool, default True
        When set to `True`, this z-normalizes subsequences prior to computing distances.
        Otherwise, this class gets re-routed to its complementary non-normalized
        equivalent set in the `@core.non_normalized` class decorator.

    p : float, default 2.0
        The p-norm to apply for computing the Minkowski distance. This parameter is
        ignored when `normalize == True`.

    Attributes
    ----------
    P_ : numpy.ndarray
        The updated matrix profile

    I_ : numpy.ndarray
        The updated matrix profile indices

    Methods
    -------
    update()
        Update the matrix profile and the matrix profile indices by computing
        additional new distances (limited by `percentage`) that make up the full
        distance matrix.

    See Also
    --------
    stumpy.stump : Compute the z-normalized matrix profile
    stumpy.stumped : Compute the z-normalized matrix profile with a distributed dask
        cluster
    stumpy.gpu_stump : Compute the z-normalized matrix profile with one or more GPU
        devices

    Notes
    -----
    `DOI: 10.1109/ICDM.2018.00099 \
    <https://www.cs.ucr.edu/~eamonn/SCRIMP_ICDM_camera_ready_updated.pdf>`__

    See Algorithm 1 and Algorithm 2

    Examples
    --------
    >>> approx_mp = stumpy.scrump(
    ...     np.array([584., -11., 23., 79., 1001., 0., -19.]),
    ...     m=3)
    >>> approx_mp.update()
    >>> approx_mp._P
    array([[2.982409  ,        inf, 2.982409  ],
           [3.28412702,        inf, 3.28412702],
           [       inf,        inf,        inf],
           [2.982409  , 2.982409  ,        inf],
           [3.28412702, 3.28412702,        inf]])
    >>> approx_mp._I
    array([[ 3, -1,  3],
           [ 4, -1,  4],
           [-1, -1, -1],
           [ 0,  0, -1],
           [ 1,  1, -1]])
    """

    def __init__(
        self,
        T_A,
        m,
        T_B=None,
        ignore_trivial=True,
        percentage=0.01,
        pre_scrump=False,
        s=None,
        normalize=True,
        p=2.0,
    ):
        """
        Initialize the `scrump` object

        Parameters
        ----------
        T_A : numpy.ndarray
            The time series or sequence for which to compute the matrix profile

        m : int
            Window size

        T_B : numpy.ndarray, default None
            The time series or sequence that will be used to annotate T_A. For every
            subsequence in T_A, its nearest neighbor in T_B will be recorded.

        ignore_trivial : bool, default True
            Set to `True` if this is a self-join. Otherwise, for AB-join, set this to
            `False`. Default is `True`.

        percentage : float, default 0.01
            Approximate percentage completed. The value is between 0.0 and 1.0.

        pre_scrump : bool, default False
            A flag for whether or not to perform the PreSCRIMP calculation prior to
            computing SCRIMP. If set to `True`, this is equivalent to computing
            SCRIMP++

        s : int, default None
            The size of the PreSCRIMP fixed interval. If `pre_scrump=True` and `s=None`,
            then `s` will automatically be set to
            `s=int(np.ceil(m / config.STUMPY_EXCL_ZONE_DENOM))`, the size of the
            exclusion zone.

        normalize : bool, default True
            When set to `True`, this z-normalizes subsequences prior to computing
            distances. Otherwise, this class gets re-routed to its complementary
            non-normalized equivalent set in the `@core.non_normalized` class decorator.

        p : float, default 2.0
            The p-norm to apply for computing the Minkowski distance. This parameter is
            ignored when `normalize == True`.
        """
        self._ignore_trivial = ignore_trivial

        if T_B is None:
            T_B = T_A
            self._ignore_trivial = True

        self._m = m
        (
            self._T_A,
            self._μ_Q,
            self._σ_Q_inverse,
            self._μ_Q_m_1,
            self._T_A_subseq_isfinite,
            self._T_A_subseq_isconstant,
        ) = core.preprocess_diagonal(T_A, self._m)

        (
            self._T_B,
            self._M_T,
            self._Σ_T_inverse,
            self._M_T_m_1,
            self._T_B_subseq_isfinite,
            self._T_B_subseq_isconstant,
        ) = core.preprocess_diagonal(T_B, self._m)

        if self._T_A.ndim != 1:  # pragma: no cover
            raise ValueError(
                f"T_A is {self._T_A.ndim}-dimensional and must be 1-dimensional. "
                "For multidimensional STUMP use `stumpy.mstump` or `stumpy.mstumped`"
            )

        if self._T_B.ndim != 1:  # pragma: no cover
            raise ValueError(
                f"T_B is {self._T_B.ndim}-dimensional and must be 1-dimensional. "
                "For multidimensional STUMP use `stumpy.mstump` or `stumpy.mstumped`"
            )

        core.check_window_size(m, max_size=min(T_A.shape[0], T_B.shape[0]))

        if self._ignore_trivial is False and core.are_arrays_equal(
            self._T_A, self._T_B
        ):  # pragma: no cover
            logger.warning("Arrays T_A, T_B are equal, which implies a self-join.")
            logger.warning("Try setting `ignore_trivial = True`.")

        if (
            self._ignore_trivial
            and core.are_arrays_equal(self._T_A, self._T_B) is False
        ):  # pragma: no cover
            logger.warning("Arrays T_A, T_B are not equal, which implies an AB-join.")
            logger.warning("Try setting `ignore_trivial = False`.")

        self._n_A = self._T_A.shape[0]
        self._n_B = self._T_B.shape[0]
        self._l = self._n_A - self._m + 1

        self._P = np.empty((self._l, 3), dtype=np.float64)
        self._I = np.empty((self._l, 3), dtype=np.int64)
        self._P[:, :] = np.inf
        self._I[:, :] = -1

        self._excl_zone = int(np.ceil(self._m / config.STUMPY_EXCL_ZONE_DENOM))

        if s is None:
            s = self._excl_zone

        if pre_scrump:
            if self._ignore_trivial:
                P, I = prescrump(T_A, m, s=s)
            else:
                P, I = prescrump(T_A, m, T_B=T_B, s=s)
            for i in range(P.shape[0]):
                if self._P[i, 0] > P[i]:
                    self._P[i, 0] = P[i]
                    self._I[i, 0] = I[i]

        if self._ignore_trivial:
            self._diags = np.random.permutation(
                range(self._excl_zone + 1, self._n_A - self._m + 1)
            ).astype(np.int64)
            if self._diags.shape[0] == 0:  # pragma: no cover
                max_m = core.get_max_window_size(self._T_A.shape[0])
                raise ValueError(
                    f"The window size, `m = {self._m}`, is too long for a self join. "
                    f"Please try a value of `m <= {max_m}`"
                )
        else:
            self._diags = np.random.permutation(
                range(-(self._n_A - self._m + 1) + 1, self._n_B - self._m + 1)
            ).astype(np.int64)

        self._n_threads = numba.config.NUMBA_NUM_THREADS
        self._percentage = np.clip(percentage, 0.0, 1.0)
        self._n_chunks = int(np.ceil(1.0 / percentage))
        self._ndist_counts = core._count_diagonal_ndist(
            self._diags, self._m, self._n_A, self._n_B
        )
        self._chunk_diags_ranges = core._get_array_ranges(
            self._ndist_counts, self._n_chunks, True
        )
        self._n_chunks = self._chunk_diags_ranges.shape[0]
        self._chunk_idx = 0

    def update(self):
        """
        Update the matrix profile and the matrix profile indices by computing
        additional new distances (limited by `percentage`) that make up the full
        distance matrix.
        """
        if self._chunk_idx < self._n_chunks:
            start_idx, stop_idx = self._chunk_diags_ranges[self._chunk_idx]

            P, I = _stump(
                self._T_A,
                self._T_B,
                self._m,
                self._M_T,
                self._μ_Q,
                self._Σ_T_inverse,
                self._σ_Q_inverse,
                self._M_T_m_1,
                self._μ_Q_m_1,
                self._T_A_subseq_isfinite,
                self._T_B_subseq_isfinite,
                self._T_A_subseq_isconstant,
                self._T_B_subseq_isconstant,
                self._diags[start_idx:stop_idx],
                self._ignore_trivial,
            )

            # Update matrix profile and indices
            for i in range(self._P.shape[0]):
                if self._P[i, 0] > P[i, 0]:
                    self._P[i, 0] = P[i, 0]
                    self._I[i, 0] = I[i, 0]
                # left matrix profile and left matrix profile indices
                if self._P[i, 1] > P[i, 1]:
                    self._P[i, 1] = P[i, 1]
                    self._I[i, 1] = I[i, 1]
                # right matrix profile and right matrix profile indices
                if self._P[i, 2] > P[i, 2]:
                    self._P[i, 2] = P[i, 2]
                    self._I[i, 2] = I[i, 2]

            self._chunk_idx += 1

    @property
    def P_(self):
        """
        Get the updated matrix profile
        """
        return self._P[:, 0].astype(np.float64)

    @property
    def I_(self):
        """
        Get the updated matrix profile indices
        """
        return self._I[:, 0].astype(np.int64)

    @property
    def left_I_(self):
        """
        Get the updated left matrix profile indices
        """
        return self._I[:, 1].astype(np.int64)

    @property
    def right_I_(self):
        """
        Get the updated right matrix profile indices
        """
        return self._I[:, 2].astype(np.int64)
