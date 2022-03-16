# STUMPY
# Copyright 2019 TD Ameritrade. Released under the terms of the 3-Clause BSD license.
# STUMPY is a trademark of TD Ameritrade IP Company, Inc. All rights reserved.

import numpy as np
from . import core, aamp, scraamp, aamped


def _normalize_pan(pan, ms, bfs_indices, n_processed, T_min, T_max, p=2.0):
    """
    Normalize the pan matrix profile nearest neighbor distances (inplace) relative
    to the corresponding subsequence length from which they were computed

    Parameters
    ----------
    pan : numpy.ndarray
        The pan matrix profile

    ms : numpy.ndarray
        The breadth-first-search sorted subsequence window sizes

    bfs_indices : numpy.ndarray
        The breadth-first-search indices

    n_processed : numpy.ndarray
        The number of subsequence window sizes and breadth-first-search indices to
        normalize

    T_min : float
        The min value in `T`

    T_max : float
        The max value in `T`

    p : float, default 2.0
        The p-norm to apply for computing the Minkowski distance.

    Returns
    -------
    None
    """
    idx = bfs_indices[:n_processed]
    norm = 1.0 / (np.abs(T_max - T_min) * np.power(ms[:n_processed], 1.0 / p))
    pan[idx] = np.minimum(1.0, pan[idx] * norm[:, np.newaxis])


class _aamp_stimp:
    """
    Compute the Pan Matrix Profile

    This is based on the SKIMP algorithm.

    Parameters
    ----------
    T : numpy.ndarray
        The time series or sequence for which to compute the pan matrix profile

    m_start : int, default 3
        The starting (or minimum) subsequence window size for which a matrix profile
        may be computed

    m_stop : int, default None
        The stopping (or maximum) subsequence window size for which a matrix profile
        may be computed. When `m_stop = Non`, this is set to the maximum allowable
        subsequence window size

    m_step : int, default 1
        The step between subsequence window sizes

    percentage : float, default 0.01
        The percentage of the full matrix profile to compute for each subsequence
        window size. When `percentage < 1.0`, then the `scraamp` algorithm is used.
        Otherwise, the `stump` algorithm is used when the exact matrix profile is
        requested.

    pre_scraamp : bool, default True
        A flag for whether or not to perform the PreSCRIMP calculation prior to
        computing SCRIMP. If set to `True`, this is equivalent to computing
        SCRIMP++. This parameter is ignored when `percentage = 1.0`.

    dask_client : client, default None
        A Dask Distributed client that is connected to a Dask scheduler and
        Dask workers. Setting up a Dask distributed cluster is beyond the
        scope of this library. Please refer to the Dask Distributed
        documentation.

    device_id : int or list, default None
        The (GPU) device number to use. The default value is `0`. A list of
        valid device ids (int) may also be provided for parallel GPU-STUMP
        computation. A list of all valid device ids can be obtained by
        executing `[device.id for device in numba.cuda.list_devices()]`.

    mp_func : object, default stump
        The matrix profile function to use when `percentage = 1.0`

    p : float, default 2.0
        The p-norm to apply for computing the Minkowski distance.

    Attributes
    ----------
    PAN_ : numpy.ndarray
        The transformed (i.e., normalized, contrasted, binarized, and repeated)
        pan matrix profile

    M_ : numpy.ndarray
        The full list of (breadth first search (level) ordered) subsequence window
        sizes

    Methods
    -------
    update():
        Compute the next matrix profile using the next available (breadth-first-search
        (level) ordered) subsequence window size and update the pan matrix profile

    Notes
    -----
    `DOI: 10.1109/ICBK.2019.00031 \
    <https://www.cs.ucr.edu/~eamonn/PAN_SKIMP%20%28Matrix%20Profile%20XX%29.pdf>`__

    See Table 2
    """

    def __init__(
        self,
        T,
        min_m=3,
        max_m=None,
        step=1,
        percentage=0.01,
        pre_scraamp=True,
        dask_client=None,
        device_id=None,
        mp_func=aamp,
        p=2.0,
    ):
        """
        Initialize the `stimp` object and compute the Pan Matrix Profile

        Parameters
        ----------
        T : numpy.ndarray
            The time series or sequence for which to compute the pan matrix profile

        min_m : int, default 3
            The minimum subsequence window size to consider computing a matrix profile
            for

        max_m : int, default None
            The maximum subsequence window size to consider computing a matrix profile
            for. When `max_m = None`, this is set to the maximum allowable subsequence
            window size

        step : int, default 1
            The step between subsequence window sizes

        percentage : float, default 0.01
            The percentage of the full matrix profile to compute for each subsequence
            window size. When `percentage < 1.0`, then the `scraamp` algorithm is used.
            Otherwise, the `stump` algorithm is used when the exact matrix profile is
            requested.

        pre_scraamp : bool, default True
            A flag for whether or not to perform the PreSCRIMP calculation prior to
            computing SCRIMP. If set to `True`, this is equivalent to computing
            SCRIMP++. This parameter is ignored when `percentage = 1.0`.

        dask_client : client, default None
            A Dask Distributed client that is connected to a Dask scheduler and
            Dask workers. Setting up a Dask distributed cluster is beyond the
            scope of this library. Please refer to the Dask Distributed
            documentation.

        device_id : int or list, default None
            The (GPU) device number to use. The default value is `0`. A list of
            valid device ids (int) may also be provided for parallel GPU-STUMP
            computation. A list of all valid device ids can be obtained by
            executing `[device.id for device in numba.cuda.list_devices()]`.

        mp_func : object, default stump
            The matrix profile function to use when `percentage = 1.0`

        p : float, default 2.0
            The p-norm to apply for computing the Minkowski distance.
        """
        self._T = T.copy()
        self._T_min = np.min(self._T[np.isfinite(self._T)])
        self._T_max = np.max(self._T[np.isfinite(self._T)])
        self._p = p
        if max_m is None:
            max_m = max(min_m + 1, core.get_max_window_size(self._T.shape[0]))
            M = np.arange(min_m, max_m + 1, step).astype(np.int64)
        else:
            min_m, max_m = sorted([min_m, max_m])
            M = np.arange(
                max(3, min_m),
                min(core.get_max_window_size(self._T.shape[0]), max_m) + 1,
                step,
            ).astype(np.int64)
        self._bfs_indices = core._bfs_indices(M.shape[0])
        self._M = M[self._bfs_indices]
        self._n_processed = 0
        percentage = np.clip(percentage, 0.0, 1.0)
        self._percentage = percentage
        self._pre_scraamp = pre_scraamp
        partial_mp_func = core._get_partial_mp_func(
            mp_func, dask_client=dask_client, device_id=device_id
        )
        self._mp_func = partial_mp_func

        self._PAN = np.full(
            (self._M.shape[0], self._T.shape[0]), fill_value=np.inf, dtype=np.float64
        )

    def update(self):
        """
        Update the pan matrix profile by computing a single matrix profile using the
        next available subsequence window size

        Notes
        -----
        `DOI: 10.1109/ICBK.2019.00031 \
        <https://www.cs.ucr.edu/~eamonn/PAN_SKIMP%20%28Matrix%20Profile%20XX%29.pdf>`__

        See Table 2
        """
        if self._n_processed < self._M.shape[0]:
            m = self._M[self._n_processed]
            if self._percentage < 1.0:
                approx = scraamp(
                    self._T,
                    m,
                    ignore_trivial=True,
                    percentage=self._percentage,
                    pre_scraamp=self._pre_scraamp,
                    p=self._p,
                )
                approx.update()
                self._PAN[
                    self._bfs_indices[self._n_processed], : approx.P_.shape[0]
                ] = approx.P_
            else:
                out = self._mp_func(self._T, m, ignore_trivial=True, p=self._p)
                self._PAN[
                    self._bfs_indices[self._n_processed], : out[:, 0].shape[0]
                ] = out[:, 0]
            self._n_processed += 1

    def pan(self, threshold=0.2, normalize=True, contrast=True, binary=True, clip=True):
        """
        Generate a transformed (i.e., normalized, contrasted, binarized, and repeated)
        pan matrix profile

        Parameters
        ----------
        threshold : float, default 0.2
            The distance `threshold` in which to center the pan matrix profile around
            for best contrast and this value is also used for binarizing the pan matrix
            profile

        normalize : bool, default True
            A flag for whether or not each individual matrix profile within the pan
            matrix profile is normalized by its corresponding subsequence window size.
            If set to `True`, normalization is performed.

        contrast : bool, default True
            A flag for whether or not the pan matrix profile is centered around the
            desired `threshold` in order to provide higher contrast. If set to `True`,
            centering is performed.

        binary : bool, default True
            A flag for whether or not the pan matrix profile is binarized. If set to
            `True`, all values less than or equal to `threshold` are set to `0.0` while
            all other values are set to `1.0`.

        clip : bool, default True
            A flag for whether or not the pan matrix profile is clipped. If set to
            `True`, all values are ensured to be clipped between `0.0` and `1.0`.

        Returns
        -------
        None
        """
        PAN = self._PAN.copy()
        # Retrieve the row indices where the matrix profile was actually computed
        idx = self._bfs_indices[: self._n_processed]
        sorted_idx = np.sort(idx)
        PAN[PAN == np.inf] = np.nan

        if normalize:
            _normalize_pan(
                PAN,
                self._M,
                self._bfs_indices,
                self._n_processed,
                self._T_min,
                self._T_max,
                self._p,
            )
        if contrast:
            core._contrast_pan(PAN, threshold, self._bfs_indices, self._n_processed)
        if binary:
            core._binarize_pan(PAN, threshold, self._bfs_indices, self._n_processed)
        if clip:
            PAN[idx] = np.clip(PAN[idx], 0.0, 1.0)

        # Below, for each matrix profile that was computed, we take that matrix profile
        # and copy/repeat it downwards to replace other rows in the `PAN` where the
        # matrix profile has yet to be computed. Instead of only having lines/values in
        # the rows where matrix profiles were computed, this gives us the "blocky" look
        nrepeat = np.diff(np.append(-1, sorted_idx))
        PAN[: np.sum(nrepeat)] = np.repeat(PAN[sorted_idx], nrepeat, axis=0)
        PAN[np.isnan(PAN)] = np.nanmax(PAN)

        return PAN

    @property
    def PAN_(self):
        """
        Get the transformed (i.e., normalized, contrasted, binarized, and repeated) pan
        matrix profile
        """
        return self.pan().astype(np.float64)

    @property
    def M_(self):
        """
        Get all of the (breadth first searched (level) ordered) subsequence window sizes
        """
        return self._M.astype(np.int64)

    # @property
    # def bfs_indices_(self):
    #     """
    #     Get the breadth first search (level order) indices
    #     """
    #     return self._bfs_indices.astype(np.int64)

    # @property
    # def n_processed_(self):  # pragma: no cover
    #     """
    #     Get the total number of windows that have been processed
    #     """
    #     return self._n_processed


class aamp_stimp(_aamp_stimp):
    """
    Compute the Pan Matrix Profile

    This is based on the SKIMP algorithm.

    Parameters
    ----------
    T : numpy.ndarray
        The time series or sequence for which to compute the pan matrix profile

    m_start : int, default 3
        The starting (or minimum) subsequence window size for which a matrix profile
        may be computed

    m_stop : int, default None
        The stopping (or maximum) subsequence window size for which a matrix profile
        may be computed. When `m_stop = Non`, this is set to the maximum allowable
        subsequence window size

    m_step : int, default 1
        The step between subsequence window sizes

    percentage : float, default 0.01
        The percentage of the full matrix profile to compute for each subsequence
        window size. When `percentage < 1.0`, then the `scraamp` algorithm is used.
        Otherwise, the `stump` algorithm is used when the exact matrix profile is
        requested.

    pre_scraamp : bool, default True
        A flag for whether or not to perform the PreSCRIMP calculation prior to
        computing SCRIMP. If set to `True`, this is equivalent to computing
        SCRIMP++. This parameter is ignored when `percentage = 1.0`.

    p : float, default 2.0
        The p-norm to apply for computing the Minkowski distance.

    Attributes
    ----------
    PAN_ : numpy.ndarray
        The transformed (i.e., normalized, contrasted, binarized, and repeated)
        pan matrix profile

    M_ : numpy.ndarray
        The full list of (breadth first search (level) ordered) subsequence window
        sizes

    Methods
    -------
    update():
        Compute the next matrix profile using the next available (breadth-first-search
        (level) ordered) subsequence window size and update the pan matrix profile

    Notes
    -----
    `DOI: 10.1109/ICBK.2019.00031 \
    <https://www.cs.ucr.edu/~eamonn/PAN_SKIMP%20%28Matrix%20Profile%20XX%29.pdf>`__

    See Table 2
    """

    def __init__(
        self,
        T,
        min_m=3,
        max_m=None,
        step=1,
        percentage=0.01,
        pre_scraamp=True,
        p=2.0,
    ):
        """
        Initialize the `stimp` object and compute the Pan Matrix Profile

        Parameters
        ----------
        T : numpy.ndarray
            The time series or sequence for which to compute the pan matrix profile

        min_m : int, default 3
            The minimum subsequence window size to consider computing a matrix profile
            for

        max_m : int, default None
            The maximum subsequence window size to consider computing a matrix profile
            for. When `max_m = None`, this is set to the maximum allowable subsequence
            window size

        step : int, default 1
            The step between subsequence window sizes

        percentage : float, default 0.01
            The percentage of the full matrix profile to compute for each subsequence
            window size. When `percentage < 1.0`, then the `scraamp` algorithm is used.
            Otherwise, the `stump` algorithm is used when the exact matrix profile is
            requested.

        pre_scraamp : bool, default True
            A flag for whether or not to perform the PreSCRIMP calculation prior to
            computing SCRIMP. If set to `True`, this is equivalent to computing
            SCRIMP++. This parameter is ignored when `percentage = 1.0`.

        p : float, default 2.0
            The p-norm to apply for computing the Minkowski distance.
        """
        super().__init__(
            T,
            min_m=min_m,
            max_m=max_m,
            step=step,
            percentage=percentage,
            pre_scraamp=pre_scraamp,
            mp_func=aamp,
            p=2.0,
        )


class aamp_stimped(_aamp_stimp):
    """
    Compute the Pan Matrix Profile with a distributed dask cluster

    This is based on the SKIMP algorithm.

    Parameters
    ----------
    dask_client : client
            A Dask Distributed client that is connected to a Dask scheduler and
            Dask workers. Setting up a Dask distributed cluster is beyond the
            scope of this library. Please refer to the Dask Distributed
            documentation.

    T : numpy.ndarray
        The time series or sequence for which to compute the pan matrix profile

    m_start : int, default 3
        The starting (or minimum) subsequence window size for which a matrix profile
        may be computed

    m_stop : int, default None
        The stopping (or maximum) subsequence window size for which a matrix profile
        may be computed. When `m_stop = Non`, this is set to the maximum allowable
        subsequence window size

    m_step : int, default 1
        The step between subsequence window sizes

    p : float, default 2.0
        The p-norm to apply for computing the Minkowski distance.

    Attributes
    ----------
    PAN_ : numpy.ndarray
        The transformed (i.e., normalized, contrasted, binarized, and repeated)
        pan matrix profile

    M_ : numpy.ndarray
        The full list of (breadth first search (level) ordered) subsequence window
        sizes

    Methods
    -------
    update():
        Compute the next matrix profile using the next available (breadth-first-search
        (level) ordered) subsequence window size and update the pan matrix profile

    Notes
    -----
    `DOI: 10.1109/ICBK.2019.00031 \
    <https://www.cs.ucr.edu/~eamonn/PAN_SKIMP%20%28Matrix%20Profile%20XX%29.pdf>`__

    See Table 2
    """

    def __init__(
        self,
        dask_client,
        T,
        min_m=3,
        max_m=None,
        step=1,
        p=2.0,
    ):
        """
        Initialize the `stimp` object and compute the Pan Matrix Profile

        Parameters
        ----------
        dask_client : client
            A Dask Distributed client that is connected to a Dask scheduler and
            Dask workers. Setting up a Dask distributed cluster is beyond the
            scope of this library. Please refer to the Dask Distributed
            documentation.

        T : numpy.ndarray
            The time series or sequence for which to compute the pan matrix profile

        min_m : int, default 3
            The minimum subsequence window size to consider computing a matrix profile
            for

        max_m : int, default None
            The maximum subsequence window size to consider computing a matrix profile
            for. When `max_m = None`, this is set to the maximum allowable subsequence
            window size

        step : int, default 1
            The step between subsequence window sizes
        """
        super().__init__(
            T,
            min_m=min_m,
            max_m=max_m,
            step=step,
            percentage=1.0,
            pre_scraamp=False,
            dask_client=dask_client,
            mp_func=aamped,
            p=2.0,
        )
