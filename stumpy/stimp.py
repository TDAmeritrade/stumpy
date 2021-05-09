# STUMPY
# Copyright 2019 TD Ameritrade. Released under the terms of the 3-Clause BSD license.
# STUMPY is a trademark of TD Ameritrade IP Company, Inc. All rights reserved.

import numpy as np
from . import core, stump, scrump


def _bfs_indices(n):
    """
    Generate the level order indices from the implicit construction of a binary
    search tree followed by a breadth first (level order) search

    Parameters
    ----------
    n : int
        The number indices to generate the ordered indices for

    Returns
    -------
    level_idx : ndarray
        The breadth first search (level order) indices
    """
    if n == 1:  # pragma: no cover
        return np.array([0], dtype=np.int64)

    nlevel = np.floor(np.log2(n) + 1).astype(np.int64)
    nindices = np.power(2, np.arange(nlevel))
    cumsum_nindices = np.cumsum(nindices)
    nindices[-1] = n - cumsum_nindices[np.searchsorted(cumsum_nindices, n) - 1]

    indices = np.empty((2, nindices.max()), dtype=np.int64)
    indices[0, 0] = 0
    indices[1, 0] = n
    tmp_indices = np.empty((2, 2 * nindices.max()), dtype=np.int64)

    out = np.empty(n, dtype=np.int64)
    out_idx = 0

    for nidx in nindices:
        level_indices = (indices[0, :nidx] + indices[1, :nidx]) // 2

        if out_idx + len(level_indices) < n:
            tmp_indices[0, 0 : 2 * nidx : 2] = indices[0, :nidx]
            tmp_indices[0, 1 : 2 * nidx : 2] = level_indices + 1
            tmp_indices[1, 0 : 2 * nidx : 2] = level_indices
            tmp_indices[1, 1 : 2 * nidx : 2] = indices[1, :nidx]

            mask = tmp_indices[0, : 2 * nidx] < tmp_indices[1, : 2 * nidx]
            mask_sum = np.count_nonzero(mask)
            indices[0, :mask_sum] = tmp_indices[0, : 2 * nidx][mask]
            indices[1, :mask_sum] = tmp_indices[1, : 2 * nidx][mask]

        # for level_idx in level_indices:
        #     yield level_idx

        out[out_idx : out_idx + len(level_indices)] = level_indices
        out_idx += len(level_indices)

    return out


class stimp:
    """
    Compute the Pan Matrix Profile and Pan Matrix Profile Indices

    Parameters
    ----------
    T : ndarray
        The time series or sequence for which to compute the pan matrix profile

    Returns
    -------
    None
    """

    def __init__(
        self,
        T,
        min_m=3,
        max_m=None,
        step=1,
        percentage=0.01,
        pre_scrump=True,
        normalize=True,
    ):
        """
        Initialize the `stimp` object

        Parameters
        ----------
        T : ndarray
            The time series or sequence for which to compute the pan matrix profile

        Returns
        -------
        None
        """
        self._T = T
        if max_m is None:
            max_m = max(min_m + 1, core.get_max_window_size(self._T.shape[0]))
            M = np.arange(min_m, max_m + 1, step)
        else:
            min_m, max_m = sorted([min_m, max_m])
            M = np.arange(
                max(3, min_m),
                min(core.get_max_window_size(self._T.shape[0]), max_m) + 1,
                step,
            )
        self._bfs_indices = _bfs_indices(M.shape[0])
        self._M = M[self._bfs_indices]
        self._idx = 0
        percentage = min(1.0, percentage)
        percentage = max(0.0, percentage)
        self._percentage = percentage
        self._pre_scrump = pre_scrump
        self._normalize = normalize

        self._P = np.full((self._M.shape[0], self._T.shape[0]), fill_value=np.inf)
        self._I = np.ones((self._M.shape[0], self._T.shape[0]), dtype=np.int64) * -1

    def update(self):
        """
        Update the pan matrix profile and the pan matrix profile indices by computing
        additional new matrix profiles
        """
        if self._idx < self._M.shape[0]:
            m = self._M[self._idx]
            if self._percentage < 1.0:
                approx = scrump(
                    self._T,
                    m,
                    ignore_trivial=True,
                    percentage=self._percentage,
                    pre_scrump=self._pre_scrump,
                    normalize=self._normalize,
                )
                approx.update()
                self._P[self._bfs_indices[self._idx], : approx.P_.shape[0]] = approx.P_
                self._I[self._bfs_indices[self._idx], : approx.I_.shape[0]] = approx.I_
            else:
                out = stump(self._T, m, ignore_trivial=True, normalize=self._normalize)
                self._P[self._bfs_indices[self._idx], : out[:, 0].shape[0]] = out[:, 0]
                self._I[self._bfs_indices[self._idx], : out[:, 1].shape[0]] = out[:, 1]
            self._idx += 1

    @property
    def P_(self):
        """
        Get the pan matrix profile
        """
        return self._P.astype(np.float64)

    @property
    def I_(self):
        """
        Get the pan matrix profile indices
        """
        return self._I.astype(np.int64)

    @property
    def M_(self):
        """
        Get all of the (breadth first searched (level) ordered) window sizes
        """
        return self._M.astype(np.int64)

    @property
    def bfs_indices_(self):
        """
        Get the breadth first search (level order) indices
        """
        return self._bfs_indices.astype(np.int64)

    @property
    def n_processed_(self):  # pragma: no cover
        """
        Get the total number of windows that have been processed
        """
        return self._idx
