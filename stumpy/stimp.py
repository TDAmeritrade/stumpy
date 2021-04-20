# STUMPY
# Copyright 2019 TD Ameritrade. Released under the terms of the 3-Clause BSD license.
# STUMPY is a trademark of TD Ameritrade IP Company, Inc. All rights reserved.

import numpy as np


def _bfs_indices(n):
    """
    Generate the level order indices from the implicit construction of a binary
    search tree followed by a breadth first (level order) search

    Parameters
    ----------
    n : int
        The number indices to generate the ordered indices for

    Yields
    ------
    level_idx : int
        The next breadth first search (level order) index
    """
    nlevel = np.floor(np.log2(n) + 1).astype(int)
    nindices = np.power(2, np.arange(nlevel))
    cumsum_nindices = np.cumsum(nindices)
    nindices[-1] = n - cumsum_nindices[np.searchsorted(cumsum_nindices, n) - 1]

    indices = np.empty((2, nindices.max()), dtype=np.int32)
    indices[0, 0] = 0
    indices[1, 0] = n
    tmp_indices = np.empty((2, 2 * nindices.max()), dtype=np.int32)

    # out = np.empty(n, dtype=np.int32)
    out_idx = 0

    for nidx in nindices:
        level_indices = (indices[0, :nidx] + indices[1, :nidx]) // 2

        # Computing the next slices is not needed for the last step
        if out_idx + len(level_indices) < n:
            tmp_indices[0, 0 : 2 * nidx : 2] = indices[0, :nidx]
            tmp_indices[0, 1 : 2 * nidx : 2] = level_indices + 1
            tmp_indices[1, 0 : 2 * nidx : 2] = level_indices
            tmp_indices[1, 1 : 2 * nidx : 2] = indices[1, :nidx]

            # Discard invalid slices
            mask = tmp_indices[0, : 2 * nidx] < tmp_indices[1, : 2 * nidx]
            mask_sum = np.count_nonzero(mask)
            indices[0, :mask_sum] = tmp_indices[0, : 2 * nidx][mask]
            indices[1, :mask_sum] = tmp_indices[1, : 2 * nidx][mask]

        for level_idx in level_indices:
            yield level_idx

        # Fast appending
        # out[out_idx : out_idx + len(level_indices)] = level_indices
        out_idx += len(level_indices)

        # return out
