# STUMPY
# Copyright 2019 TD Ameritrade. Released under the terms of the 3-Clause BSD license.  # noqa: E501
# STUMPY is a trademark of TD Ameritrade IP Company, Inc. All rights reserved.

from collections import deque

import numpy as np


def atsc(IL, IR, j):
    """
    Compute the anchored time series chain (ATSC)

    Parameters
    ----------
    IL : ndarray
        Left matrix profile indices

    IR : ndarray
        Right matrix profile indices

    j : int
        The index value for which to compute the ATSC

    Returns
    -------
    output : ndarray
        Anchored time series chain for index, `j`

    Notes
    -----
    `DOI: 10.1109/ICDM.2017.79 <https://www.cs.ucr.edu/~eamonn/chains_ICDM.pdf>`__

    See Table I

    This is the implementation for the anchored time series chains (ATSC).

    Unlike the original paper, we've replaced the while-loop with a more stable
    for-loop.
    """
    C = deque([j])
    for i in range(IL.size):
        if IR[j] == -1 or IL[IR[j]] != j:
            break
        else:
            j = IR[j]
            C.append(j)

    return np.array(list(C), dtype=np.int64)


def allc(IL, IR):
    """
    Compute the all-chain set (ALLC)

    Parameters
    ----------
    IL : ndarray
        Left matrix profile indices

    IR : ndarray
        Right matrix profile indices

    Returns
    -------
    S : list(ndarray)
        All-chain set

    C : ndarray
        Anchored time series chain for the longest chain (also known as the
        unanchored chain)

    Notes
    -----
    `DOI: 10.1109/ICDM.2017.79 <https://www.cs.ucr.edu/~eamonn/chains_ICDM.pdf>`__

    See Table II

    Unlike the original paper, we've replaced the while-loop with a more stable
    for-loop.

    This is the implementation for the all-chain set (ALLC) and the unanchored
    chain is simply the longest one among the all-chain set. Both the
    all-chain set and unanchored chain are returned.

    The all-chain set, S, is returned as a list of unique numpy arrays.
    """
    L = np.ones(IL.size, dtype=np.int64)
    S = set()  # type: ignore
    for i in range(IL.size):
        if L[i] == 1:
            j = i
            C = deque([j])
            for k in range(IL.size):
                if IR[j] == -1 or IL[IR[j]] != j:
                    break
                else:
                    j = IR[j]
                    L[j] = -1
                    L[i] = L[i] + 1
                    C.append(j)
            S.update([tuple(C)])
    C = atsc(IL, IR, L.argmax())
    S = [np.array(s, dtype=np.int64) for s in S]  # type: ignore

    return S, C  # type: ignore
