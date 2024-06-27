# STUMPY
# Copyright 2019 TD Ameritrade. Released under the terms of the 3-Clause BSD license.  # noqa: E501
# STUMPY is a trademark of TD Ameritrade IP Company, Inc. All rights reserved.

from collections import deque

import numpy as np


def atsc(IL, IR, j):
    """
    Compute the anchored time series chain (ATSC)

    Note that since the matrix profile indices, ``IL`` and ``IR``, are pre-computed,
    this function is agnostic to subsequence normalization.

    Parameters
    ----------
    IL : numpy.ndarray
        Left matrix profile indices.

    IR : numpy.ndarray
        Right matrix profile indices.

    j : int
        The index value for which to compute the ATSC.

    Returns
    -------
    out : numpy.ndarray
        Anchored time series chain for index, ``j``

    See Also
    --------
    stumpy.allc : Compute the all-chain set (ALLC)

    Notes
    -----
    `DOI: 10.1109/ICDM.2017.79 <https://www.cs.ucr.edu/~eamonn/chains_ICDM.pdf>`__

    See Table I

    This is the implementation for the anchored time series chains (ATSC).

    Unlike the original paper, we've replaced the while-loop with a more stable
    for-loop.

    Examples
    --------
    >>> import stumpy
    >>> import numpy as np
    >>> mp = stumpy.stump(np.array([584., -11., 23., 79., 1001., 0., -19.]), m=3)
    >>> stumpy.atsc(mp[:, 2], mp[:, 3], 1)
    array([1, 3])

    >>> # Alternative example using named attributes
    >>>
    >>> mp = stumpy.stump(np.array([584., -11., 23., 79., 1001., 0., -19.]), m=3)
    >>> stumpy.atsc(mp.left_I_, mp.right_I_, 1)
    array([1, 3])
    """
    C = deque([j])
    for i in range(IL.size):
        if IR[j] == -1 or IL[IR[j]] != j:
            break
        else:
            j = IR[j]
            C.append(j)

    out = np.array(list(C), dtype=np.int64)

    return out


def allc(IL, IR):
    """
    Compute the all-chain set (ALLC)

    Note that since the matrix profile indices, ``IL`` and ``IR``, are pre-computed,
    this function is agnostic to subsequence normalization.

    Parameters
    ----------
    IL : numpy.ndarray
        Left matrix profile indices.

    IR : numpy.ndarray
        Right matrix profile indices.

    Returns
    -------
    S : list(numpy.ndarray)
        All-chain set.

    C : numpy.ndarray
        Anchored time series chain for the longest chain (also known as the unanchored
        chain). Note that when there are multiple different chains with length equal to
        ``len(C)``, then only one chain from this set is returned. You may iterate over
        the all-chain set, ``S``, to find all other possible chains with length
        ``len(C)``.

    See Also
    --------
    stumpy.atsc : Compute the anchored time series chain (ATSC)

    Notes
    -----
    `DOI: 10.1109/ICDM.2017.79 <https://www.cs.ucr.edu/~eamonn/chains_ICDM.pdf>`__

    See Table II

    Unlike the original paper, we've replaced the while-loop with a more stable
    for-loop.

    This is the implementation for the all-chain set (ALLC) and the unanchored
    chain is simply the longest one among the all-chain set. Both the
    all-chain set and unanchored chain are returned.

    The all-chain set, ``S``, is returned as a list of unique numpy arrays.

    Examples
    --------
    >>> import stumpy
    >>> import numpy as np
    >>> mp = stumpy.stump(np.array([584., -11., 23., 79., 1001., 0., -19.]), m=3)
    >>> stumpy.allc(mp[:, 2], mp[:, 3])
    ([array([1, 3]), array([2]), array([0, 4])], array([0, 4]))

    >>> # Alternative example using named attributes
    >>>
    >>> mp = stumpy.stump(np.array([584., -11., 23., 79., 1001., 0., -19.]), m=3)
    >>> stumpy.allc(mp.left_I_, mp.right_I_)
    ([array([1, 3]), array([2]), array([0, 4])], array([0, 4]))
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
