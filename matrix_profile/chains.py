from collections import deque
import numpy as np

def atsc(IL, IR, j):
    """
    DOI: 10.1109/ICDM.2017.79

    See Table I

    This is the implementation for the anchored time series chains (ATSC).

    Note that we replace the while-loop with a more stable for-loop.
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
    DOI: 10.1109/ICDM.2017.79

    See Table II

    Note that we replace the while-loop with a more stable for-loop.

    This is the implementation for the all-chain set (ALLC) and the unanchored
    chain is simply the longest one among the all-chain set. Both the 
    all-chain set and unanchored chain are returned.

    The all-chain set, S, is returned as a list of unique numpy arrays.
    """
    L = np.ones(IL.size, dtype=np.int64)
    S = set()
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
    S = [np.array(s, dtype=np.int64) for s in S]

    return S, C