import numpy as np
from . import core, stump

def ostinato(tss, m):
    """
    Find the consensus motif of multiple time series

    Parameters
    ----------
    tss : list
        List of time series for which to find the consensus motif

    m : int
        Window size

    Returns
    -------
    bsf_rad : float
        Radius of the consensus motif

    ts_ind : int
        Index of time series which contains the consensus motif

    ss_ind : int
        Start index of consensus motif within the time series ts_ind
        that contains it

    Notes
    -----
    <https://www.cs.ucr.edu/~eamonn/consensus_Motif_ICDM_Long_version.pdf>

    See Table 2
    """
    # Preprocess means and stddevs and handle np.nan/np.inf
    Ts = [None] * len(tss)
    M_Ts = [None] * len(tss)
    Σ_Ts = [None] * len(tss)
    for i, T in enumerate(tss):
        Ts[i], M_Ts[i], Σ_Ts[i] = core.preprocess(T, m)

    bsf_rad, ts_ind, ss_ind = np.inf, 0, 0
    k = len(Ts)
    for j in range(k):
        if j < (k - 1):
            h = j + 1
        else:
            h = 0

        mp = stump(Ts[j], m, Ts[h], ignore_trivial=False)
        si = np.argsort(mp[:, 0])
        for q in si:
            rad = mp[q, 0]
            if rad >= bsf_rad:
                break
            for i in range(k):
                if ~np.isin(i, [j, h]):
                    QT = core.sliding_dot_product(Ts[j][q:q+m], Ts[i])
                    rad = np.max((
                            rad,
                            np.min(
                                core._mass(
                                    Ts[j][q:q+m],
                                    Ts[i],
                                    QT,
                                    M_Ts[j][q],
                                    Σ_Ts[j][q],
                                    M_Ts[i],
                                    Σ_Ts[i],))
                                    ))
                    if rad >= bsf_rad:
                        break
            if rad < bsf_rad:
                bsf_rad, ts_ind, ss_ind = rad, j, q

    return bsf_rad, ts_ind, ss_ind
