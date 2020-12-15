# STUMPY
# Copyright 2019 TD Ameritrade. Released under the terms of the 3-Clause BSD license.
# STUMPY is a trademark of TD Ameritrade IP Company, Inc. All rights reserved.

from . import core, gpu_stump
from .ostinato import _ostinato, _get_central_motif


def gpu_ostinato(Ts, m, device_id=0):
    """
    Find the z-normalized consensus motif of multiple time series with one or more GPU
    devices

    This is a wrapper around the vanilla version of the ostinato algorithm
    which finds the best radius and a helper function that finds the most
    central conserved motif.

    Parameters
    ----------
    Ts : list
        A list of time series for which to find the most central consensus motif

    m : int
        Window size

    device_id : int or list, default 0
        The (GPU) device number to use. The default value is `0`. A list of
        valid device ids (int) may also be provided for parallel GPU-STUMP
        computation. A list of all valid device ids can be obtained by
        executing `[device.id for device in numba.cuda.list_devices()]`.

    Returns
    -------
    central_radius : float
        Radius of the most central consensus motif

    central_Ts_idx : int
        The time series index in `Ts` which contains the most central consensus motif

    central_subseq_idx : int
        The subsequence index within time series `Ts[central_motif_Ts_idx]` the contains
        most central consensus motif

    Notes
    -----
    `DOI: 10.1109/ICDM.2019.00140 \
    <https://www.cs.ucr.edu/~eamonn/consensus_Motif_ICDM_Long_version.pdf>`__

    See Table 2

    The ostinato algorithm proposed in the paper finds the best radius
    in `Ts`. Intuitively, the radius is the minimum distance of a
    subsequence to encompass at least one nearest neighbor subsequence
    from all other time series. The best radius in `Ts` is the minimum
    radius amongst all radii. Some data sets might contain multiple
    subsequences which have the same optimal radius.
    The greedy Ostinato algorithm only finds one of them, which might
    not be the most central motif. The most central motif amongst the
    subsequences with the best radius is the one with the smallest mean
    distance to nearest neighbors in all other time series. To find this
    central motif it is necessary to search the subsequences with the
    best radius via `stumpy.ostinato._get_central_motif`
    """
    M_Ts = [None] * len(Ts)
    Σ_Ts = [None] * len(Ts)
    for i, T in enumerate(Ts):
        Ts[i], M_Ts[i], Σ_Ts[i] = core.preprocess(T, m)

    bsf_radius, bsf_Ts_idx, bsf_subseq_idx = _ostinato(
        Ts, m, M_Ts, Σ_Ts, device_id=device_id, mp_func=gpu_stump
    )

    (
        central_radius,
        central_Ts_idx,
        central_subseq_idx,
    ) = _get_central_motif(Ts, bsf_radius, bsf_Ts_idx, bsf_subseq_idx, m, M_Ts, Σ_Ts)

    return central_radius, central_Ts_idx, central_subseq_idx
