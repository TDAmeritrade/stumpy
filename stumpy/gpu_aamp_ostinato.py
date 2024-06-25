# STUMPY
# Copyright 2019 TD Ameritrade. Released under the terms of the 3-Clause BSD license.
# STUMPY is a trademark of TD Ameritrade IP Company, Inc. All rights reserved.

from . import core, gpu_aamp
from .aamp_ostinato import _aamp_ostinato, _get_aamp_central_motif


def gpu_aamp_ostinato(Ts, m, device_id=0, p=2.0):
    """
    Find the non-normalized (i.e., without z-normalization) consensus motif of multiple
    time series with one or more GPU devices

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

    p : float, default 2.0
        The p-norm to apply for computing the Minkowski distance. Minkowski distance is
        typically used with `p` being 1 or 2, which correspond to the Manhattan distance
        and the Euclidean distance, respectively.

    Returns
    -------
    central_radius : float
        Radius of the most central consensus motif

    central_Ts_idx : int
        The time series index in `Ts` that contains the most central consensus motif

    central_subseq_idx : int
        The subsequence index within time series `Ts[central_motif_Ts_idx]` that
        contains the most central consensus motif

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
    if not isinstance(Ts, list):  # pragma: no cover
        raise ValueError(f"`Ts` is of type `{type(Ts)}` but a `list` is expected")

    Ts_copy = [None] * len(Ts)
    Ts_subseq_isfinite = [None] * len(Ts)
    for i, T in enumerate(Ts):
        (
            Ts_copy[i],
            Ts_subseq_isfinite[i],
        ) = core.preprocess_non_normalized(T, m, copy=True)

    bsf_radius, bsf_Ts_idx, bsf_subseq_idx = _aamp_ostinato(
        Ts_copy,
        m,
        Ts_subseq_isfinite,
        p=p,
        device_id=device_id,
        mp_func=gpu_aamp,
    )

    (
        central_radius,
        central_Ts_idx,
        central_subseq_idx,
    ) = _get_aamp_central_motif(
        Ts_copy,
        bsf_radius,
        bsf_Ts_idx,
        bsf_subseq_idx,
        m,
        Ts_subseq_isfinite,
        p=p,
    )

    return central_radius, central_Ts_idx, central_subseq_idx
