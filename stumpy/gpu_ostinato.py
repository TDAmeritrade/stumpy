# STUMPY
# Copyright 2019 TD Ameritrade. Released under the terms of the 3-Clause BSD license.
# STUMPY is a trademark of TD Ameritrade IP Company, Inc. All rights reserved.

from . import core
from .gpu_aamp_ostinato import gpu_aamp_ostinato
from .gpu_stump import gpu_stump
from .ostinato import _get_central_motif, _ostinato


@core.non_normalized(
    gpu_aamp_ostinato,
    exclude=["normalize", "p", "Ts_subseq_isconstant"],
)
def gpu_ostinato(Ts, m, device_id=0, normalize=True, p=2.0, Ts_subseq_isconstant=None):
    """
    Find the z-normalized consensus motif of multiple time series with one or more GPU
    devices

    This is a wrapper around the vanilla version of the ostinato algorithm
    which finds the best radius and a helper function that finds the most
    central conserved motif.

    Parameters
    ----------
    Ts : list
        A list of time series for which to find the most central consensus motif.

    m : int
        Window size.

    device_id : int or list, default 0
        The (GPU) device number to use. The default value is ``0``. A list of
        valid device ids (``int``) may also be provided for parallel GPU-STUMP
        computation. A list of all valid device ids can be obtained by
        executing ``[device.id for device in numba.cuda.list_devices()]``.

    normalize : bool, default True
        When set to ``True``, this z-normalizes subsequences prior to computing
        distances. Otherwise, this function gets re-routed to its complementary
        non-normalized equivalent set in the ``@core.non_normalized`` function
        decorator.

    p : float, default 2.0
        The p-norm to apply for computing the Minkowski distance. Minkowski distance is
        typically used with ``p`` being ``1`` or ``2``, which correspond to the
        Manhattan distance and the Euclidean distance, respectively. This parameter is
        ignored when ``normalize == True``.

    Ts_subseq_isconstant : list, default None
        A list of rolling window isconstant for each time series in ``Ts``.

    Returns
    -------
    central_radius : float
        Radius of the most central consensus motif.

    central_Ts_idx : int
        The time series index in ``Ts`` that contains the most central consensus motif.

    central_subseq_idx : int
        The subsequence index within time series ``Ts[central_motif_Ts_idx]`` that
        contains the most central consensus motif.

    See Also
    --------
    stumpy.ostinato : Find the z-normalized consensus motif of multiple time series
    stumpy.ostinatoed : Find the z-normalized consensus motif of multiple time series
        with a ``dask``/``ray`` cluster

    Notes
    -----
    `DOI: 10.1109/ICDM.2019.00140 \
    <https://www.cs.ucr.edu/~eamonn/consensus_Motif_ICDM_Long_version.pdf>`__

    See Table 2

    The ostinato algorithm proposed in the paper finds the best radius
    in ``Ts``. Intuitively, the radius is the minimum distance of a
    subsequence to encompass at least one nearest neighbor subsequence
    from all other time series. The best radius in ``Ts`` is the minimum
    radius amongst all radii. Some data sets might contain multiple
    subsequences which have the same optimal radius.
    The greedy Ostinato algorithm only finds one of them, which might
    not be the most central motif. The most central motif amongst the
    subsequences with the best radius is the one with the smallest mean
    distance to nearest neighbors in all other time series. To find this
    central motif it is necessary to search the subsequences with the
    best radius via ``stumpy.ostinato._get_central_motif``.

    Examples
    --------
    >>> import stumpy
    >>> import numpy as np
    >>> from numba import cuda
    >>> if __name__ == "__main__":
    ...     all_gpu_devices = [device.id for device in cuda.list_devices()]
    ...     stumpy.gpu_ostinato(
    ...         [np.array([584., -11., 23., 79., 1001., 0., 19.]),
    ...          np.array([600., -10., 23., 17.]),
    ...          np.array([  1.,   9.,  6.,  0.])],
    ...         m=3,
    ...         device_id=all_gpu_devices)
    (1.2370237678153826, 0, 4)
    """
    if not isinstance(Ts, list):  # pragma: no cover
        raise ValueError(f"`Ts` is of type `{type(Ts)}` but a `list` is expected")

    if Ts_subseq_isconstant is None:
        Ts_subseq_isconstant = [None] * len(Ts)

    Ts_copy = [None] * len(Ts)
    M_Ts = [None] * len(Ts)
    Σ_Ts = [None] * len(Ts)
    for i, T in enumerate(Ts):
        Ts_copy[i], M_Ts[i], Σ_Ts[i], Ts_subseq_isconstant[i] = core.preprocess(
            T, m, copy=True, T_subseq_isconstant=Ts_subseq_isconstant[i]
        )

    bsf_radius, bsf_Ts_idx, bsf_subseq_idx = _ostinato(
        Ts_copy,
        m,
        M_Ts,
        Σ_Ts,
        Ts_subseq_isconstant,
        device_id=device_id,
        mp_func=gpu_stump,
    )

    (
        central_radius,
        central_Ts_idx,
        central_subseq_idx,
    ) = _get_central_motif(
        Ts_copy,
        bsf_radius,
        bsf_Ts_idx,
        bsf_subseq_idx,
        m,
        M_Ts,
        Σ_Ts,
        Ts_subseq_isconstant,
    )

    return central_radius, central_Ts_idx, central_subseq_idx
