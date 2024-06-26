# STUMPY
# Copyright 2019 TD Ameritrade. Released under the terms of the 3-Clause BSD license.
# STUMPY is a trademark of TD Ameritrade IP Company, Inc. All rights reserved.

# import numpy as np
# import math

import functools

from . import core
from .gpu_aampdist import gpu_aampdist
from .gpu_stump import gpu_stump


@core.non_normalized(gpu_aampdist)
def gpu_mpdist(
    T_A,
    T_B,
    m,
    percentage=0.05,
    k=None,
    device_id=0,
    normalize=True,
    p=2.0,
    T_A_subseq_isconstant=None,
    T_B_subseq_isconstant=None,
):
    """
    Compute the z-normalized matrix profile distance (MPdist) measure between any two
    time series with one or more GPU devices

    The MPdist distance measure considers two time series to be similar if they share
    many subsequences, regardless of the order of matching subsequences. MPdist
    concatenates and sorts the output of an AB-join and a BA-join and returns the value
    of the ``k``-th smallest number as the reported distance. Note that MPdist is a
    measure and not a metric. Therefore, it does not obey the triangular inequality but
    the method is highly scalable.

    Parameters
    ----------
    T_A : numpy.ndarray
        The first time series or sequence for which to compute the matrix profile.

    T_B : numpy.ndarray
        The second time series or sequence for which to compute the matrix profile.

    m : int
        Window size.

    percentage : float, default 0.05
        The percentage of distances that will be used to report ``mpdist``. The value
        is between ``0.0`` and ``1.0``. This parameter is ignored when ``k`` is not
        ``None``.

    k : int, default None
        Specify the ``k``-th value in the concatenated matrix profiles to return. When
        ``k`` is not ``None``, then the `percentage` parameter is ignored.

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

    T_A_subseq_isconstant : numpy.ndarray or function, default None
        A boolean array that indicates whether a subsequence in ``T_A`` is constant
        (``True``). Alternatively, a custom, user-defined function that returns a
        boolean array that indicates whether a subsequence in ``T_A`` is constant
        (``True``). The function must only take two arguments, ``a``, a 1-D array,
        and ``w``, the window size, while additional arguments may be specified
        by currying the user-defined function using ``functools.partial``. Any
        subsequence with at least one ``np.nan``/``np.inf`` will automatically have its
        corresponding value set to ``False`` in this boolean array.

    T_B_subseq_isconstant : numpy.ndarray or function, default None
        A boolean array that indicates whether a subsequence in `T_B` is constant
        (``True``). Alternatively, a custom, user-defined function that returns a
        boolean array that indicates whether a subsequence in ``T_B`` is constant
        (``True``). The function must only take two arguments, ``a``, a 1-D array,
        and ``w``, the window size, while additional arguments may be specified
        by currying the user-defined function using ``functools.partial``. Any
        subsequence with at least one ``np.nan``/``np.inf`` will automatically have its
        corresponding value set to ``False`` in this boolean array.

    Returns
    -------
    MPdist : float
        The matrix profile distance.

    Notes
    -----
    `DOI: 10.1109/ICDM.2018.00119 \
    <https://www.cs.ucr.edu/~eamonn/MPdist_Expanded.pdf>`__

    See Section III

    Examples
    --------
    >>> import stumpy
    >>> import numpy as np
    >>> from numba import cuda
    >>> if __name__ == "__main__":
    ...     all_gpu_devices = [device.id for device in cuda.list_devices()]
    ...     stumpy.gpu_mpdist(
    ...         np.array([-11.1, 23.4, 79.5, 1001.0]),
    ...         np.array([584., -11., 23., 79., 1001., 0., -19.]),
    ...         m=3,
    ...         device_id=all_gpu_devices)
    0.00019935236191097894
    """
    partial_gpu_stump = functools.partial(
        gpu_stump,
        T_A_subseq_isconstant=T_A_subseq_isconstant,
        T_B_subseq_isconstant=T_B_subseq_isconstant,
    )
    MPdist = core._mpdist(
        T_A, T_B, m, partial_gpu_stump, percentage, k, device_id=device_id
    )

    return MPdist
