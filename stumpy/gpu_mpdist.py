# STUMPY
# Copyright 2019 TD Ameritrade. Released under the terms of the 3-Clause BSD license.
# STUMPY is a trademark of TD Ameritrade IP Company, Inc. All rights reserved.

# import numpy as np
# import math

from . import gpu_stump
from .mpdist import _mpdist


def gpu_mpdist(T_A, T_B, m, percentage=0.05, k=None, device_id=0):
    """
    Compute the z-normalized matrix profile distance (MPdist) measure between any two
    time series with one or more GPU devices

    The MPdist distance measure considers two time series to be similar if they share
    many subsequences, regardless of the order of matching subsequences. MPdist
    concatenates and sorts the output of an AB-join and a BA-join and returns the value
    of the `k`th smallest number as the reported distance. Note that MPdist is a
    measure and not a metric. Therefore, it does not obey the triangular inequality but
    the method is highly scalable.

    Parameters
    ----------
    T_A : ndarray
        The first time series or sequence for which to compute the matrix profile

    T_B : ndarray
        The second time series or sequence for which to compute the matrix profile

    m : int
        Window size

    percentage : float, default 0.05
        The percentage of distances that will be used to report `mpdist`. The value
        is between 0.0 and 1.0. This parameter is ignored when `k` is not `None`.

    k : int, default None
        Specify the `k`th value in the concatenated matrix profiles to return. When `k`
        is not `None`, then the `percentage` parameter is ignored.

    device_id : int or list, default 0
        The (GPU) device number to use. The default value is `0`. A list of
        valid device ids (int) may also be provided for parallel GPU-STUMP
        computation. A list of all valid device ids can be obtained by
        executing `[device.id for device in numba.cuda.list_devices()]`.

    Returns
    -------
    MPdist : float
        The matrix profile distance

    Notes
    -----
    `DOI: 10.1109/ICDM.2018.00119 \
    <https://www.cs.ucr.edu/~eamonn/MPdist_Expanded.pdf>`__

    See Section III
    """
    return _mpdist(T_A, T_B, m, percentage, k, device_id=device_id, mp_func=gpu_stump)
