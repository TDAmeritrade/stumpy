# STUMPY
# Copyright 2019 TD Ameritrade. Released under the terms of the 3-Clause BSD license.
# STUMPY is a trademark of TD Ameritrade IP Company, Inc. All rights reserved.

# import numpy as np
# import math

from . import gpu_stump
from .mpdist import _mpdist


def gpu_mpdist(T_A, T_B, m, percentage=0.05, device_id=0):
    """
    A convenience function for computing the matrix profile distance
    (MPdist) measure between any two time series

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
        is between 0.0 and 1.0.

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
    # percentage = min(percentage, 1.0)
    # percentage = max(percentage, 0.0)
    # n_A = T_A.shape[0]
    # n_B = T_B.shape[0]
    # P_ABBA = np.empty(n_A - m + 1 + n_B - m + 1, dtype=np.float64)
    # k = min(math.ceil(percentage * (n_A + n_B)), n_A - m + 1 + n_B - m + 1 - 1)

    # P_ABBA[: n_A - m + 1] = gpu_stump(
    #     T_A, m, T_B, ignore_trivial=False, device_id=device_id
    # )[:, 0]
    # P_ABBA[n_A - m + 1 :] = gpu_stump(
    #     T_B, m, T_A, ignore_trivial=False, device_id=device_id
    # )[:, 0]

    # P_ABBA.sort()
    # MPdist = P_ABBA[k]
    # if ~np.isfinite(MPdist):  # pragma: no cover
    #     k = np.count_nonzero(np.isfinite(P_ABBA[:k])) - 1
    #     MPdist = P_ABBA[k]

    # return MPdist
    return _mpdist(T_A, T_B, m, percentage, device_id=device_id, custom_func=gpu_stump)
