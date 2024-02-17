# STUMPY
# Copyright 2019 TD Ameritrade. Released under the terms of the 3-Clause BSD license.
# STUMPY is a trademark of TD Ameritrade IP Company, Inc. All rights reserved.

from . import gpu_aamp
from .aamp_stimp import _aamp_stimp


class gpu_aamp_stimp(_aamp_stimp):
    """
    Compute the Pan Matrix Profile with with one or more GPU devices

    This is based on the SKIMP algorithm.

    Parameters
    ----------
    T : numpy.ndarray
        The time series or sequence for which to compute the pan matrix profile

    m_start : int, default 3
        The starting (or minimum) subsequence window size for which a matrix profile
        may be computed

    m_stop : int, default None
        The stopping (or maximum) subsequence window size for which a matrix profile
        may be computed. When `m_stop = None`, this is set to the maximum allowable
        subsequence window size

    m_step : int, default 1
        The step between subsequence window sizes

    device_id : int or list, default 0
        The (GPU) device number to use. The default value is `0`. A list of
        valid device ids (int) may also be provided for parallel GPU-STUMP
        computation. A list of all valid device ids can be obtained by
        executing `[device.id for device in numba.cuda.list_devices()]`.

    p : float, default 2.0
        The p-norm to apply for computing the Minkowski distance. Minkowski distance is
        typically used with `p` being 1 or 2, which correspond to the Manhattan distance
        and the Euclidean distance, respectively.

    Attributes
    ----------
    PAN_ : numpy.ndarray
        The transformed (i.e., normalized, contrasted, binarized, and repeated)
        pan matrix profile

    M_ : numpy.ndarray
        The full list of (breadth first search (level) ordered) subsequence window
        sizes

    Methods
    -------
    update():
        Compute the next matrix profile using the next available (breadth-first-search
        (level) ordered) subsequence window size and update the pan matrix profile

    -----
    `DOI: 10.1109/ICBK.2019.00031 \
    <https://www.cs.ucr.edu/~eamonn/PAN_SKIMP%20%28Matrix%20Profile%20XX%29.pdf>`__

    See Table 2
    """

    def __init__(
        self,
        T,
        min_m=3,
        max_m=None,
        step=1,
        device_id=0,
        p=2.0,
    ):
        """
        Initialize the `stimp` object and compute the Pan Matrix Profile

        Parameters
        ----------
        T : numpy.ndarray
            The time series or sequence for which to compute the pan matrix profile

        min_m : int, default 3
            The minimum subsequence window size to consider computing a matrix profile
            for

        max_m : int, default None
            The maximum subsequence window size to consider computing a matrix profile
            for. When `max_m = None`, this is set to the maximum allowable subsequence
            window size

        step : int, default 1
            The step between subsequence window sizes

        device_id : int or list, default 0
            The (GPU) device number to use. The default value is `0`. A list of
            valid device ids (int) may also be provided for parallel GPU-STUMP
            computation. A list of all valid device ids can be obtained by
            executing `[device.id for device in numba.cuda.list_devices()]`.

        p : float, default 2.0
            The p-norm to apply for computing the Minkowski distance. Minkowski distance
            is typically used with `p` being 1 or 2, which correspond to the Manhattan
            distance and the Euclidean distance, respectively.
        """
        super().__init__(
            T,
            min_m=min_m,
            max_m=max_m,
            step=step,
            percentage=1.0,
            pre_scraamp=False,
            device_id=device_id,
            mp_func=gpu_aamp,
            p=2.0,
        )
