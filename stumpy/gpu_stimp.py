# STUMPY
# Copyright 2019 TD Ameritrade. Released under the terms of the 3-Clause BSD license.
# STUMPY is a trademark of TD Ameritrade IP Company, Inc. All rights reserved.

from . import core
from .gpu_aamp_stimp import gpu_aamp_stimp
from .gpu_stump import gpu_stump
from .stimp import _stimp


@core.non_normalized(
    gpu_aamp_stimp,
    exclude=["pre_scrump", "normalize", "p", "pre_scraamp", "T_subseq_isconstant_func"],
    replace={"pre_scrump": "pre_scraamp"},
)
class gpu_stimp(_stimp):
    """
    A class to compute the Pan Matrix Profile with with one or more GPU devices

    This is based on the SKIMP algorithm.

    Parameters
    ----------
    T : numpy.ndarray
        The time series or sequence for which to compute the pan matrix profile.

    min_m : int, default 3
        The starting (or minimum) subsequence window size for which a matrix profile
        may be computed.

    max_m : int, default None
        The stopping (or maximum) subsequence window size for which a matrix profile
        may be computed. When ``m_stop = None``, this is set to the maximum allowable
        subsequence window size.

    step : int, default 1
        The step between subsequence window sizes.

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

    T_subseq_isconstant_func : function, default None
        A custom, user-defined function that returns a boolean array that indicates
        whether a subsequence in ``T`` is constant (``True``). The function must
        only take two arguments, ``a``, a 1-D array, and ``w``, the window size,
        while additional arguments may be specified by currying the user-defined
        function using ``functools.partial``. Any subsequence with at least one
        ``np.nan``/``np.inf`` will automatically have its corresponding value set to
        ``False`` in this boolean array.

    Attributes
    ----------
    PAN_ : numpy.ndarray
        The transformed (i.e., normalized, contrasted, binarized, and repeated)
        pan matrix profile.

    M_ : numpy.ndarray
        The full list of (breadth first search (level) ordered) subsequence window
        sizes.

    Methods
    -------
    update():
        Compute the next matrix profile using the next available (breadth-first-search
        (level) ordered) subsequence window size and update the pan matrix profile.

    See Also
    --------
    stumpy.stimp : Compute the Pan Matrix Profile
    stumpy.stimped : Compute the Pan Matrix Profile with a ``dask``/``ray`` cluster

    Notes
    -----
    `DOI: 10.1109/ICBK.2019.00031 \
    <https://www.cs.ucr.edu/~eamonn/PAN_SKIMP%20%28Matrix%20Profile%20XX%29.pdf>`__

    See Table 2

    Examples
    --------
    >>> import stumpy
    >>> import numpy as np
    >>> from numba import cuda
    >>> if __name__ == "__main__":
    ...     all_gpu_devices = [device.id for device in cuda.list_devices()]
    ...     pmp = stumpy.gpu_stimp(
    ...         np.array([584., -11., 23., 79., 1001., 0., -19.]),
    ...         device_id=all_gpu_devices)
    ...     pmp.update()
    ...     pmp.PAN_
    array([[0., 1., 1., 1., 1., 1., 1.],
           [0., 1., 1., 1., 1., 1., 1.]])
    """

    def __init__(
        self,
        T,
        min_m=3,
        max_m=None,
        step=1,
        device_id=0,
        normalize=True,
        p=2.0,
        T_subseq_isconstant_func=None,
    ):
        """
        Initialize the ``stimp`` object and compute the Pan Matrix Profile

        Parameters
        ----------
        T : numpy.ndarray
            The time series or sequence for which to compute the pan matrix profile.

        min_m : int, default 3
            The minimum subsequence window size to consider computing a matrix profile
            for.

        max_m : int, default None
            The maximum subsequence window size to consider computing a matrix profile
            for. When ``max_m = None``, this is set to the maximum allowable
            subsequence window size.

        step : int, default 1
            The step between subsequence window sizes.

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
            The p-norm to apply for computing the Minkowski distance. Minkowski distance
            is typically used with ``p`` being ``1`` or ``2``, which correspond to the
            Manhattan distance and the Euclidean distance, respectively. This parameter
            is ignored when ``normalize == True``.

        T_subseq_isconstant_func : function, default None
            A custom, user-defined function that returns a boolean array that indicates
            whether a subsequence in ``T`` is constant (`True``). The function must
            only take two arguments, ``a``, a 1-D array, and ``w``, the window size,
            while additional arguments may be specified by currying the user-defined
            function using  `functools.partial``. Any subsequence with at least one
            ``np.nan``/``np.inf`` will automatically have its corresponding value set
            to ``False`` in this boolean array.
        """
        super().__init__(
            T,
            min_m=min_m,
            max_m=max_m,
            step=step,
            percentage=1.0,
            pre_scrump=False,
            device_id=device_id,
            T_subseq_isconstant_func=T_subseq_isconstant_func,
            mp_func=gpu_stump,
        )
