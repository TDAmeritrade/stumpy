# STUMPY
# Copyright 2019 TD Ameritrade. Released under the terms of the 3-Clause BSD license.
# STUMPY is a trademark of TD Ameritrade IP Company, Inc. All rights reserved.

import warnings

import numpy as np

_STUMPY_DEFAULTS = {
    "STUMPY_THREADS_PER_BLOCK": 512,
    "STUMPY_MEAN_STD_NUM_CHUNKS": 1,
    "STUMPY_MEAN_STD_MAX_ITER": 10,
    "STUMPY_DENOM_THRESHOLD": 1e-14,
    "STUMPY_STDDEV_THRESHOLD": 1e-7,
    "STUMPY_P_NORM_THRESHOLD": 1e-14,
    "STUMPY_TEST_PRECISION": 5,
    "STUMPY_MAX_P_NORM_DISTANCE": np.finfo(np.float64).max,
    "STUMPY_MAX_DISTANCE": np.sqrt(np.finfo(np.float64).max),
    "STUMPY_EXCL_ZONE_DENOM": 4,
    "STUMPY_FASTMATH_TRUE": True,
    "STUMPY_FASTMATH_FLAGS": {"nsz", "arcp", "contract", "afn", "reassoc"},
    "STUMPY_FASTMATH_FASTMATH._ADD_ASSOC": True,
}

# In addition to these configuration variables, there exist config variables
# that have the default value of the fastmath flag of the njit functions. The
# name of this config variable has the following format:
# STUMPY_FASTMATH_<module_name>.<function_name>
# See __init__.py for more details

STUMPY_THREADS_PER_BLOCK = _STUMPY_DEFAULTS["STUMPY_THREADS_PER_BLOCK"]
STUMPY_MEAN_STD_NUM_CHUNKS = _STUMPY_DEFAULTS["STUMPY_MEAN_STD_NUM_CHUNKS"]
STUMPY_MEAN_STD_MAX_ITER = _STUMPY_DEFAULTS["STUMPY_MEAN_STD_MAX_ITER"]
STUMPY_DENOM_THRESHOLD = _STUMPY_DEFAULTS["STUMPY_DENOM_THRESHOLD"]
STUMPY_STDDEV_THRESHOLD = _STUMPY_DEFAULTS["STUMPY_STDDEV_THRESHOLD"]
STUMPY_P_NORM_THRESHOLD = _STUMPY_DEFAULTS["STUMPY_P_NORM_THRESHOLD"]
STUMPY_TEST_PRECISION = _STUMPY_DEFAULTS["STUMPY_TEST_PRECISION"]
STUMPY_MAX_P_NORM_DISTANCE = _STUMPY_DEFAULTS["STUMPY_MAX_P_NORM_DISTANCE"]
STUMPY_MAX_DISTANCE = _STUMPY_DEFAULTS["STUMPY_MAX_DISTANCE"]
STUMPY_EXCL_ZONE_DENOM = _STUMPY_DEFAULTS["STUMPY_EXCL_ZONE_DENOM"]
STUMPY_FASTMATH_TRUE = _STUMPY_DEFAULTS["STUMPY_FASTMATH_TRUE"]
STUMPY_FASTMATH_FLAGS = _STUMPY_DEFAULTS["STUMPY_FASTMATH_FLAGS"]


def _reset(var=None):
    """
    Reset the value of a configuration variable(s) to their default value(s)

    Parameters
    ----------
    var : str, default None
        The name of the configuration variable. If None, then all
        configuration variables are reset to their default values.

    Returns
    -------
    None
    """
    config_vars = [
        k for k, _ in globals().items() if k.isupper() and k.startswith("STUMPY")
    ]

    if var is None:
        for config_var in config_vars:
            globals()[config_var] = _STUMPY_DEFAULTS[config_var]
    elif var in config_vars:
        globals()[var] = _STUMPY_DEFAULTS[var]
    else:  # pragma: no cover
        msg = (
            f"Configuration reset was skipped for unrecognized '_STUMPY_DEFAULT[{var}]'"
        )
        warnings.warn(msg)

    return
