# STUMPY
# Copyright 2019 TD Ameritrade. Released under the terms of the 3-Clause BSD license.
# STUMPY is a trademark of TD Ameritrade IP Company, Inc. All rights reserved.

import numpy as np

STUMPY_THREADS_PER_BLOCK = 512
STUMPY_MEAN_STD_NUM_CHUNKS = 1
STUMPY_MEAN_STD_MAX_ITER = 10
STUMPY_DENOM_THRESHOLD = 1e-14
STUMPY_STDDEV_THRESHOLD = 1e-7
STUMPY_D_SQUARED_THRESHOLD = 1e-14
STUMPY_TEST_PRECISION = 5
STUMPY_MAX_SQUARED_DISTANCE = np.finfo(np.float64).max
STUMPY_MAX_DISTANCE = np.sqrt(STUMPY_MAX_SQUARED_DISTANCE)
