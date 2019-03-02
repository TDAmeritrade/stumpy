#!/usr/bin/env python

import matrix_profile as mp
import numpy as np
from timeit import default_timer as timer

for i in range(3, 7):
    n = 10**i
    x = np.random.rand(n)

    start = timer()
    mp.stump.stump(x, x, 50, ignore_trivial=True)
    end = timer()
    print(f'n: {n}  time: {end-start}')
