#!/usr/bin/env python

import matrix_profile as mp
import numpy as np
from timeit import default_timer as timer
import logging

logger = logging.getLogger(__name__)

def get_human_readable_time(total_time):
        hours, rem = divmod(total_time, 3600)
        minutes, seconds = divmod(rem, 60)
        hours = int(hours)
        minutes = int(minutes)

        return f"{hours:0>2}:{minutes:0>2}:{seconds:05.2f}"

if __name__ == '__main__':
    # Makes sure stump function is already jit compiled
    x = np.random.rand(51)
    mp.stump.stump(x, x, 50, ignore_trivial=True)

    for i in range(6, 25):
        n = 2**i
        x = np.random.rand(n)

        start = timer()
        mp.stump.stump(x, x, 50, ignore_trivial=True)
        end = timer()
        elapsed_time = get_human_readable_time(end-start)
        logger.warning(f'i: {i}  n: {n}  time: {elapsed_time}')
