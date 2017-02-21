#!/usr/bin/env python

from . import core

def rolling_window(a, window):
    shape = a.shape[:-1] + (a.shape[-1] - window + 1, window)
    strides = a.strides + (a.strides[-1],)
    return np.lib.stride_tricks.as_strided(a, shape=shape, strides=strides)

class STAMP(object):

    def __init__(self):
        """
        """
        self.T = None
        self.Q = None

    def get_matrix_profile(self):
        """
        """
        return


if __name__ == '__main__':
    core.check_python_version()
    parser = core.get_parser()
    args = parser.parse_args()
    mp = STAMP()
    mp.get_matrix_profile()
