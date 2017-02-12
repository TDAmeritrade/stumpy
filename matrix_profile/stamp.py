#!/usr/bin/env python

from . import core

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
