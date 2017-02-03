#!/usr/bin/env python

import sys
import numpy as np

import core

class STAMP(object):
    def __init__(self):
        """
        """
        self.T = None
        self.Q = None



if __name__ == '__main__':
    core.check_python_version()
    parser = core.get_parser()
    args = parser.parse_args()
    print(args)