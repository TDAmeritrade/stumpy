#!/usr/bin/env python

import sys
import numpy as np

import core



if __name__ == '__main__':
    core.check_python_version()
    parser = core.get_parser()
    args = parser.parse_args()
    print(args)