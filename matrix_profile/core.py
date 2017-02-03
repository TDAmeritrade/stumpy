#!/usr/bin/env python

import sys
import argparse

def check_python_version():
    if (sys.version_info < (3, 0)):
        raise Exception('Matrix Profile is only compatible with python3.x')

def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('ts_file', help='Time series input file')
    parser.add_argument('subseq_length', help='Subsequence length', type=int)

    return parser

def sliding_dot_product():
    pass

if __name__ == '__main__':
    check_python_version()
    parser = get_parser()
    args = parser.parse_args()
    print(args)