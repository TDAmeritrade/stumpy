#!/bin/sh

python3 `which pytest` --capture=sys

#python3 `which pytest` tests/test_stamp.py::test_stamp_A_B_join
