#!/bin/sh

conda install -y cython
pip uninstall -y matrix_profile
pip install .
