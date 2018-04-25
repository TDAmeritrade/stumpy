#!/bin/sh

conda install -y cython
conda install -y scipy
pip uninstall -y matrix_profile
pip install .
