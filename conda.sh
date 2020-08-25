#!/bin/sh

conda update -y conda
conda update -y --all
conda env update --file environment.yml
#conda install -y -c conda-forge numpy scipy numba pandas flake8 flake8-docstrings black pytest-cov
#conda install -y -c conda-forge dask distributed
