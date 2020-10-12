#!/bin/sh

conda update -y conda
conda update -y --all
conda env update --file environment.yml
if [[ `uname` == "Linux" && `which nvcc | wc -l` -lt "1" ]]; then
    conda install -y -c conda-forge cudatoolkit-dev
fi
#conda install -y -c conda-forge numpy scipy numba pandas flake8 flake8-docstrings black pytest-cov
#conda install -y -c conda-forge dask distributed
