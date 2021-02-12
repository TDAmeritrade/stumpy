#!/bin/bash

install_mode="normal"

# Parse first command line argument
if [ $# -gt 0 ]; then
    if [ $1 == "min" ]; then
        install_mode="min"
        echo "Installing minimum dependencies with install_mode=\"min\""
    else
        echo "Using default install_mode=\"normal\""
    fi
fi

###############
#  Functions  #
###############

generate_min_environment_yaml()
{
    echo "Generating \"environment.min.yml\" File"
    numpy="$(grep -E "numpy" environment.yml)"
    scipy="$(grep -E "scipy" environment.yml)"
    numba="$(grep -E "numba" environment.yml)"
    min_numpy="$(grep -E "numpy" environment.yml | sed 's/>//')"
    min_scipy="$(grep -E "scipy" environment.yml | sed 's/>//')"
    min_numba="$(grep -E "numba" environment.yml | sed 's/>//')"

    sed "s/${numpy}/${min_numpy}/" environment.yml | sed "s/${scipy}/${min_scipy}/" | sed "s/${numba}/${min_numba}/" > environment.min.yml
}

fix_libopenblas()
{
    if [ ! -f $CONDA_PREFIX/lib/libopenblas.dylib ]; then
        if [ -f $CONDA_PREFIX/lib/libopenblas.0.dylib ]; then
            ln -s $CONDA_PREFIX/lib/libopenblas.0.dylib $CONDA_PREFIX/lib/libopenblas.dylib
        fi
    fi
}

clean_up()
{
    echo "Cleaning Up"
    rm -rf "environment.min.yml"
}

###########
#   Main  #
###########

conda update -y conda
conda update -y --all
conda install -y -c conda-forge mamba

if [ $install_mode == "min" ]; then
    generate_min_environment_yaml
    # conda env update --file environment.min.yml
    mamba env update --file environment.min.yml
else
    # conda env update --file environment.yml
    mamba env update --file environment.yml
    echo "Please reboot the server to resolve any CUDA driver/library version mismatches"
fi

fix_libopenblas

if [[ `uname` == "Linux" && `which nvcc | wc -l` -lt "1" ]]; then
    # conda install -y -c conda-forge cudatoolkit-dev
    mamba install -y -c conda-forge cudatoolkit-dev
fi

#conda install -y -c conda-forge numpy scipy numba pandas flake8 flake8-docstrings black pytest-cov
#conda install -y -c conda-forge dask distributed

clean_up
