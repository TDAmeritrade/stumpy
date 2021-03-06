#!/bin/bash

arch_name="$(uname -m)"

if [ "${arch_name}" = "arm64" ]; then
    echo "Sorry, cannot install numba release candidate envrionment for ARM64 architecture"
else
    conda update -y conda
    conda update -y --all

    if [ -z "$2" ]; then
        numba_version=`conda search --override-channels -c numba numba | tail -n 1 | awk '{print $2}'`
    else
        numba_version="$2"
    fi

    if [ -z "$3" ]; then
        python_version=`conda search --override-channels -c conda-forge python | tail -n 1 | awk '{print $2}'`
        # Strip away patch version
        # python_version="${python_version%.*}"
    else
        python_version="$3"
    fi

    echo ""
    echo "Installing python=$python_version"
    echo ""
    conda install -y -c conda-forge python=$python_version

    echo ""
    echo "Installing numba=$numba_version"
    echo ""
    conda install -y -c numba numba=$numba_version

    echo ""
    echo "Installing STUMPY environment"
    echo ""
    grep -Ev "numba|python" environment.yml > environment.numba.yml
    conda env update --file environment.numba.yml
    rm -rf environment.numba.yml
fi
