#!/bin/bash

arch_name="$(uname -m)"

if [ "${arch_name}" = "arm64" ]; then
    echo "Sorry, cannot install numba release candidate envrionment for ARM64 architecture"
else
    echo "Installing environment for numba release candidate"
    if [ -z "$2" ]; then
        numba_version=`conda search --override-channels -c numba numba | tail -n 1 | awk '{print $2}'`
    else
        numba_version="$2"
    fi
    if [ -z "$3" ]; then
        python_version=`conda search --override-channels -c conda-forge python | tail -n 1 | awk '{print $2}'`
    else
        python_version="$3"
    fi
    conda install -y -c conda-forge python=$python_version
    conda install -y -c numba numba=$numba_version
    ./conda.sh
fi
