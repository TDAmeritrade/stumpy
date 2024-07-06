#!/bin/bash

MAMBA_NO_LOW_SPEED_LIMIT=0
conda_env="$(conda info --envs | grep '*' | awk '{print $1}')"
arch_name="$(uname -m)"
if [[ $1 == "numba" ]] && [[ $arch_name == "arm64" ]]; then
    echo "Sorry, cannot install numba release candidate envrionment for ARM64 architecture"
fi
install_mode="normal"

# Parse first command line argument
if [[ $# -gt 0 ]]; then
    if [ $1 == "min" ]; then
        install_mode="min"
        echo "Installing minimum dependencies with install_mode=\"min\""
    elif [[ $1 == "ray" ]]; then
        install_mode="ray"
        echo "Installing ray dependencies with install_mode=\"ray\""
    elif [[ $1 == "numba" ]] && [[ "${arch_name}" != "arm64" ]]; then
        install_mode="numba"
        echo "Installing numba release candidate dependencies with install_mode=\"numba\""
        if [[ -z $2 ]]; then
            numba_version=`conda search --override-channels -c numba numba | tail -n 1 | awk '{print $2}'`
        else
            numba_version=$2
        fi
        # Set Python version
        if [[ -z $3 ]]; then
            python_version=`conda search --override-channels -c conda-forge python | tail -n 1 | awk '{print $2}'`
            # Strip away patch version
            # python_version="${python_version%.*}"
        else
            python_version=$3
        fi
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

generate_numba_environment_yaml()
{
    echo "Generating \"environment.numba.yml\" File"
    grep -Ev "numba|python" environment.yml > environment.numba.yml
}

generate_ray_environment_yaml()
{
    # Limit max Python version and append pip install ray
    echo "Generating \"environment.ray.yml\" File"
    ray_python=`./ray_python_version.py`
    sed "/  - python/ s/$/,<=$ray_python/" environment.yml | cat - <(echo $'  - pip\n  - pip:\n    - ray>=2.23.0') > environment.ray.yml
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
    rm -rf "environment.numba.yml"
    rm -rf "environment.ray.yml"
}

###########
#   Main  #
###########

conda update -c conda-forge -y conda
conda update -c conda-forge -y --all
conda install -y -c conda-forge mamba

if [[ `uname` == "Linux" && `which nvcc | wc -l` -lt "1" ]]; then
    rm -rf /tmp/cuda-installer.log
    # conda install -y -c conda-forge cudatoolkit-dev'<11.4'
    conda install -y -c conda-forge cudatoolkit-dev
    # mamba install -y -c conda-forge cudatoolkit-dev
    echo "Please reboot the server to resolve any CUDA driver/library version mismatches"
fi

if [[ $install_mode == "min" ]]; then
    generate_min_environment_yaml
    mamba env update --name $conda_env --file environment.min.yml || conda env update --name $conda_env --file environment.min.yml
elif [[ $install_mode == "ray" ]]; then
    generate_ray_environment_yaml
    mamba env update --name $conda_env --file environment.ray.yml || conda env update --name $conda_env --file environment.ray.yml
elif [[ $install_mode == "numba" ]]; then
    echo ""
    echo "Installing python=$python_version"
    echo ""
    mamba install -y -c conda-forge python=$python_version || conda install -y -c conda-forge python=$python_version

    echo ""
    echo "Installing numba=$numba_version"
    echo ""
    mamba install -y -c numba numba=$numba_version || conda install -y -c numba numba=$numba_version

    generate_numba_environment_yaml
    mamba env update --name $conda_env --file environment.numba.yml || conda env update --name $conda_env --file environment.numba.yml
else
    mamba env update --name $conda_env --file environment.yml || conda env update --name $conda_env --file environment.yml
    conda update -c conda-forge -y numpy scipy numba black twine
fi

fix_libopenblas

#conda install -y -c conda-forge numpy scipy numba pandas flake8 flake8-docstrings black pytest-cov
#conda install -y -c conda-forge dask distributed

clean_up
