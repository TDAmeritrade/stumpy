#!/bin/sh

mode=""
extra=""

echo "y" |  python -m pip uninstall stumpy

# Parse command line arguments
for var in "$@"
do
    if [[ $var == "dev" ]] || [[ $var == "ci" ]]; then
        echo 'Installing stumpy locally with extra "ci" requirement'
        mode=""
        extra="[ci]"
    elif [[ $var == "edit" ]]; then
        echo 'Installing stumpy locally in "--editable" mode'
        mode="--editable"
    elif [[ $var == "-e" ]]; then
        echo 'Installing stumpy locally in "--editable" mode'
        mode="--editable"
    else
        echo "Installing stumpy in site-packages"
    fi
done

python -m pip install $mode .$extra
rm -rf build dist stumpy.egg-info __pycache__

site_pkgs=$(python -c 'import site; print(site.getsitepackages()[0])')
if [ -d "$site_pkgs/stumpy/__pycache__" ]; then
    rm -rf $site_pkgs/stumpy/__pycache__/*nb*
fi
