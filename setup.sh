#!/bin/sh

echo "y" |  python -m pip uninstall stumpy
python -m pip install .
rm -rf stumpy.egg-info build dist __pycache__
