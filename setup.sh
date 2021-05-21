#!/bin/sh

echo "y" |  python -m pip uninstall stumpy
python -m pip install --use-feature=in-tree-build .
rm -rf stumpy.egg-info build dist __pycache__
