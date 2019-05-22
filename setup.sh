#!/bin/sh

pip uninstall -y stumpy
python setup.py install
rm -rf stumpy.egg-info build dist