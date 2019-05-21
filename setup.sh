#!/bin/sh

pip install -r requirements.txt
pip uninstall -y stumpy
pip install -e .[deploy]
rm -rf stumpy.egg-info build dist
