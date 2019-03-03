#!/bin/sh

pip install -r requirements.txt
pip uninstall -y matrix_profile
pip install .
