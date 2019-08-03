#!/bin/sh

# 1. Update version number in setup.py
# 2. Update CHANGELOG
# 3. conda-forge should update itself automatically
#
# To check that the distribution is valid, execute:
# twine check dist/* 

rm -rf dist
python3 setup.py sdist bdist_wheel 
twine upload --verbose --repository-url https://test.pypi.org/legacy/ dist/*
#twine upload dist/*
