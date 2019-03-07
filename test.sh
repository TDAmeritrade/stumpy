#!/bin/sh

echo "Testing Numba JIT Compiled Functions"
py.test --capture=sys -s tests/test_stump.py tests/test_stumped.py

echo "Disabling Numba JIT Compiled Functions"
export NUMBA_DISABLE_JIT=1

echo "Testing Python Functions"
SITE_PACKAGES_DIR=`pip show matrix_profile | grep Location | awk '{print $2}'`
py.test --cov=$SITE_PACKAGES_DIR/matrix_profile --cov-report term-missing tests/ --capture=sys -s
