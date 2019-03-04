#!/bin/sh

export NUMBA_DISABLE_JIT=1
SITE_PACKAGES_DIR=`pip show matrix_profile | grep Location | awk '{print $2}'`
py.test --cov=$SITE_PACKAGES_DIR/matrix_profile --cov-report term-missing tests/ --capture=sys -s
