#!/bin/sh

echo "Testing Numba JIT Compiled Functions"
py.test -W ignore::RuntimeWarning -W ignore::DeprecationWarning tests/test_stump.py tests/test_mstump.py 
py.test -W ignore::RuntimeWarning -W ignore::DeprecationWarning tests/test_stumped.py tests/test_mstumped.py

echo "Disabling Numba JIT  and CUDA Compiled Functions"
export NUMBA_DISABLE_JIT=1
export NUMBA_ENABLE_CUDASIM=1

echo "Testing Python Functions"
py.test -W ignore::RuntimeWarning -W ignore::DeprecationWarning tests

echo "Test Code Coverage"
coverage run --source stumpy -m py.test -W ignore::RuntimeWarning -W ignore::DeprecationWarning
coverage report -m

echo "Cleaning Up"
rm -rf "dask-worker-space"
