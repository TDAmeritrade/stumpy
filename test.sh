#!/bin/sh

#############################
#   Define Helper Function  #
#############################

create_coveragerc()
{
echo "[report]" > .coveragerc
echo "exclude_lines=" >> .coveragerc
echo "    pragma: no cover" >> .coveragerc
if numba -s | grep -q "Found.*CUDA"; then
    numba -s | grep "Found.*CUDA"
else
    echo "    pragma: no gpu cover" >> .coveragerc
fi
}

echo "Create .coveragerc file for environment"
create_coveragerc

###############
#  Run Tests  #
###############

echo "Testing Numba JIT Compiled Functions"
py.test -W ignore::RuntimeWarning -W ignore::DeprecationWarning tests/test_stump.py tests/test_stumped.py tests/test_mstump.py tests/test_mstumped.py

echo "Disabling Numba JIT Compiled Functions"
export NUMBA_DISABLE_JIT=1

echo "Testing Python Functions"
py.test -W ignore::RuntimeWarning -W ignore::DeprecationWarning tests

echo "Test Code Coverage"
coverage run --source stumpy -m py.test -W ignore::RuntimeWarning -W ignore::DeprecationWarning
coverage report -m

echo "Cleaning Up"
rm -rf "dask-worker-space"
