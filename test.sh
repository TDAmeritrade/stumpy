#!/bin/sh

###############
#  Functions  #
###############

check_errs()
{
  # Function. Parameter 1 is the return code
  if [ "${1}" -ne "0" ]; then
    echo "Error: pytest encountered exit code ${1}"
    # as a bonus, make our script exit with the right error code.
    exit ${1}
  fi
}

echo "Testing Numba JIT Compiled Functions"
py.test -rsx -W ignore::RuntimeWarning -W ignore::DeprecationWarning tests/test_gpu_stump.py
py.test -x -W ignore::RuntimeWarning -W ignore::DeprecationWarning tests/test_stump.py tests/test_mstump.py 
check_errs $?
py.test -x -W ignore::RuntimeWarning -W ignore::DeprecationWarning tests/test_stumped.py tests/test_mstumped.py
check_errs $?

echo "Disabling Numba JIT  and CUDA Compiled Functions"
export NUMBA_DISABLE_JIT=1
export NUMBA_ENABLE_CUDASIM=1

echo "Testing Python Functions"
py.test -x -W ignore::RuntimeWarning -W ignore::DeprecationWarning tests
check_errs $?

echo "Test Code Coverage"
coverage run --source stumpy -m py.test -x -W ignore::RuntimeWarning -W ignore::DeprecationWarning
check_errs $?
coverage report -m

echo "Cleaning Up"
rm -rf "dask-worker-space"
