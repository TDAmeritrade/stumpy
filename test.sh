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

echo "Checking Black Code Formatting"
black --check --diff ./
check_errs $?

echo "Checking Flake8 Style Guide Enforcement"
flake8 ./
check_errs $?

echo "Testing Numba JIT Compiled Functions"
py.test -rsx -W ignore::RuntimeWarning -W ignore::DeprecationWarning tests/test_gpu_stump.py
py.test -x -W ignore::RuntimeWarning -W ignore::DeprecationWarning tests/test_stump.py tests/test_mstump.py
check_errs $?
py.test -x -W ignore::RuntimeWarning -W ignore::DeprecationWarning tests/test_stumped.py tests/test_stumped_one_constant_subsequence.py tests/test_stumped_two_constant_subsequences.py tests/test_mstumped.py
check_errs $?
py.test -x -W ignore::RuntimeWarning -W ignore::DeprecationWarning tests/test_stumped_one_subsequence_nan_self_join.py tests/test_stumped_one_subsequence_inf_self_join.py tests/test_stumped_one_subsequence_nan_A_B_join.py tests/test_stumped_one_subsequence_inf_A_B_join.py
check_errs $?
py.test -x -W ignore::RuntimeWarning -W ignore::DeprecationWarning tests/test_stumped_two_subsequences_nan_A_B_join.py tests/test_stumped_two_subsequences_inf_A_B_join.py tests/test_stumped_two_subsequences_nan_inf_A_B_join.py tests/test_stumped_two_subsequences_nan_inf_A_B_join_swap.py
check_errs $?
py.test -rsx -W ignore::RuntimeWarning -W ignore::DeprecationWarning tests/test_scrimp.py
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
