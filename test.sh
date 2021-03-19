#!/bin/bash

test_mode="all"

# Parse first command line argument
if [[ $# -gt 0 ]]; then
    if [[ $1 == "unit" ]]; then
        test_mode="unit"
    elif [[ $1 == "coverage" ]]; then
        test_mode="coverage"
    elif [[ $1 == "custom" ]]; then
        test_mode="custom"
    else
        echo "Using default test_mode=\"all\""
    fi
fi

###############
#  Functions  #
###############

check_errs()
{
  # Function. Parameter 1 is the return code
  if [[ $1 -ne "0" ]]; then
    echo "Error: pytest encountered exit code $1"
    # as a bonus, make our script exit with the right error code.
    exit $1
  fi
}

check_black()
{
    echo "Checking Black Code Formatting"
    black --check --diff ./
    check_errs $?
}

check_flake()
{
    echo "Checking Flake8 Style Guide Enforcement"
    flake8 ./
    check_errs $?
}

check_print()
{
    if [[ `grep print */*.py | wc -l` -gt "0" ]]; then
        echo "Error: print statement found in code"
        grep print */*.py
        exit 1
    fi
}

test_custom()
{
    # export NUMBA_DISABLE_JIT=1
    # export NUMBA_ENABLE_CUDASIM=1
    # Test one or more user-defined functions repeatedly
    for VARIABLE in {1..10}
    do
        py.test -x -W ignore::RuntimeWarning -W ignore::DeprecationWarning tests/test_.py
        check_errs $?
    done
    clean_up
    exit 0
}

test_unit()
{
    echo "Testing Numba JIT Compiled Functions"
    py.test -rsx -W ignore::RuntimeWarning -W ignore::DeprecationWarning tests/test_gpu_stump.py
    py.test -x -W ignore::RuntimeWarning -W ignore::DeprecationWarning tests/test_core.py
    check_errs $?
    py.test -x -W ignore::RuntimeWarning -W ignore::DeprecationWarning tests/test_stump.py tests/test_mstump.py tests/test_scrump.py tests/test_stumpi.py
    check_errs $?
    py.test -x -W ignore::RuntimeWarning -W ignore::DeprecationWarning tests/test_stumped.py
    check_errs $?
    py.test -x -W ignore::RuntimeWarning -W ignore::DeprecationWarning tests/test_mstumped.py
    check_errs $?
    py.test -x -W ignore::RuntimeWarning -W ignore::DeprecationWarning tests/test_ostinato.py
    check_errs $?
    py.test -x -W ignore::RuntimeWarning -W ignore::DeprecationWarning tests/test_gpu_ostinato.py
    py.test -x -W ignore::RuntimeWarning -W ignore::DeprecationWarning tests/test_mpdist.py
    check_errs $?
    py.test -x -W ignore::RuntimeWarning -W ignore::DeprecationWarning tests/test_gpu_mpdist.py
    # aamp
    py.test -rsx -W ignore::RuntimeWarning -W ignore::DeprecationWarning tests/test_gpu_aamp.py
    py.test -x -W ignore::RuntimeWarning -W ignore::DeprecationWarning tests/test_aamp.py tests/test_maamp.py tests/test_scraamp.py tests/test_aampi.py
    check_errs $?
    py.test -x -W ignore::RuntimeWarning -W ignore::DeprecationWarning tests/test_aamped.py
    check_errs $?
    py.test -x -W ignore::RuntimeWarning -W ignore::DeprecationWarning tests/test_maamped.py
    check_errs $?
    py.test -x -W ignore::RuntimeWarning -W ignore::DeprecationWarning tests/test_aamp_ostinato.py
    check_errs $?
    py.test -x -W ignore::RuntimeWarning -W ignore::DeprecationWarning tests/test_gpu_aamp_ostinato.py
    py.test -x -W ignore::RuntimeWarning -W ignore::DeprecationWarning tests/test_aampdist.py
    check_errs $?
    py.test -x -W ignore::RuntimeWarning -W ignore::DeprecationWarning tests/test_gpu_aampdist.py
    py.test -x -W ignore::RuntimeWarning -W ignore::DeprecationWarning tests/test_non_normalized_decorator.py
}

test_coverage()
{
    echo "Disabling Numba JIT and CUDA Compiled Functions"
    export NUMBA_DISABLE_JIT=1
    export NUMBA_ENABLE_CUDASIM=1

    # echo "Testing Python Functions"
    # py.test -x -W ignore::RuntimeWarning -W ignore::DeprecationWarning tests
    # check_errs $?

    echo "Testing Code Coverage"
    coverage run --source stumpy -m py.test -x -W ignore::RuntimeWarning -W ignore::DeprecationWarning
    check_errs $?
    coverage report -m
}

clean_up()
{
    echo "Cleaning Up"
    rm -rf "dask-worker-space"
}

###########
#   Main  #
###########

clean_up
check_black
check_flake
check_print

if [[ $test_mode == "unit" ]]; then
    echo "Executing Unit Tests Only"
    test_unit
elif [[ $test_mode == "coverage" ]]; then
    echo "Executing Code Coverage Only"
    test_coverage
elif [[ $test_mode == "custom" ]]; then
    echo "Executing Custom User-Defined Tests Only"
    # Define tests in `test_custom` function above
    # echo "Disabling Numba JIT and CUDA Compiled Functions"
    # export NUMBA_DISABLE_JIT=1
    # export NUMBA_ENABLE_CUDASIM=1
    test_custom
else
    echo "Executing Unit Tests And Code Coverage"
    test_unit
    clean_up
    test_coverage
fi

clean_up
