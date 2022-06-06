#!/bin/bash

test_mode="all"
print_mode="verbose"
custom_testfiles=()
max_iter=10

# Parse command line arguments
for var in "$@"
do
    if [[ $var == "unit" ]]; then
        test_mode="unit"
    elif [[ $var == "coverage" ]]; then
        test_mode="coverage"
    elif [[ $var == "custom" ]]; then
        test_mode="custom"
    elif [[ $var == "silent" || $var == "print" ]]; then
        print_mode="silent"
    elif [[ "$var" == *"test_"*".py"* ]]; then
        custom_testfiles+=("$var")
    elif [[ $var =~ ^[\-0-9]+$ ]]; then
        max_iter=$var
    else
        echo "Using default test_mode=\"all\""
    fi
done

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
    if [[ $print_mode == "verbose" ]]; then
        if [[ `grep print */*.py | wc -l` -gt "0" ]]; then
            echo "Error: print statement found in code"
            grep print */*.py
            exit 1
        fi
    fi
}

check_naive()
{
    # Check if there are any naive implementations not at start of test file
    for testfile in tests/test_*.py
    do
        last_naive="$(grep -n 'def naive_' $testfile | tail -n 1 | awk -F: '{print $1}')"
        first_test="$(grep -n 'def test_' $testfile | head -n 1 | awk -F: '{print $1}')"
        if [[ ! -z $last_naive && ! -z $first_test && $last_naive -gt $first_test ]]; then
            echo "Error: naive implementation found in the middle of $testfile line $last_naive"
            exit 1
        fi
    done
}

test_custom()
{
    # export NUMBA_DISABLE_JIT=1
    # export NUMBA_ENABLE_CUDASIM=1
    # Test one or more user-defined functions repeatedly
    # 
    # ./test.sh custom tests/test_stump.py
    # ./test.sh custom 5 tests/test_stump.py
    # ./test.sh custom 5 tests/test_stump.py::test_stump_self_join

    if [[ ${#custom_testfiles[@]}  -eq "0" ]]; then
        echo ""
        echo "Error: Missing custom test file(s)"
        echo "Please specify one or more custom test files"
        echo "Example: ./test.sh custom tests/test_stump.py"
        exit 1
    else
        for i in $(seq $max_iter)
        do
            echo "Custom Test: $i / $max_iter"
            for testfile in "${custom_testfiles[@]}"
            do
                pytest -x -W ignore::DeprecationWarning $testfile
                check_errs $?
            done
        done
        clean_up
        exit 0
    fi
}

test_unit()
{
    echo "Testing Numba JIT Compiled Functions"
    pytest -rsx -W ignore::RuntimeWarning -W ignore::DeprecationWarning tests/test_gpu_stump.py
    pytest -x -W ignore::RuntimeWarning -W ignore::DeprecationWarning tests/test_core.py
    check_errs $?
    pytest -x -W ignore::RuntimeWarning -W ignore::DeprecationWarning tests/test_config.py
    check_errs $?
    pytest -x -W ignore::RuntimeWarning -W ignore::DeprecationWarning tests/test_stump.py tests/test_mstump.py tests/test_stumpi.py
    check_errs $?
    pytest -x -W ignore::RuntimeWarning -W ignore::DeprecationWarning tests/test_scrump.py
    check_errs $?
    pytest -x -W ignore::RuntimeWarning -W ignore::DeprecationWarning tests/test_stumped.py
    check_errs $?
    pytest -x -W ignore::RuntimeWarning -W ignore::DeprecationWarning tests/test_mstumped.py
    check_errs $?
    pytest -x -W ignore::RuntimeWarning -W ignore::DeprecationWarning tests/test_ostinato.py
    check_errs $?
    pytest -x -W ignore::RuntimeWarning -W ignore::DeprecationWarning tests/test_gpu_ostinato.py
    pytest -x -W ignore::RuntimeWarning -W ignore::DeprecationWarning tests/test_mpdist.py
    check_errs $?
    pytest -x -W ignore::RuntimeWarning -W ignore::DeprecationWarning tests/test_motifs.py
    check_errs $?
    pytest -x -W ignore::RuntimeWarning -W ignore::DeprecationWarning tests/test_mmotifs.py
    check_errs $?
    pytest -x -W ignore::RuntimeWarning -W ignore::DeprecationWarning tests/test_gpu_mpdist.py
    pytest -x -W ignore::RuntimeWarning -W ignore::DeprecationWarning tests/test_snippets.py
    check_errs $?
    pytest -rsx -W ignore::RuntimeWarning -W ignore::DeprecationWarning tests/test_gpu_stimp.py
    pytest -x -W ignore::RuntimeWarning -W ignore::DeprecationWarning tests/test_stimp.py
    check_errs $?
    # aamp
    pytest -rsx -W ignore::RuntimeWarning -W ignore::DeprecationWarning tests/test_gpu_aamp.py
    pytest -x -W ignore::RuntimeWarning -W ignore::DeprecationWarning tests/test_aamp.py tests/test_maamp.py tests/test_scraamp.py tests/test_aampi.py
    check_errs $?
    pytest -x -W ignore::RuntimeWarning -W ignore::DeprecationWarning tests/test_scraamp.py
    check_errs $?
    pytest -x -W ignore::RuntimeWarning -W ignore::DeprecationWarning tests/test_aamped.py
    check_errs $?
    pytest -x -W ignore::RuntimeWarning -W ignore::DeprecationWarning tests/test_maamped.py
    check_errs $?
    pytest -x -W ignore::RuntimeWarning -W ignore::DeprecationWarning tests/test_aamp_ostinato.py
    check_errs $?
    pytest -x -W ignore::RuntimeWarning -W ignore::DeprecationWarning tests/test_gpu_aamp_ostinato.py
    pytest -x -W ignore::RuntimeWarning -W ignore::DeprecationWarning tests/test_aampdist.py
    check_errs $?
    pytest -x -W ignore::RuntimeWarning -W ignore::DeprecationWarning tests/test_aamp_motifs.py
    check_errs $?
    pytest -x -W ignore::RuntimeWarning -W ignore::DeprecationWarning tests/test_aamp_mmotifs.py
    check_errs $?
    pytest -x -W ignore::RuntimeWarning -W ignore::DeprecationWarning tests/test_gpu_aampdist.py
    pytest -x -W ignore::RuntimeWarning -W ignore::DeprecationWarning tests/test_aampdist_snippets.py
    check_errs $?
    pytest -rsx -W ignore::RuntimeWarning -W ignore::DeprecationWarning tests/test_gpu_aamp_stimp.py
    pytest -x -W ignore::RuntimeWarning -W ignore::DeprecationWarning tests/test_aamp_stimp.py
    check_errs $?
    pytest -x -W ignore::DeprecationWarning tests/test_non_normalized_decorator.py
    check_errs $?
}

test_coverage()
{
    echo "Disabling Numba JIT and CUDA Compiled Functions"
    export NUMBA_DISABLE_JIT=1
    export NUMBA_ENABLE_CUDASIM=1

    # echo "Testing Python Functions"
    # pytest -x -W ignore::RuntimeWarning -W ignore::DeprecationWarning tests
    # check_errs $?

    echo "Testing Code Coverage"
    coverage erase
    for testfile in tests/test_*.py
    do
        coverage run --append --source=. -m pytest -x -W ignore::RuntimeWarning -W ignore::DeprecationWarning $testfile
        check_errs $?
    done
    coverage report -m --skip-covered --omit=setup.py
}

clean_up()
{
    echo "Cleaning Up"
    rm -rf "dask-worker-space"
    rm -rf "stumpy/__pycache__/"
}

###########
#   Main  #
###########

clean_up
check_black
check_flake
check_print
check_naive

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
