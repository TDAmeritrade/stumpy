#!/bin/bash

test_mode="all"
print_mode="verbose"
custom_testfiles=()
max_iter=10
site_pkgs=$(python -c 'import site; print(site.getsitepackages()[0])')

# Parse command line arguments
for var in "$@"
do
    if [[ $var == "unit" ]]; then
        test_mode="unit"
    elif [[ $var == "coverage" ]]; then
        test_mode="coverage"
    elif [[ $var == "notebooks" ]]; then
        test_mode="notebooks"
    elif [[ $var == "gpu" ]] || [[ $var == "gpus" ]]; then
        test_mode="gpu"
    elif [[ $var == "show" ]]; then
        test_mode="show"
    elif [[ $var == "custom" ]]; then
        test_mode="custom"
    elif [[ $var == "report" ]]; then
        test_mode="report"
    elif [[ $var == "silent" || $var == "print" ]]; then
        print_mode="silent"
    elif [[ "$var" == *"test_"*".py"* ]]; then
        custom_testfiles+=("$var")
    elif [[ $var =~ ^[\-0-9]+$ ]]; then
        max_iter=$var
    elif [[ "$var" == "links" ]]; then
        test_mode="links"
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
  if [[ $1 -ne "0" && $1 -ne "5" ]]; then
    echo "Error: Test execution encountered exit code $1"
    # as a bonus, make our script exit with the right error code.
    exit $1
  fi
}

check_black()
{
    echo "Checking Black Code Formatting"
    black --check --exclude=".*\.ipynb" --diff ./
    check_errs $?
}

check_isort()
{
    echo "Checking iSort Import Formatting"
    isort --profile black --check-only ./
    check_errs $?
}

check_docstrings()
{
    echo "Checking Missing Docstrings"
    ./docstring.py
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

gen_ray_coveragerc()
{
    # Generate a .coveragerc_ray file that excludes Ray functions and tests
    echo "[report]" > .coveragerc_ray
    echo "; Regexes for lines to exclude from consideration" >> .coveragerc_ray
    echo "exclude_also =" >> .coveragerc_ray
    echo "    def .*_ray_*" >> .coveragerc_ray
    echo "    def ,*_ray\(*" >> .coveragerc_ray
    echo "    def ray_.*" >> .coveragerc_ray
    echo "    def test_.*_ray*" >> .coveragerc_ray
}

gen_coverage_report()
{
    # If `ray` command is not found then generate a .coveragerc_ray file
    if ! command -v ray &> /dev/null
    then
        echo "Ray Not Installed"
        gen_ray_coveragerc
        fcoverage="--rcfile=.coveragerc_ray"
    else
        echo "Ray Installed"
        fcoverage=""
    fi

    # This prints a coverage report to screen
    coverage report -m --fail-under=100 --skip-covered --omit=setup.py,docstring.py,min_versions.py,ray_python_version.py,stumpy/cache.py $fcoverage
    # This saves the coverage report in Cobertura XML format, which is compatible with codecov
    coverage xml --fail-under=100 --omit=setup.py,docstring.py,min_versions.py,ray_python_version.py,stumpy/cache.py $fcoverage
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
    #
    # You may mimic coverage testing conditions by disabling `numba` JIT
    # and enabling the `cuda` simulator by setting two environment
    # variables prior to calling `test.sh`:
    #
    # NUMBA_DISABLE_JIT=1 NUMBA_ENABLE_CUDASIM=1 ./test.sh custom 5 tests/test_gpu_stump.py

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
                pytest -rsx -W ignore::RuntimeWarning -W ignore::DeprecationWarning -W ignore::UserWarning $testfile
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
    for testfile in tests/test_*.py
    do
        pytest -rsx -W ignore::RuntimeWarning -W ignore::DeprecationWarning -W ignore::UserWarning $testfile
        check_errs $?
    done
}

test_coverage()
{
    echo "Disabling Numba JIT and CUDA Compiled Functions"
    export NUMBA_DISABLE_JIT=1
    export NUMBA_ENABLE_CUDASIM=1

    # echo "Testing Python Functions"
    # pytest -rsx -W ignore::RuntimeWarning -W ignore::DeprecationWarning -W ignore::UserWarning tests
    # check_errs $?

    echo "Testing Code Coverage"
    coverage erase

    # We always attempt to test everything but we may ignore things (ray, helper scripts) when we generate the coverage report

    for testfile in tests/test_*.py
    do
        coverage run --append --source=. -m pytest -rsx -W ignore::RuntimeWarning -W ignore::DeprecationWarning -W ignore::UserWarning $testfile
        check_errs $?
        break
    done

    gen_coverage_report
}

test_gpu()
{
    echo "Testing Numba JIT CUDA GPU Compiled Functions"
    #for testfile in tests/test_*gpu*.py tests/test_core.py tests/test_precision.py tests/test_non_normalized_decorator.py
    for testfile in $(grep gpu tests/* | awk -v FS=':' '{print $1}' | uniq);
    do
        pytest -rsx -W ignore::RuntimeWarning -W ignore::DeprecationWarning -W ignore::UserWarning $testfile
        check_errs $?
    done
}

show()
{
    echo "Current working directory: " `pwd`
    echo "Black version: " `python -c "import black; print(black.__version__)"`
    echo "Flake8 versoin: " `python -c "import flake8; print(flake8.__version__)"`
    echo "Python version: " `python -c "import platform; print(platform.python_version())"`
    echo "NumPy version: " `python -c "import numpy; print(numpy.__version__)"`
    echo "SciPy version: " `python -c "import scipy; print(scipy.__version__)"`
    echo "Numba version: " `python -c "import numba; print(numba.__version__)"` 
    exit 0
}

check_links()
{
    echo "Checking notebook links"
    pytest --check-links docs/Tutorial_*.ipynb
}

clean_up()
{
    echo "Cleaning Up"
    rm -rf "dask-worker-space"
    rm -rf "stumpy/__pycache__/"
    rm -rf "tests/__pycache__/"
    rm -rf build dist stumpy.egg-info __pycache__
    rm -f docs/*.nbconvert.ipynb
    rm -rf ".coveragerc_ray"
    if [ -d "$site_pkgs/stumpy/__pycache__" ]; then
        rm -rf $site_pkgs/stumpy/__pycache__/*nb*
    fi

}

convert_notebooks()
{
    echo "testing notebooks"
    for notebook in `grep ipynb docs/tutorials.rst | sed -e 's/^[ \t]*//'`
    do
        jupyter nbconvert --to notebook --execute "docs/$notebook"
        check_errs $?
    done
}

###########
#   Main  #
###########

if [[ $test_mode == "show" ]]; then
    echo "Show development/test environment"
    show
fi

clean_up
check_black
check_isort
check_flake
check_docstrings
check_print
check_naive

if [[ $test_mode == "notebooks" ]]; then
    echo "Executing Tutorial Notebooks Only"
    convert_notebooks
elif [[ $test_mode == "unit" ]]; then
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
elif [[ $test_mode == "report" ]]; then
    echo "Generate Coverage Report Only"
    # Assume coverage tests have already been executed
    # and a coverage file exists
    gen_coverage_report
elif [[ $test_mode == "gpu" ]]; then
    echo "Executing GPU Unit Tests Only"
    test_gpu
elif [[ $test_mode == "links" ]]; then
    echo "Check Notebook Links  Only"
    check_links
else
    echo "Executing Unit Tests And Code Coverage"
    test_unit
    clean_up
    test_coverage
fi

clean_up
