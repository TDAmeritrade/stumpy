#!/usr/bin/env python

import argparse
import re

import pandas as pd
from packaging.specifiers import SpecifierSet
from packaging.version import Version


def get_min_python_version():
    """
    Find the minimum version of Python supported (i.e., not end-of-life)
    """
    min_python = (
        pd.read_html("https://devguide.python.org/versions/")[0].iloc[-1].Branch
    )
    return min_python


def get_min_numba_numpy_version(min_python):
    """
    Find the minimum versions of Numba and NumPy that supports the specified
    `min_python` version
    """
    df = (
        pd.read_html(
            "https://numba.readthedocs.io/en/stable/user/installing.html#version-support-information"  # noqa
        )[0]
        .dropna()
        .drop(columns=["Numba.1", "llvmlite", "LLVM", "TBB"])
        .query('`Python`.str.contains("2.7") == False')
        .query('`Numba`.str.contains(".x") == False')
        .query('`Numba`.str.contains("{") == False')
        .pipe(
            lambda df: df.assign(
                MIN_PYTHON_SPEC=(
                    df.Python.str.split().str[1].replace({"<": "="}, regex=True)
                    + df.Python.str.split().str[0].replace({".x": ""}, regex=True)
                ).apply(SpecifierSet)
            )
        )
        .pipe(
            lambda df: df.assign(
                MIN_NUMPY=(df.NumPy.str.split().str[0].replace({".x": ""}, regex=True))
            )
        )
        .assign(
            COMPATIBLE=lambda row: row.apply(
                check_python_compatibility, axis=1, args=(Version(min_python),)
            )
        )
        .query("COMPATIBLE == True")
        .pipe(lambda df: df.assign(MINOR=df.Numba.str.split(".").str[1]))
        .pipe(lambda df: df.assign(PATCH=df.Numba.str.split(".").str[2]))
        .sort_values(["MINOR", "PATCH"], ascending=[False, True])
        .iloc[-1]
    )
    return df.Numba, df.MIN_NUMPY


def check_python_compatibility(row, min_python):
    """
    Determine the Python version compatibility
    """
    python_compatible = min_python in (row.MIN_PYTHON_SPEC)
    return python_compatible


def check_scipy_compatibility(row, min_python, min_numpy):
    """
    Determine the Python and NumPy version compatibility
    """
    python_compatible = min_python in (row.MIN_PYTHON_SPEC & row.MAX_PYTHON_SPEC)
    numpy_compatible = min_numpy in (row.MIN_NUMPY_SPEC & row.MAX_NUMPY_SPEC)
    return python_compatible & numpy_compatible


def get_min_scipy_version(min_python, min_numpy):
    """
    Determine the SciPy version compatibility
    """
    colnames = pd.read_html(
        "https://docs.scipy.org/doc/scipy/dev/toolchain.html#numpy"
    )[1].columns
    converter = {colname: str for colname in colnames}
    df = (
        pd.read_html(
            "https://docs.scipy.org/doc/scipy/dev/toolchain.html#numpy",
            converters=converter,
        )[1]
        .rename(columns=lambda x: x.replace(" ", "_"))
        .replace({".x": ""}, regex=True)
        .pipe(
            lambda df: df.assign(
                SciPy_version=df.SciPy_version.str.replace(
                    r"\d\/", "", regex=True  # noqa
                )
            )
        )
        .query('`Python_versions`.str.contains("2.7") == False')
        .pipe(
            lambda df: df.assign(
                MIN_PYTHON_SPEC=df.Python_versions.str.split(",")
                .str[0]
                .apply(SpecifierSet)
            )
        )
        .pipe(
            lambda df: df.assign(
                MAX_PYTHON_SPEC=df.Python_versions.str.split(",")
                .str[1]
                .apply(SpecifierSet)
            )
        )
        .pipe(
            lambda df: df.assign(
                MIN_NUMPY_SPEC=df.NumPy_versions.str.split(",")
                .str[0]
                .apply(SpecifierSet)
            )
        )
        .pipe(
            lambda df: df.assign(
                MAX_NUMPY_SPEC=df.NumPy_versions.str.split(",")
                .str[1]
                .apply(SpecifierSet)
            )
        )
        .assign(
            COMPATIBLE=lambda row: row.apply(
                check_scipy_compatibility,
                axis=1,
                args=(Version(min_python), Version(min_numpy)),
            )
        )
        .query("COMPATIBLE == True")
        .pipe(lambda df: df.assign(MINOR=df.SciPy_version.str.split(".").str[1]))
        .pipe(lambda df: df.assign(PATCH=df.SciPy_version.str.split(".").str[2]))
        .sort_values(["MINOR", "PATCH"], ascending=[False, True])
        .iloc[-1]
    )
    return df.SciPy_version


def match_pkg_version(line, pkg_name):
    """
    Regular expression to match package versions
    """
    matches = re.search(
        rf"""
                        {pkg_name}  # Package name
                        [\s=><:"\'\[\]]*  # Zero or more spaces or special characters
                        (\d+\.\d+[\.0-9]*)  # Capture "version" in `matches`
                        """,
        line,
        re.VERBOSE | re.IGNORECASE,  # Ignores all whitespace and case in pattern
    )

    return matches


def find_pkg_mismatches(pkg_name, pkg_version, fnames):
    """
    Determine if any package version has mismatches
    """
    pkg_mismatches = []

    for fname in fnames:
        with open(fname, "r") as file:
            for line_num, line in enumerate(file, start=1):
                l = line.strip().replace(" ", "").lower()
                matches = match_pkg_version(l, pkg_name)
                if matches is not None:
                    version = matches.groups()[0]
                    if version != pkg_version:
                        pkg_mismatches.append((pkg_name, version, fname, line_num))

    return pkg_mismatches


def test_pkg_mismatch_regex():
    """
    Validation function for the package mismatch regex
    """
    pkgs = {
        "numpy": "0.0",
        "scipy": "0.0",
        "python": "2.7",
        "python-version": "2.7",
        "numba": "0.0",
    }

    lines = [
        "Programming Language :: Python :: 3.8",
        "STUMPY supports Python 3.8",
        "python-version: ['3.8']",
        'requires-python = ">=3.8"',
        "numba>=0.55.2",
    ]

    for line in lines:
        match_found = False
        for pkg_name, pkg_version in pkgs.items():
            matches = match_pkg_version(line, pkg_name)

            if matches:
                match_found = True
                break

        if not match_found:
            raise ValueError(f'Package mismatch regex fails to cover/match "{line}"')


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("min_python", nargs="?", default=None)
    args = parser.parse_args()

    if args.min_python is not None:
        MIN_PYTHON = str(args.min_python)
    else:
        MIN_PYTHON = get_min_python_version()
    MIN_NUMBA, MIN_NUMPY = get_min_numba_numpy_version(MIN_PYTHON)
    MIN_SCIPY = get_min_scipy_version(MIN_PYTHON, MIN_NUMPY)

    print(
        f"python: {MIN_PYTHON}\n"
        f"numba: {MIN_NUMBA}\n"
        f"numpy: {MIN_NUMPY}\n"
        f"scipy: {MIN_SCIPY}"
    )

    pkgs = {
        "numpy": MIN_NUMPY,
        "scipy": MIN_SCIPY,
        "numba": MIN_NUMBA,
        "python": MIN_PYTHON,
        "python-version": MIN_PYTHON,
    }

    fnames = [
        "pyproject.toml",
        "requirements.txt",
        "environment.yml",
        ".github/workflows/github-actions.yml",
        "README.rst",
    ]

    test_pkg_mismatch_regex()

    for pkg_name, pkg_version in pkgs.items():
        for name, version, fname, line_num in find_pkg_mismatches(
            pkg_name, pkg_version, fnames
        ):
            print(
                f"{pkg_name} {pkg_version} Mismatch: Version {version} "
                f"found in {fname}:{line_num}"
            )
