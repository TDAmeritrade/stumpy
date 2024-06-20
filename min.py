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
    df = (
        pd.read_html("https://docs.scipy.org/doc/scipy/dev/toolchain.html#numpy")[1]
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


def find_pkg_mismatches(
    fnames=[
        "pyproject.toml",
        "requirements.txt",
        "environment.yml",
        ".github/workflows/github-actions.yml",
        "README.rst",
    ],
    pkg="python",
    version=get_min_python_version(),
):

    # Forcing pkg to lower
    pkg = pkg.lower()

    # Instantiating a dictionary of lists for storing
    # any hits on minimum version mismatches while scanning
    pkg_mismatches = []

    # Iteratively scanning files for minimum version mismatches
    for fname in fnames:
        with open(fname, "r") as file:

            # Iteratively capturing the line number
            # associated with each minimum version mismatch
            for line_num, line in enumerate(file, start=1):

                # Handling different version listing formats
                if (
                    f"{pkg}-version: [" in line
                    or re.search(
                        rf"({pkg}|{pkg})\s*==?\s*['\"]?(\d+\.\d+(\.\d+)?)['\"]?",
                        line,
                        re.IGNORECASE,
                    )
                    or re.search(
                        rf"({pkg})\s*>=\s*['\"]?(\d+\.\d+(\.\d+)?)['\"]?",
                        line,
                        re.IGNORECASE,
                    )
                    or re.search(
                        rf"::\s*{pkg}\s*::\s*['\"]?(\d+\.\d+(\.\d+)?)['\"]?",
                        line,
                        re.IGNORECASE,
                    )
                    or re.search(
                        rf"{pkg}\s+[a-z]+\s+(\d+\.\d+(\.\d+)?)", line, re.IGNORECASE
                    )
                ):
                    # Extracting version information
                    versions = re.findall(r"'([\d.]+)'|(\d+\.\d+(\.\d+)?)", line)
                    for version_tuple in versions:
                        version_listed = next((v for v in version_tuple if v), None)
                        version_tuple = tuple(map(int, version_listed.split(".")))
                        min_version_tuple = tuple(map(int, version.split(".")))

                        # If the version listed is less than the minimum
                        # supported version then append to the mismatch
                        # list of lists
                        if version_tuple < min_version_tuple:
                            pkg_mismatches.append(
                                [pkg, fname, line_num, version_listed]
                            )

                # Handling if line structured as "STUMPY supports ..."
                elif "STUMPY supports `Python" in line:
                    # Extracting Python version from the line
                    python_version = re.search(r"(\d+\.\d+(\.\d+)?)", line)
                    if python_version:
                        version_listed = tuple(
                            map(int, python_version.group(1).split("."))
                        )
                        min_version = tuple(map(int, version.split(".")))

                        # If the version listed is less than the minimum
                        # supported version then append to the mismatch
                        # list of lists
                        if version_listed < min_version:
                            pkg_mismatches.append(
                                [pkg, fname, line_num, version_listed]
                            )

    return pkg_mismatches


def scan_files_for_minimum_versions(package_versions):
    pkg_mismatches = []

    for pkg, version in package_versions.items():
        mismatch_results = find_pkg_mismatches(pkg=pkg, version=version)
        pkg_mismatches.extend(mismatch_results)

    return pkg_mismatches


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

    # Instantiating a list of files to scan for minimum version mismatches
    fnames = [
        "pyproject.toml",
        "requirements.txt",
        "environment.yml",
        ".github/workflows/github-actions.yml",
        "README.rst",
    ]

    # Instantiating a dictionary of minimum versions to
    # scan the list 'files_to_scan' for
    package_versions = {
        "python": MIN_PYTHON,
        "numba": MIN_NUMBA,
        "numpy": MIN_NUMPY,
        "scipy": MIN_SCIPY,
    }

    # Printing a dictionary of lists containing
    # any minimum version mismatches identified
    print(scan_files_for_minimum_versions(package_versions))
