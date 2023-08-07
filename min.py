#!/usr/bin/env python

import pandas as pd
from packaging.specifiers import SpecifierSet
from packaging.version import Version


def check_compatibility(row, min_python, min_numpy):
    """
    Determines the Python and NumPy version compatibility
    """
    python_compatible = min_python in (row.MIN_PYTHON_SPEC & row.MAX_PYTHON_SPEC)
    numpy_compatible = min_numpy in (row.MIN_NUMPY_SPEC & row.MAX_NUMPY_SPEC)
    return python_compatible & numpy_compatible


MIN_PYTHON = "3.8"  # Change this

df = (
    pd.read_html(
        "https://numba.readthedocs.io/en/stable/user/installing.html#version-support-information"  # noqa
    )[0]
    .dropna()
    .query(f'Python.str.startswith("{MIN_PYTHON}")', engine="python")
    .pipe(lambda df: df.assign(NumPy=df.NumPy.str.split().str[0]))
    .iloc[-1]
)
MIN_NUMBA = df.Numba
MIN_NUMPY = df.NumPy

df = (
    pd.read_html("https://docs.scipy.org/doc/scipy/dev/toolchain.html#numpy")[1]
    .replace({".x": ""}, regex=True)
    .rename(columns=lambda x: x.replace(" ", "_"))
    .query('`Python_versions`.str.contains("2.7") == False')
    .pipe(
        lambda df: df.assign(
            MIN_PYTHON_SPEC=df.Python_versions.str.split(",").str[0].apply(SpecifierSet)
        )
    )
    .pipe(
        lambda df: df.assign(
            MAX_PYTHON_SPEC=df.Python_versions.str.split(",").str[1].apply(SpecifierSet)
        )
    )
    .pipe(
        lambda df: df.assign(
            MIN_NUMPY_SPEC=df.NumPy_versions.str.split(",").str[0].apply(SpecifierSet)
        )
    )
    .pipe(
        lambda df: df.assign(
            MAX_NUMPY_SPEC=df.NumPy_versions.str.split(",").str[1].apply(SpecifierSet)
        )
    )
    .assign(
        COMPATIBLE=lambda row: row.apply(
            check_compatibility, axis=1, args=(Version(MIN_PYTHON), Version(MIN_NUMPY))
        )
    )
    .query("COMPATIBLE == True")
    .iloc[-1]
)
MIN_SCIPY = df.SciPy_version

print(
    f"python: {MIN_PYTHON}\nnumba: {MIN_NUMBA}\nnumpy: {MIN_NUMPY}\nscipy: {MIN_SCIPY}"
)
