from setuptools import setup
import sys


def readme():
    with open("README.rst") as readme_file:
        return readme_file.read()


def get_extras_require():
    extras = [
        "pandas >= 0.20.0",
        "dask >= 1.2.2",
        "distributed >= 1.28.1",
        "coverage >= 4.5.3",
        "flake8 >= 3.7.7",
        "flake8-docstrings >= 1.5.0",
        "black >= 19.3b0",
        "pytest >= 4.4.1",
        "codecov",
    ]

    if "linux" in sys.platform:
        extras.append("tbb >= 2019.5")

    return extras


configuration = {
    "name": "stumpy",
    "version": "1.11.1",
    "python_requires=": ">=3.7",
    "author": "Sean M. Law",
    "author_email": "seanmylaw@gmail.com",
    "description": (
        "A powerful and scalable library that can be used "
        "for a variety of time series data mining tasks"
    ),
    "long_description_content_type": "text/x-rst",
    "long_description": readme(),
    "classifiers": [
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: BSD License",
        "Programming Language :: Python",
        "Topic :: Software Development",
        "Topic :: Scientific/Engineering",
        "Operating System :: Microsoft :: Windows",
        "Operating System :: POSIX",
        "Operating System :: Unix",
        "Operating System :: MacOS",
        "Programming Language :: Python :: 3.7",
    ],
    "keywords": "time series matrix profile motif discord",
    "url": "https://github.com/TDAmeritrade/stumpy",
    "maintainer": "Sean M. Law",
    "maintainer_email": "seanmylaw@gmail.com",
    "license": "3-clause BSD License",
    "packages": ["stumpy"],
    "install_requires": ["numpy >= 1.17", "scipy >= 1.5", "numba >= 0.54"],
    "ext_modules": [],
    "cmdclass": {},
    "tests_require": ["pytest"],
    "data_files": (),
    "extras_require": {"ci": get_extras_require()},
}

setup(**configuration)
