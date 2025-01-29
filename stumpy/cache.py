# STUMPY
# Copyright 2019 TD Ameritrade. Released under the terms of the 3-Clause BSD license.
# STUMPY is a trademark of TD Ameritrade IP Company, Inc. All rights reserved.

import ast
import importlib
import inspect
import os
import pathlib
import site
import warnings

import numba

CACHE_WARNING = "Caching `numba` functions is purely for experimental purposes "
CACHE_WARNING += "and should never be used or depended upon as it is not supported! "
CACHE_WARNING += "All caching capabilities are not tested and may be removed/changed "
CACHE_WARNING += "without prior notice. Please proceed with caution!"


def get_njit_funcs():
    """
    Identify all njit functions

    Parameters
    ----------
    None

    Returns
    -------
    out : list
        A list of (`module_name`, `func_name`) pairs
    """
    ignore_py_files = ["__init__", "__pycache__"]

    pkg_dir = pathlib.Path(__file__).parent
    module_names = []
    for fname in pkg_dir.iterdir():
        if fname.stem not in ignore_py_files and not fname.stem.startswith("."):
            module_names.append(fname.stem)

    njit_funcs = []
    for module_name in module_names:
        filepath = pkg_dir / f"{module_name}.py"
        file_contents = ""
        with open(filepath, encoding="utf8") as f:
            file_contents = f.read()
        module = ast.parse(file_contents)
        for node in module.body:
            if isinstance(node, ast.FunctionDef):
                func_name = node.name
                for decorator in node.decorator_list:
                    decorator_name = None
                    if isinstance(decorator, ast.Name):  # pragma: no cover
                        # Bare decorator
                        decorator_name = decorator.id
                    if isinstance(decorator, ast.Call) and isinstance(
                        decorator.func, ast.Name
                    ):
                        # Decorator is a function
                        decorator_name = decorator.func.id

                    if decorator_name == "njit":
                        njit_funcs.append((module_name, func_name))

    return njit_funcs


def _enable():
    """
    Enable numba caching

    Parameters
    ----------
    None

    Returns
    -------
    None
    """
    frame = inspect.currentframe()
    caller_name = inspect.getouterframes(frame)[1].function
    if caller_name != "_save":  # pragma: no cover
        msg = (
            "The 'cache._enable()' function is deprecated and no longer supported. "
            + "Please use 'cache.save()' instead"
        )
        warnings.warn(msg, DeprecationWarning, stacklevel=2)

    njit_funcs = get_njit_funcs()
    for module_name, func_name in njit_funcs:
        module = importlib.import_module(f".{module_name}", package="stumpy")
        func = getattr(module, func_name)
        try:
            func.enable_caching()
        except AttributeError as e:
            if (
                numba.config.DISABLE_JIT
                and str(e) == "'function' object has no attribute 'enable_caching'"
            ):
                pass
            else:  # pragma: no cover
                raise


def _clear(cache_dir=None):
    """
    Clear numba cache

    Parameters
    ----------
    cache_dir : str
        The path to the numba cache directory

    Returns
    -------
    None
    """
    if cache_dir is not None:  # pragma: no cover
        numba_cache_dir = str(cache_dir)
    elif "PYTEST_CURRENT_TEST" in os.environ:
        numba_cache_dir = "stumpy/__pycache__"
    else:  # pragma: no cover
        site_pkg_dir = site.getsitepackages()[0]
        numba_cache_dir = site_pkg_dir + "/stumpy/__pycache__"

    [f.unlink() for f in pathlib.Path(numba_cache_dir).glob("*nb*") if f.is_file()]


def clear(cache_dir=None):
    """
    Clear numba cache directory

    Parameters
    ----------
    cache_dir : str
        The path to the numba cache directory

    Returns
    -------
    None
    """
    warnings.warn(CACHE_WARNING)
    _clear(cache_dir)

    return


def _get_cache(cache_dir=None):
    """
    Retrieve a list of cached numba functions

    Parameters
    ----------
    cache_dir : str
        The path to the numba cache directory

    Returns
    -------
    out : list
        A list of cached numba functions
    """
    warnings.warn(CACHE_WARNING)
    if cache_dir is not None:  # pragma: no cover
        numba_cache_dir = str(cache_dir)
    if "PYTEST_CURRENT_TEST" in os.environ:
        numba_cache_dir = "stumpy/__pycache__"
    else:  # pragma: no cover
        site_pkg_dir = site.getsitepackages()[0]
        numba_cache_dir = site_pkg_dir + "/stumpy/__pycache__"
    return [f.name for f in pathlib.Path(numba_cache_dir).glob("*nb*") if f.is_file()]


def _recompile():
    """
    Recompile all njit functions

    Parameters
    ----------
    None

    Returns
    -------
    None

    Notes
    -----
    If the `numba` cache is enabled, this results in saving (and/or overwriting)
    the cached numba functions to disk.
    """
    for module_name, func_name in get_njit_funcs():
        module = importlib.import_module(f".{module_name}", package="stumpy")
        func = getattr(module, func_name)
        try:
            func.recompile()
        except AttributeError as e:
            if (
                numba.config.DISABLE_JIT
                and str(e) == "'function' object has no attribute 'recompile'"
            ):
                pass
            else:  # pragma: no cover
                raise

    return


def _save():
    """
    Save all njit functions

    Parameters
    ----------
    None

    Returns
    -------
    None
    """
    _enable()
    _clear()
    _recompile()

    return


def save():
    """
    Save/overwrite all the cache data files of
    all-so-far compiled njit functions.

    Parameters
    ----------
    None

    Returns
    -------
    None
    """
    if numba.config.DISABLE_JIT:
        msg = "Could not save/cache function because NUMBA JIT is disabled"
        warnings.warn(msg)
    else:  # pragma: no cover
        warnings.warn(CACHE_WARNING)

    _save()

    return
