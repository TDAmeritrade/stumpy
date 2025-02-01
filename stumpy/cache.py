# STUMPY
# Copyright 2019 TD Ameritrade. Released under the terms of the 3-Clause BSD license.
# STUMPY is a trademark of TD Ameritrade IP Company, Inc. All rights reserved.

import ast
import importlib
import inspect
import pathlib
import site
import warnings

import numba

CACHE_WARNING = "Caching `numba` functions is purely for experimental purposes "
CACHE_WARNING += "and should never be used or depended upon as it is not supported! "
CACHE_WARNING += "All caching capabilities are not tested and may be removed/changed "
CACHE_WARNING += "without prior notice. Please proceed with caution!"
CACHE_CLEARED = True


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
    cache_dir : str, default None
        The path to the numba cache directory

    Returns
    -------
    None
    """
    global CACHE_CLEARED

    if cache_dir is not None:
        numba_cache_dir = str(cache_dir)
    else:  # pragma: no cover
        site_pkg_dir = site.getsitepackages()[0]
        numba_cache_dir = site_pkg_dir + "/stumpy/__pycache__"

    [f.unlink() for f in pathlib.Path(numba_cache_dir).glob("*nb*") if f.is_file()]

    CACHE_CLEARED = True


def clear(cache_dir=None):
    """
    Clear numba cache directory

    Parameters
    ----------
    cache_dir : str, default None
        The path to the numba cache directory. When `cache_dir` is `None`, then this
        defaults to `site-packages/stumpy/__pycache__`.

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
    if cache_dir is not None:
        numba_cache_dir = str(cache_dir)
    else:  # pragma: no cover
        site_pkg_dir = site.getsitepackages()[0]
        numba_cache_dir = site_pkg_dir + "/stumpy/__pycache__"

    return [
        f"{numba_cache_dir}/{f.name}"
        for f in pathlib.Path(numba_cache_dir).glob("*nb*")
        if f.is_file()
    ]


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
    global CACHE_CLEARED

    if not CACHE_CLEARED:  # pragma: no cover
        msg = "Numba njit cached files are  not cleared before saving/overwriting. "
        msg = "You may need to call `cache.clear()` before calling `cache.save()`."
        warnings.warn(msg)

    _enable()
    _recompile()

    CACHE_CLEARED = False

    return


def save():
    """
    Save/overwrite all of the cached njit functions.

    Parameters
    ----------
    None

    Returns
    -------
    None

    Notes
    -----
    The cache is never cleared before saving/overwriting and may be explicitly cleared
    by calling `cache.clear()` before saving. It is best practice to call `cache.save()`
    only after calling all of your `njit` functions. If `cache.save()` is called for the
    first time (before any `njit` function is called) then only the `.nbi` files (i.e.,
    the "cache index") for all `njit` functions are saved. As each `njit` function (and
    sub-functions) is called then their corresponding `.nbc` file (i.e., "object code")
    is saved. Each `.nbc` file will only be saved after its `njit` function is called
    at least once. However, subsequent calls to `cache.save()` (after clearing the cache
    via `cache.clear()`) will automatically save BOTH the `.nbi` files as well as the
    `.nbc` files as long as their `njit` function has been called at least once.

    Examples
    --------
    >>> import stumpy
    >>> from stumpy import cache
    >>> import numpy as np
    >>> cache.clear()
    >>> mp = stumpy.stump(np.array([584., -11., 23., 79., 1001., 0., -19.]), m=3)
    >>> cache.save()
    """
    if numba.config.DISABLE_JIT:
        msg = "Could not save/cache function because NUMBA JIT is disabled"
        warnings.warn(msg)
    else:  # pragma: no cover
        warnings.warn(CACHE_WARNING)

    if numba.config.CACHE_DIR != "":  # pragma: no cover
        msg = "Found user specified `NUMBA_CACHE_DIR`/`numba.config.CACHE_DIR`. "
        msg += "The `stumpy` cache files may not be saved/cleared correctly!"
        warnings.warn(msg)

    _save()

    return
