import ast
import pkgutil
import pathlib
import warnings
import importlib
import site


CACHE_WARNING = "Caching `numba` functions is purely for experimental purposes "
CACHE_WARNING += "and should never be used or depended upon as it is not supported! "
CACHE_WARNING += "All caching capabilities are not tested and may be removed/changed "
CACHE_WARNING += "without prior notice. Please proceed with caution!"


def get_njit_funcs():
    """
    Identify all njit functions

    Returns
    -------
    out : list
        A list of (`module_name`, `func_name`) pairs
    """
    pkg_dir = pathlib.Path(__file__).parent
    module_names = [name for _, name, _ in pkgutil.iter_modules([str(pkg_dir)])]

    njit_funcs = []

    for module_name in module_names:
        filepath = pathlib.Path(__file__).parent / f"{module_name}.py"
        file_contents = ""
        with open(filepath, encoding="utf8") as f:
            file_contents = f.read()
        module = ast.parse(file_contents)
        for node in module.body:
            if isinstance(node, ast.FunctionDef):
                func_name = node.name
                for decorator in node.decorator_list:
                    if isinstance(decorator, ast.Call) and isinstance(
                        decorator.func, ast.Name
                    ):
                        if decorator.func.id == "njit":
                            njit_funcs.append((module_name, func_name))

    return njit_funcs


def _enable():
    """
    Enable numba caching

    Returns
    -------
    None
    """
    warnings.warn(CACHE_WARNING)
    njit_funcs = get_njit_funcs()
    for module_name, func_name in njit_funcs:
        module = importlib.import_module(f".{module_name}", package="stumpy")
        func = getattr(module, func_name)
        func.enable_caching()


def _clear():
    """
    Clear numba cache

    Returns
    -------
    None
    """
    warnings.warn(CACHE_WARNING)
    site_pkg_dir = site.getsitepackages()[0]
    numba_cache_dir = site_pkg_dir + "/stumpy/__pycache__"
    [f.unlink() for f in pathlib.Path(numba_cache_dir).glob("*nb*") if f.is_file()]


def _get_cache():
    """
    Retrieve a list of cached numba functions

    Returns
    -------
    out : list
        A list of cached numba functions
    """
    warnings.warn(CACHE_WARNING)
    site_pkg_dir = site.getsitepackages()[0]
    numba_cache_dir = site_pkg_dir + "/stumpy/__pycache__"
    return [f.name for f in pathlib.Path(numba_cache_dir).glob("*nb*") if f.is_file()]
