import ast
import importlib
import pathlib

from stumpy import config


def get_default_fastmath(module_name):
    """
    Retrieve a dictionary where key is njit function name
    and value is the default fastmath flag.

    Parameters
    ----------
    module_name : str
        The module name

    Returns
    -------
    out : dict
        A dictionary of njit functions, where key is the function name
        and value is the fastmath flag.
    """
    filepath = pathlib.Path(__file__).parent / f"{module_name}.py"
    file_contents = ""
    with open(filepath, encoding="utf8") as f:
        file_contents = f.read()

    out = {}
    module = ast.parse(file_contents)
    for node in module.body:
        if not isinstance(node, ast.FunctionDef):
            continue

        func_name = node.name
        for decorator in node.decorator_list:
            if not isinstance(decorator, ast.Call) or not isinstance(
                decorator.func, ast.Name
            ):
                continue

            if decorator.func.id != "njit":
                continue

            fastmath_default = None
            for item in decorator.keywords:
                if item.arg == "fastmath":
                    config_var = item.value.attr
                    fastmath_default = config._STUMPY_DEFAULTS[config_var]
                    break

            out[func_name] = fastmath_default

    return out


def set_flag(module_name, func_name, flag=None):
    """
    Set a flag for a given function

    Parameters
    ----------
    module_name : str
        The module name
    func_name : str
        The function name
    flag : str
        The flag to set

    Returns
    -------
    None
    """
    funcs_flags = get_default_fastmath(module_name)
    if func_name not in funcs_flags.keys():
        msg = f"The module `{module_name}` does not have a njit function `{func_name}`"
        raise ValueError(msg)
    default_flag = funcs_flags[func_name]

    if flag is None:
        flag = default_flag

    if flag is not None:
        module = importlib.import_module(f".{module_name}", package="stumpy")
        func = getattr(module, func_name)
        func.targetoptions["fastmath"] = flag
        func.recompile()

    return
