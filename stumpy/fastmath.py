import ast
import importlib
import pathlib

from stumpy import config


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


def _reset(module_name, func_name):
    """
    Reset the value of fastmath its default value

    Parameters
    ----------
    module_name : str
        The module name
    func_name : str
        The function name

    Returns
    -------
    None
    """
    key = module_name + "." + func_name
    key = "STUMPY_FASTMATH_" + key.upper()
    default_flag = config._STUMPY_DEFAULTS[key]
    set_flag(module_name, func_name, default_flag)

    return
