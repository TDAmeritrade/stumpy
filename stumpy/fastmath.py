import importlib

from stumpy import config


def _set(module_name, func_name, flag):
    """
    Set fastmath flag for a given function

    Parameters
    ----------
    module_name : str
        The module name
    func_name : str
        The function name
    flag : set or bool
        The fastmath flag

    Returns
    -------
    None
    """
    module = importlib.import_module(f".{module_name}", package="stumpy")
    func = getattr(module, func_name)
    func.targetoptions["fastmath"] = flag
    func.recompile()

    return


def _reset(module_name, func_name):
    """
    Reset the value of fastmath to its default value

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
    _set(module_name, func_name, default_flag)

    return
