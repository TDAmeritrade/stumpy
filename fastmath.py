#!/usr/bin/env python

import argparse
import importlib

from stumpy.cache import get_njit_funcs


def check_fastmath():
    """
    Check if all njit functions have the `fastmath` flag set

    Parameters
    ----------
    None

    Returns
    -------
    None
    """
    missing_fastmath = []  # list of njit functions with missing fastmath flags
    for module_name, func_name in get_njit_funcs():
        module = importlib.import_module(f".{module_name}", package="stumpy")
        func = getattr(module, func_name)
        if "fastmath" not in func.targetoptions.keys():
            missing_fastmath.append(f"{module_name}.{func_name}")

    if len(missing_fastmath) > 0:
        msg = "Found one or more functions that are missing the `fastmath` flag. "
        msg += f"The function(s) are:\n {missing_fastmath}\n"
        raise ValueError(msg)

    return


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()

    if args.check:
        check_fastmath()
