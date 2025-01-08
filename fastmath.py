#!/usr/bin/env python

import argparse
import importlib
import pathlib
import re


def get_njit_funcs():
    """
    Retrieve a list of all njit functions

    Parameters
    ----------
    None

    Returns
    -------
    njit_funcs : list
        A list of all njit functions, where each element is a tuple of the form
        (module_name, func_name)
    """
    pattern = r"@njit.*?def\s+\w+\("

    stumpy_path = pathlib.Path(__file__).parent / "stumpy"
    filepaths = sorted(f for f in pathlib.Path(stumpy_path).iterdir() if f.is_file())

    out = []
    ignore = ["__init__.py", "__pycache__"]
    for filepath in filepaths:
        fname = filepath.name
        if fname not in ignore and fname.endswith(".py"):
            file_contents = ""
            with open(filepath, encoding="utf8") as f:
                file_contents = f.read()

            matches = re.findall(pattern, file_contents, re.DOTALL)
            for match in matches:
                func_name = match.split("def ")[-1].split("(")[0]
                out.append((fname.removesuffix(".py"), func_name))

    return out


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
