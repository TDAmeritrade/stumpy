#!/usr/bin/env python

import ast
import pathlib
import re


def get_docstring_args(fd, file_name, func_name, class_name=None):
    """
    Extract docstring parameters from function definition
    """
    docstring = ast.get_docstring(fd)
    if len(re.findall(r"Parameters", docstring)) != 1:
        msg = "Missing required 'Parameters' section in docstring in \n"
        msg += f"file: {file_name}\n"
        if class_name is not None:
            msg += f"class: {class_name}\n"
        msg += f"function/method: {func_name}\n"
        raise RuntimeError(msg)
    if class_name is None and len(re.findall(r"Returns", docstring)) != 1:
        msg = "Missing required 'Returns' section in docstring in \n"
        msg += f"file: {file_name}\n"
        msg += f"function/method: {func_name}\n"
        raise RuntimeError(msg)

    if class_name is None:
        params_section = re.findall(
            r"(?<=Parameters)(.*)(?=Returns)", docstring, re.DOTALL
        )[0]
    else:
        params_section = re.findall(r"(?<=Parameters)(.*)", docstring, re.DOTALL)[0]

    args = re.findall(r"(\w+)\s+\:", params_section)
    args = set([a for a in args if a != "i"])  # `i` should never be a parameter

    return args


def get_signature_args(fd):
    """
    Extract signature arguments from function definition
    """
    return set([a.arg for a in fd.args.args if a.arg != "self"])


def check_args(doc_args, sig_args, file_name, func_name, class_name=None):
    """
    Compare docstring arguments and signature argments
    """
    diff_args = signature_args.difference(docstring_args)
    if len(diff_args) > 0:
        msg = "Found one or more arguments/parameters with missing docstring in \n"
        msg += f"file: {file_name}\n"
        if class_name is not None:
            msg += f"class: {class_name}\n"
        msg += f"function/method: {func_name}\n"
        msg += f"parameter(s): {diff_args}\n"
        raise RuntimeError(msg)

    diff_args = docstring_args.difference(signature_args)
    if len(diff_args) > 0:
        msg = "Found one or more unsupported arguments/parameters with docstring in \n"
        msg += f"file: {file_name}\n"
        if class_name is not None:
            msg += f"class: {class_name}\n"
        msg += f"function/method: {func_name}\n"
        msg += f"parameter(s): {diff_args}\n"
        raise RuntimeError(msg)


ignore = ["__init__.py", "__pycache__"]

stumpy_path = pathlib.Path(__file__).parent / "stumpy"
filepaths = sorted(f for f in pathlib.Path(stumpy_path).iterdir() if f.is_file())
for filepath in filepaths:
    if filepath.name not in ignore and str(filepath).endswith(".py"):
        file_contents = ""
        with open(filepath, encoding="utf8") as f:
            file_contents = f.read()
        module = ast.parse(file_contents)

        # Check Functions
        function_definitions = [
            node for node in module.body if isinstance(node, ast.FunctionDef)
        ]
        for fd in function_definitions:
            docstring_args = get_docstring_args(fd, filepath.name, fd.name)
            signature_args = get_signature_args(fd)
            check_args(docstring_args, signature_args, filepath.name, fd.name)

        # Check Class Methods
        class_definitions = [
            node for node in module.body if isinstance(node, ast.ClassDef)
        ]
        for cd in class_definitions:
            methods = [node for node in cd.body if isinstance(node, ast.FunctionDef)]
            for fd in methods:
                docstring_args = get_docstring_args(fd, filepath.name, fd.name, cd.name)
                signature_args = get_signature_args(fd)
                check_args(
                    docstring_args, signature_args, filepath.name, fd.name, cd.name
                )
