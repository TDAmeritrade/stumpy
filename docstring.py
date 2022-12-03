#!/usr/bin/env python

import pathlib
import ast
import re


def get_docstring_args(docstring):
    """
    Extract parameter arguments from docstring
    """
    args = re.findall(r"(\w+)\s+\:", docstring)
    args = [a for a in args if a != "self"]
    return args


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
            docstring_args = set(get_docstring_args(ast.get_docstring(fd)))
            signature_args = set([a.arg for a in fd.args.args])
            diff_args = signature_args.difference(docstring_args)
            if len(diff_args) > 0:
                print("Found one or more parameters with missing docstring:")
                print(f"    File: {filepath.name}")
                print(f"    Function: {fd.name}")
                print(f"    Parameters: {diff_args}")
                # print(ast.get_docstring(fd))
                # print(docstring_args)
                # print(signature_args)

        # Check Class Methods
        class_definitions = [
            node for node in module.body if isinstance(node, ast.ClassDef)
        ]
        for cd in class_definitions:
            methods = [node for node in cd.body if isinstance(node, ast.FunctionDef)]
            for fd in methods:
                docstring_args = set(get_docstring_args(ast.get_docstring(fd)))
                signature_args = set([a.arg for a in fd.args.args if a.arg != "self"])
                diff_args = signature_args.difference(docstring_args)
                if len(diff_args) > 0:
                    print("Found one or more parameters with missing docstring:")
                    print(f"    File: {filepath.name}")
                    print(f"    Class: {cd.name}")
                    print(f"    Method: {fd.name}")
                    print(f"    Parameters: {diff_args}")
                    # print(ast.get_docstring(fd))
                    # print(docstring_args)
                    # print(signature_args)
