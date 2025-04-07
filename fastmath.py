#!/usr/bin/env python

import argparse
import ast
import importlib
import pathlib


def get_njit_funcs(pkg_dir):
    """
    Identify all njit functions

    Parameters
    ----------
    pkg_dir : str
        The path to the directory containing some .py files

    Returns
    -------
    njit_funcs : list
        A list of all njit functions, where each element is a tuple of the form
        (module_name, func_name)
    """
    ignore_py_files = ["__init__", "__pycache__"]
    pkg_dir = pathlib.Path(pkg_dir)

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
                    if isinstance(decorator, ast.Name):
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


def check_fastmath(pkg_dir, pkg_name):
    """
    Check if all njit functions have the `fastmath` flag set

    Parameters
    ----------
    pkg_dir : str
        The path to the directory containing some .py files

    pkg_name : str
        The name of the package

    Returns
    -------
    None
    """
    missing_fastmath = []  # list of njit functions with missing fastmath flags
    for module_name, func_name in get_njit_funcs(pkg_dir):
        module = importlib.import_module(f".{module_name}", package=pkg_name)
        func = getattr(module, func_name)
        if "fastmath" not in func.targetoptions.keys():
            missing_fastmath.append(f"{module_name}.{func_name}")

    if len(missing_fastmath) > 0:
        msg = (
            "Found one or more `@njit` functions that are missing the `fastmath` flag. "
            + f"The functions are:\n {missing_fastmath}\n"
        )
        raise ValueError(msg)

    return


class FunctionCallVisitor(ast.NodeVisitor):
    """
    A class to traverse the AST of the modules of a package to collect
    the call stacks of njit functions.

    Parameters
    ----------
    pkg_dir : str
        The path to the package directory containing some .py files.

    pkg_name : str
        The name of the package.

    Attributes
    ----------
    module_names : list
        A list of module names to track the modules as the visitor traverses them.

    call_stack : list
        A list of njit functions, representing a chain of function calls,
        where each element is a string of the form "module_name.func_name".

    out : list
        A list of unique `call_stack`s.

    njit_funcs : list
        A list of all njit functions in `pkg_dir`'s modules. Each element is a tuple
        of the form `(module_name, func_name)`.

    njit_modules : set
        A set that contains the names of all modules, each of which contains at least
        one njit function.

    njit_nodes : dict
        A dictionary mapping njit function names to their corresponding AST nodes.
        A key is a string, and it is of the form "module_name.func_name", and its
        corresponding value is the AST node- with type ast.FunctionDef- of that
        function.

    ast_modules : dict
        A dictionary mapping module names to their corresponding AST objects. A key
        is the name of a module, and its corresponding value is the content of that
        module as an AST object.

    Methods
    -------
    push_module(module_name)
        Push the name of a module onto the stack `module_names`.

    pop_module()
        Pop the last module name from the stack `module_names`.

    push_call_stack(module_name, func_name)
        Push a function call onto the stack of function calls, `call_stack`.

    pop_call_stack()
        Pop the last function call from the stack of function calls, `call_stack`

    goto_deeper_func(node)
        Calls the visit method from class `ast.NodeVisitor` on all children of
        the `node`.

    goto_next_func(node)
        Calls the visit method from class `ast.NodeVisitor` on all children of
        the `node`.

    push_out()
        Push the current function call stack, `call_stack`, onto the output list, `out`,
        unless it is already included in one of the so-far-collected call stacks.

    visit_Call(node)
        This method is called when the visitor encounters a function call in the AST. It
        checks if the called function is a njit function and, if so, traverses its AST
        to collect its call stack.
    """

    def __init__(self, pkg_dir, pkg_name):
        """
        Initialize the FunctionCallVisitor class.  This method sets up the necessary
        attributes and prepares the visitor for traversing the AST of STUMPY's modules.

        Parameters
        ----------
        pkg_dir : str
            The path to the package directory containing some .py files.

        pkg_name : str
            The name of the package.

        Returns
        -------
        None
        """
        super().__init__()
        self.module_names = []
        self.call_stack = []
        self.out = []

        # Setup lists, dicts, and ast objects
        self.njit_funcs = get_njit_funcs(pkg_dir)
        self.njit_modules = set(mod_name for mod_name, func_name in self.njit_funcs)
        self.njit_nodes = {}
        self.ast_modules = {}

        filepaths = sorted(f for f in pathlib.Path(pkg_dir).iterdir() if f.is_file())
        ignore = ["__init__.py", "__pycache__"]

        for filepath in filepaths:
            file_name = filepath.name
            if (
                file_name not in ignore
                and not file_name.startswith("gpu")
                and str(filepath).endswith(".py")
            ):
                module_name = file_name.replace(".py", "")
                file_contents = ""
                with open(filepath, encoding="utf8") as f:
                    file_contents = f.read()
                self.ast_modules[module_name] = ast.parse(file_contents)

                for node in self.ast_modules[module_name].body:
                    if isinstance(node, ast.FunctionDef):
                        func_name = node.name
                        if (module_name, func_name) in self.njit_funcs:
                            self.njit_nodes[f"{module_name}.{func_name}"] = node

    def push_module(self, module_name):
        """
        Push a module name onto the stack of module names.

        Parameters
        ----------
        module_name : str
            The name of the module to be pushed onto the stack.

        Returns
        -------
        None
        """
        self.module_names.append(module_name)

        return

    def pop_module(self):
        """
        Pop the last module name from the stack of module names.

        Parameters
        ----------
        None

        Returns
        -------
        None
        """
        if self.module_names:
            self.module_names.pop()

        return

    def push_call_stack(self, module_name, func_name):
        """
        Push a function call onto the stack of function calls.

        Parameters
        ----------
        module_name : str
            A module's name

        func_name : str
            A function's name

        Returns
        -------
        None
        """
        self.call_stack.append(f"{module_name}.{func_name}")

        return

    def pop_call_stack(self):
        """
        Pop the last function call from the stack of function calls.

        Parameters
        ----------
        None

        Returns
        -------
        None
        """
        if self.call_stack:
            self.call_stack.pop()

        return

    def goto_deeper_func(self, node):
        """
        Calls the visit method from class `ast.NodeVisitor` on
        all children of the `node`.

        Parameters
        ----------
        node : ast.AST
            The AST node to be visited.

        Returns
        -------
        None
        """
        self.generic_visit(node)

        return

    def goto_next_func(self, node):
        """
        Calls the visit method from class `ast.NodeVisitor` on
        all children of the node.

        Parameters
        ----------
        node : ast.AST
            The AST node to be visited.

        Returns
        -------
        None
        """
        self.generic_visit(node)

        return

    def push_out(self):
        """
        Push the current function call stack onto the output list unless it
        is already included in one of the so-far-collected call stacks.


        Parameters
        ----------
        None

        Returns
        -------
        None
        """
        unique = True
        for cs in self.out:
            if " ".join(self.call_stack) in " ".join(cs):
                unique = False
                break

        if unique:
            self.out.append(self.call_stack.copy())

        return

    def visit_Call(self, node):
        """
        Called when visiting an AST node of type `ast.Call`.

        Parameters
        ----------
        node : ast.Call
            The AST node representing a function call.

        Returns
        -------
        None
        """
        callee_name = ast.unparse(node.func)

        module_changed = False
        if "." in callee_name:
            new_module_name, new_func_name = callee_name.split(".")[:2]

            if new_module_name in self.njit_modules:
                self.push_module(new_module_name)
                module_changed = True
        else:
            if self.module_names:
                new_module_name = self.module_names[-1]
                new_func_name = callee_name
                callee_name = f"{new_module_name}.{new_func_name}"

        if callee_name in self.njit_nodes.keys():
            callee_node = self.njit_nodes[callee_name]
            self.push_call_stack(new_module_name, new_func_name)
            self.goto_deeper_func(callee_node)
            self.push_out()
            self.pop_call_stack()
            if module_changed:
                self.pop_module()

        self.goto_next_func(node)

        return


def get_njit_call_stacks(pkg_dir, pkg_name):
    """
    Get the call stacks of all njit functions in `pkg_dir`

    Parameters
    ----------
    pkg_dir : str
        The path to the package directory containing some .py files

    pkg_name : str
        The name of the package

    Returns
    -------
    out : list
        A list of unique function call stacks. Each item is of type list,
        representing a chain of function calls.
    """
    visitor = FunctionCallVisitor(pkg_dir, pkg_name)

    for module_name in visitor.njit_modules:
        visitor.push_module(module_name)

        for node in visitor.ast_modules[module_name].body:
            if isinstance(node, ast.FunctionDef):
                func_name = node.name
                if (module_name, func_name) in visitor.njit_funcs:
                    visitor.push_call_stack(module_name, func_name)
                    visitor.visit(node)
                    visitor.pop_call_stack()

        visitor.pop_module()

    return visitor.out


def check_call_stack_fastmath(pkg_dir, pkg_name):
    """
    Check if all njit functions in a call stack have the same `fastmath` flag.
    This function raises a ValueError if it finds any inconsistencies in the
    `fastmath` flags in at lease one call stack of njit functions.

    Parameters
    ----------
    pkg_dir : str
        The path to the directory containing some .py files

    pkg_name : str
        The name of the package

    Returns
    -------
    None
    """
    # List of call stacks with inconsistent fastmath flags
    inconsistent_call_stacks = []

    njit_call_stacks = get_njit_call_stacks(pkg_dir, pkg_name)
    for cs in njit_call_stacks:
        # Set the fastmath flag of the first function in the call stack
        # as the reference flag
        module_name, func_name = cs[0].split(".")
        module = importlib.import_module(f".{module_name}", package="stumpy")
        func = getattr(module, func_name)
        flag_ref = func.targetoptions["fastmath"]

        for item in cs[1:]:
            module_name, func_name = cs[0].split(".")
            module = importlib.import_module(f".{module_name}", package="stumpy")
            func = getattr(module, func_name)
            flag = func.targetoptions["fastmath"]
            if flag != flag_ref:
                inconsistent_call_stacks.append(cs)
                break

    if len(inconsistent_call_stacks) > 0:
        msg = (
            "Found at least one call stack that has inconsistent `fastmath` flags. "
            + f"Those call stacks are:\n {inconsistent_call_stacks}\n"
        )
        raise ValueError(msg)

    return


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--check", dest="pkg_dir")
    args = parser.parse_args()

    if args.pkg_dir:
        pkg_dir = pathlib.Path(args.pkg_dir)
        pkg_name = pkg_dir.name
        check_fastmath(str(pkg_dir), pkg_name)
        check_call_stack_fastmath(str(pkg_dir), pkg_name)
