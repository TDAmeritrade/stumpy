import ast

import pathlib

from stumpy import cache

def check_fastmath(decorator):
    """
    For the given `decorator` node with type `ast.Call`,
    return the value of the `fastmath` argument if it exists.
    Otherwise, return `None`.
    """
    fastmath_value = None
    for n in ast.iter_child_nodes(decorator):
        if isinstance(n, ast.keyword) and n.arg == "fastmath":
            if isinstance(n.value, ast.Constant):
                fastmath_value = n.value.value
            elif isinstance(n.value, ast.Set):
                fastmath_value = set(item.value for item in n.value.elts)
            else:
                pass
            break

    return fastmath_value


def check_njit(fd):
    """
    For the given `fd` node with type `ast.FunctionDef`,
    return the node of the `njit` decorator if it exists.
    Otherwise, return `None`.
    """
    decorator_node = None
    for decorator in fd.decorator_list:
        if not isinstance(decorator, ast.Call):
            continue

        obj = decorator.func
        if isinstance(obj, ast.Attribute):
            name = obj.attr
        elif isinstance(obj, ast.Subscript):
            name = obj.value.id
        elif isinstance(obj, ast.Name):
            name = obj.id
        else:
            msg = f"The type {type(obj)} is not supported."
            raise ValueError(msg)

        if name == "njit":
            decorator_node = decorator
            break

    return decorator_node


def check_functions(filepath):
    """
    For the given `filepath`, return the function names,
    whether the function is decorated with `@njit`,
    and the value of the `fastmath` argument if it exists

    Parameters
    ----------
    filepath : str
        The path to the file

    Returns
    -------
    func_names : list
        List of function names

    is_njit : list
        List of boolean values indicating whether the function is decorated with `@njit`

    fastmath_value : list
        List of values of the `fastmath` argument if it exists
    """
    file_contents = ""
    with open(filepath, encoding="utf8") as f:
        file_contents = f.read()
    module = ast.parse(file_contents)

    function_definitions = [
        node for node in module.body if isinstance(node, ast.FunctionDef)
    ]

    func_names = [fd.name for fd in function_definitions]

    njit_nodes = [check_njit(fd) for fd in function_definitions]
    is_njit = [node is not None for node in njit_nodes]

    fastmath_values = [None] * len(njit_nodes)
    for i, node in enumerate(njit_nodes):
        if node is not None:
            fastmath_values[i] = check_fastmath(node)

    return func_names, is_njit, fastmath_values


def _get_callees(node, all_functions):
    for n in ast.iter_child_nodes(node):
        if isinstance(n, ast.Call):
            obj = n.func
            if isinstance(obj, ast.Attribute):
                name = obj.attr
            elif isinstance(obj, ast.Subscript):
                name = obj.value.id
            elif isinstance(obj, ast.Name):
                name = obj.id
            else:
                msg = f"The type {type(obj)} is not supported"
                raise ValueError(msg)

            all_functions.append(name)

        _get_callees(n, all_functions)


def get_all_callees(fd):
    """
    For a given node of type ast.FunctionDef, visit all of its child nodes,
    and return a list of all of its callees
    """
    all_functions = []
    _get_callees(fd, all_functions)

    return all_functions


def check_callees(filepath):
    """
    For the given `filepath`, return a dictionary with the key
    being the function name and the value being a set of function names
    that are called by the function
    """
    file_contents = ""
    with open(filepath, encoding="utf8") as f:
        file_contents = f.read()
    module = ast.parse(file_contents)

    function_definitions = [
        node for node in module.body if isinstance(node, ast.FunctionDef)
    ]

    callees = {}
    for fd in function_definitions:
        callees[fd.name] = set(get_all_callees(fd))

    return callees


stumpy_path = pathlib.Path(__file__).parent # / "stumpy"
filepaths = sorted(f for f in pathlib.Path(stumpy_path).iterdir() if f.is_file())

all_functions = {}

ignore = ["__init__.py", "__pycache__"]
for filepath in filepaths:
    file_name = filepath.name
    if file_name not in ignore and str(filepath).endswith(".py"):
        prefix = file_name.replace(".py", "")

        func_names, is_njit, fastmath_values = check_functions(filepath)
        func_names = [f"{prefix}.{fn}" for fn in func_names]

        all_functions[file_name] = {
            "func_names": func_names,
            "is_njit": is_njit,
            "fastmath_values": fastmath_values,
        }

all_stumpy_functions = set()
for file_name, file_functions_metadata in all_functions.items():
    all_stumpy_functions.update(file_functions_metadata["func_names"])

all_stumpy_functions = list(all_stumpy_functions)
all_stumpy_functions_no_prefix = [f.split(".")[-1] for f in all_stumpy_functions]


# output 1: func_metadata
func_metadata = {}
for file_name, file_functions_metadata in all_functions.items():
    for i, f in enumerate(file_functions_metadata["func_names"]):
        is_njit = file_functions_metadata["is_njit"][i]
        fastmath_value = file_functions_metadata["fastmath_values"][i]
        func_metadata[f] = [is_njit, fastmath_value]


# output 2: func_callers
func_callers = {}
for f in func_metadata.keys():
    func_callers[f] = []

for filepath in filepaths:
    file_name = filepath.name
    if file_name in ignore or not str(filepath).endswith(".py"):
        continue

    prefix = file_name.replace(".py", "")
    callees = check_callees(filepath)

    current_callers = set(callees.keys())
    for caller, callee_set in callees.items():
        s = list(callee_set.intersection(all_stumpy_functions_no_prefix))
        if len(s) == 0:
            continue

        for c in s:
            if c in current_callers:
                c_name = prefix + "." + c
            else:
                idx = all_stumpy_functions_no_prefix.index(c)
                c_name = all_stumpy_functions[idx]

            func_callers[c_name].append(f"{prefix}.{caller}")


for f, callers in func_callers.items():
    func_callers[f] = list(set(callers))



for modue_name, func_name in cache.get_njit_funcs():
    f = f"{modue_name}.{func_name}"
    print(f, func_callers[f])