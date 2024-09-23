import ast


def check_fastmath(decorator):
    """
    For the given `decorator` node with type `ast.Call`, 
    return the value of the `fastmath` argument if it exists.
    Otherwise, return `None`.
    """
    fastmath_value = None
    for n in ast.iter_child_nodes(decorator):
        if isinstance(n, ast.keyword) and n.arg == 'fastmath':
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

    fastmath_value = [None] * len(njit_nodes)
    for i, node in enumerate(njit_nodes):
        if node is not None:
            fastmath_value[i] = check_fastmath(node)

    return func_names, is_njit, fastmath_values