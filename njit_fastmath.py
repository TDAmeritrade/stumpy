import pathlib

from utils import check_callees, check_functions

stumpy_path = pathlib.Path(__file__).parent / "stumpy"
filepaths = sorted(f for f in pathlib.Path(stumpy_path).iterdir() if f.is_file())

all_functions = {}
callees = {}

ignore = ["__init__.py", "__pycache__"]
for filepath in filepaths:
    name = filepath.name
    if name not in ignore and str(filepath).endswith(".py"):
        callees[name] = check_callees(filepath)

        func_names, is_njit, fastmath_values = check_functions(filepath)
        all_functions[name] = {
            "func_names": func_names,
            "is_njit": is_njit,
            "fastmath_values": fastmath_values,
        }


stumpy_functions = set()
for fname, func_data in all_functions.items():
    prefix = fname.replace(".py", "")
    stumpy_functions.update([prefix + "." + x for x in func_data["func_names"]])
stumpy_functions = list(stumpy_functions)
stumpy_functions_no_prefix = [x.split(".")[1] for x in stumpy_functions]


# create  a dictionary where keys are function names in stumpy_functions, and the value
# is a tuple, where the first element says whether the function is decorated with njit
# and the second element is the list of callees

callers_callees = {}
for func_name in stumpy_functions:
    callers_callees[func_name] = None

    prefix, func = func_name.split(".")
    fname = prefix + ".py"

    idx = all_functions[fname]["func_names"].index(func)
    is_njit = all_functions[fname]["is_njit"][idx]
    fastmath_val = all_functions[fname]["fastmath_values"][idx]

    callees_functions = callees[fname][func]

    pruned_callees_functions = []
    for callee in callees_functions:
        if callee in all_functions[fname]["func_names"]:
            pruned_callees_functions.append(prefix + "." + callee)
        elif callee in stumpy_functions:
            idx = stumpy_functions_no_prefix.index(callee)
            pruned_callees_functions.append(stumpy_functions[idx])
        else:
            continue

    callers_callees[func_name] = (is_njit, fastmath_val, pruned_callees_functions)


# Create callees_callers dictionary using callers_callees dictionary
callees_callers = {}
for func_name, func_metadata in  callers_callees.items():
    callees_callers[func_name] = [func_metadata[0], func_metadata[1], []]


for func_name, func_metadata in callers_callees.items():
    for callee in func_metadata[2]:
        callees_callers[callee][-1].append(func_name)


for func_name, func_metadata in callees_callers.items():
    callees_callers[func_name][2] = set(callees_callers[func_name][2])
    callees_callers[func_name] = tuple(callees_callers[func_name])


