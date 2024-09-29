import pathlib

from utils import check_callees, check_functions

stumpy_path = pathlib.Path(__file__).parent / "stumpy"
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
