from pkg_resources import get_distribution, DistributionNotFound
import os.path
from .stump import (  # noqa: F401
    stump,
    _stump,
)
from .stumped import stumped  # noqa: F401
from .mstump import (  # noqa: F401
    mstump,
    _mstump,
    _query_mstump_profile,
    _get_first_mstump_profile,
    _get_multi_QT,
    _multi_mass,
    _apply_include,
)
from .mstumped import mstumped  # noqa: F401
from .aamp import aamp, _aamp  # noqa: F401
from .aamped import aamped  # noqa: F401
from .aampi import aampi  # noqa: F401
from .chains import atsc, allc  # noqa: F401
from .floss import floss, fluss, _nnmark, _iac, _cac, _rea  # noqa: F401
from .scrump import scrump  # noqa: F401
from .stumpi import stumpi  # noqa: F401
from numba import cuda

if cuda.is_available():
    from .gpu_stump import gpu_stump  # noqa: F401
    from .gpu_aamp import gpu_aamp  # noqa: F401
else:  # pragma: no cover
    from .core import _gpu_stump_driver_not_found as gpu_stump  # noqa: F401
    from .core import _gpu_aamp_driver_not_found as gpu_aamp  # noqa: F401
    import ast
    import pathlib

    # Fix GPU-STUMP Docs
    gpu_stump.__doc__ = ""
    filepath = pathlib.Path(__file__).parent / "gpu_stump.py"

    file_contents = ""
    with open(filepath, encoding="utf8") as f:
        file_contents = f.read()
    module = ast.parse(file_contents)
    function_definitions = [
        node for node in module.body if isinstance(node, ast.FunctionDef)
    ]
    for fd in function_definitions:
        if fd.name == "gpu_stump":
            gpu_stump.__doc__ = ast.get_docstring(fd)

    # Fix GPU-AAMP Docs
    gpu_aamp.__doc__ = ""
    filepath = pathlib.Path(__file__).parent / "gpu_aamp.py"

    file_contents = ""
    with open(filepath, encoding="utf8") as f:
        file_contents = f.read()
    module = ast.parse(file_contents)
    function_definitions = [
        node for node in module.body if isinstance(node, ast.FunctionDef)
    ]
    for fd in function_definitions:
        if fd.name == "gpu_aamp":
            gpu_aamp.__doc__ = ast.get_docstring(fd)

try:
    _dist = get_distribution("stumpy")
    # Normalize case for Windows systems
    dist_loc = os.path.normcase(_dist.location)
    here = os.path.normcase(__file__)
    if not here.startswith(os.path.join(dist_loc, "stumpy")):
        # not installed, but there is another version that *is*
        raise DistributionNotFound  # pragma: no cover
except DistributionNotFound:  # pragma: no cover
    __version__ = "Please install this project with setup.py"
else:  # pragma: no cover
    __version__ = _dist.version
