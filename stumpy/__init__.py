from pkg_resources import get_distribution, DistributionNotFound
import os.path
from .stump import stump  # noqa: F401
from .stumped import stumped  # noqa: F401
from .mstump import mstump, subspace  # noqa: F401
from .mstumped import mstumped  # noqa: F401
from .aamp import aamp  # noqa: F401
from .aamped import aamped  # noqa: F401
from .aampi import aampi  # noqa: F401
from .chains import atsc, allc  # noqa: F401
from .floss import floss, fluss  # noqa: F401
from .ostinato import ostinato, ostinatoed  # noqa: F401
from .aamp_ostinato import aamp_ostinato, aamp_ostinatoed  # noqa: F401
from .scrump import scrump  # noqa: F401
from .stumpi import stumpi  # noqa: F401
from .mpdist import mpdist, mpdisted  # noqa: F401
from .aampdist import aampdist, aampdisted  # noqa: F401
from numba import cuda

if cuda.is_available():
    from .gpu_stump import gpu_stump  # noqa: F401
    from .gpu_aamp import gpu_aamp  # noqa: F401
    from .gpu_ostinato import gpu_ostinato  # noqa: F401
    from .gpu_aamp_ostinato import gpu_aamp_ostinato  # noqa: F401
    from .gpu_mpdist import gpu_mpdist  # noqa: F401
    from .gpu_aampdist import gpu_aampdist  # noqa: F401
else:  # pragma: no cover
    from .core import _gpu_stump_driver_not_found as gpu_stump  # noqa: F401
    from .core import _gpu_aamp_driver_not_found as gpu_aamp  # noqa: F401
    from .core import _gpu_ostinato_driver_not_found as gpu_ostinato  # noqa: F401
    from .core import (
        _gpu_aamp_ostinato_driver_not_found as gpu_aamp_ostinato,
    )  # noqa: F401
    from .core import _gpu_mpdist_driver_not_found as gpu_mpdist  # noqa: F401
    from .core import _gpu_aampdist_driver_not_found as gpu_aampdist  # noqa: F401
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

    # Fix GPU-OSTINATO Docs
    gpu_ostinato.__doc__ = ""
    filepath = pathlib.Path(__file__).parent / "gpu_ostinato.py"

    file_contents = ""
    with open(filepath, encoding="utf8") as f:
        file_contents = f.read()
    module = ast.parse(file_contents)
    function_definitions = [
        node for node in module.body if isinstance(node, ast.FunctionDef)
    ]
    for fd in function_definitions:
        if fd.name == "gpu_ostinato":
            gpu_ostinato.__doc__ = ast.get_docstring(fd)

    # Fix GPU-AAMP-OSTINATO Docs
    gpu_aamp_ostinato.__doc__ = ""
    filepath = pathlib.Path(__file__).parent / "gpu_aamp_ostinato.py"

    file_contents = ""
    with open(filepath, encoding="utf8") as f:
        file_contents = f.read()
    module = ast.parse(file_contents)
    function_definitions = [
        node for node in module.body if isinstance(node, ast.FunctionDef)
    ]
    for fd in function_definitions:
        if fd.name == "gpu_aamp_ostinato":
            gpu_aamp_ostinato.__doc__ = ast.get_docstring(fd)

    # Fix GPU-MPDIST Docs
    gpu_mpdist.__doc__ = ""
    filepath = pathlib.Path(__file__).parent / "gpu_mpdist.py"

    file_contents = ""
    with open(filepath, encoding="utf8") as f:
        file_contents = f.read()
    module = ast.parse(file_contents)
    function_definitions = [
        node for node in module.body if isinstance(node, ast.FunctionDef)
    ]
    for fd in function_definitions:
        if fd.name == "gpu_mpdist":
            gpu_mpdist.__doc__ = ast.get_docstring(fd)

    # Fix GPU-AAMPDIST Docs
    gpu_aampdist.__doc__ = ""
    filepath = pathlib.Path(__file__).parent / "gpu_aampdist.py"

    file_contents = ""
    with open(filepath, encoding="utf8") as f:
        file_contents = f.read()
    module = ast.parse(file_contents)
    function_definitions = [
        node for node in module.body if isinstance(node, ast.FunctionDef)
    ]
    for fd in function_definitions:
        if fd.name == "gpu_aampdist":
            gpu_aampdist.__doc__ = ast.get_docstring(fd)

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
