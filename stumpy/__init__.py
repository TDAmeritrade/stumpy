from pkg_resources import get_distribution, DistributionNotFound
import os.path
from .core import mass  # noqa: F401
from .stump import stump  # noqa: F401
from .stumped import stumped  # noqa: F401
from .mstump import mstump, subspace, mdl  # noqa: F401
from .mstumped import mstumped  # noqa: F401
from .aamp import aamp  # noqa: F401
from .aamped import aamped  # noqa: F401
from .maamp import maamp, maamp_subspace, maamp_mdl  # noqa: F401
from .maamped import maamped  # noqa: F401
from .aampi import aampi  # noqa: F401
from .chains import atsc, allc  # noqa: F401
from .floss import floss, fluss  # noqa: F401
from .ostinato import ostinato, ostinatoed  # noqa: F401
from .aamp_ostinato import aamp_ostinato, aamp_ostinatoed  # noqa: F401
from .scrump import scrump, prescrump  # noqa: F401
from .scraamp import scraamp, prescraamp  # noqa: F401
from .stumpi import stumpi  # noqa: F401
from .mpdist import mpdist, mpdisted  # noqa: F401
from .aampdist import aampdist, aampdisted  # noqa: F401
from .motifs import motifs, match  # noqa: F401
from .aamp_motifs import aamp_motifs, aamp_match  # noqa: F401
from .mmotifs import mmotifs  # noqa: F401
from .aamp_mmotifs import aamp_mmotifs  # noqa: F401
from .snippets import snippets  # noqa: F401
from .aampdist_snippets import aampdist_snippets  # noqa: F401
from .stimp import stimp, stimped  # noqa: F401
from .aamp_stimp import aamp_stimp, aamp_stimped  # noqa: F401
from numba import cuda

if cuda.is_available():
    from .gpu_stump import gpu_stump  # noqa: F401
    from .gpu_aamp import gpu_aamp  # noqa: F401
    from .gpu_ostinato import gpu_ostinato  # noqa: F401
    from .gpu_aamp_ostinato import gpu_aamp_ostinato  # noqa: F401
    from .gpu_mpdist import gpu_mpdist  # noqa: F401
    from .gpu_aampdist import gpu_aampdist  # noqa: F401
    from .gpu_stimp import gpu_stimp  # noqa: F401
    from .gpu_aamp_stimp import gpu_aamp_stimp  # noqa: F401
else:  # pragma: no cover
    from .core import _gpu_stump_driver_not_found as gpu_stump  # noqa: F401
    from .core import _gpu_aamp_driver_not_found as gpu_aamp  # noqa: F401
    from .core import _gpu_ostinato_driver_not_found as gpu_ostinato  # noqa: F401
    from .core import (
        _gpu_aamp_ostinato_driver_not_found as gpu_aamp_ostinato,
    )  # noqa: F401
    from .core import _gpu_mpdist_driver_not_found as gpu_mpdist  # noqa: F401
    from .core import _gpu_aampdist_driver_not_found as gpu_aampdist  # noqa: F401
    from .core import _gpu_stimp_driver_not_found as gpu_stimp  # noqa: F401
    from .core import _gpu_aamp_stimp_driver_not_found as gpu_aamp_stimp  # noqa: F401
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

    # Fix GPU-STIMP Docs
    # Note that this is a special case for class definitions.
    # See above for function definitions.
    # Also, please update docs/api.rst
    gpu_stimp.__doc__ = ""
    filepath = pathlib.Path(__file__).parent / "gpu_stimp.py"

    file_contents = ""
    with open(filepath, encoding="utf8") as f:
        file_contents = f.read()
    module = ast.parse(file_contents)
    class_definitions = [node for node in module.body if isinstance(node, ast.ClassDef)]
    for cd in class_definitions:
        if cd.name == "gpu_stimp":
            gpu_stimp.__doc__ = ast.get_docstring(cd)

    # Fix GPU-AAMP-STIMP Docs
    # Note that this is a special case for class definitions.
    # See above for function definitions.
    # Also, please update docs/api.rst
    gpu_aamp_stimp.__doc__ = ""
    filepath = pathlib.Path(__file__).parent / "gpu_aamp_stimp.py"

    file_contents = ""
    with open(filepath, encoding="utf8") as f:
        file_contents = f.read()
    module = ast.parse(file_contents)
    class_definitions = [node for node in module.body if isinstance(node, ast.ClassDef)]
    for cd in class_definitions:
        if cd.name == "gpu_aamp_stimp":
            gpu_aamp_stimp.__doc__ = ast.get_docstring(cd)

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
