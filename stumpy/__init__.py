from .stomp import stomp  # noqa: F401
from .stump import (  # noqa: F401
    stump,
    _stump,
    _calculate_squared_distance_profile,
    _get_first_stump_profile,
    _get_QT,
)
from .stumped import stumped  # noqa: F401
from .mstump import (  # noqa: F401
    mstump,
    _mstump,
    _get_first_mstump_profile,
    _get_multi_QT,
    _multi_compute_mean_std,
    _multi_mass,
)
from .mstumped import mstumped  # noqa: F401
from .chains import atsc, allc  # noqa: F401

from pkg_resources import get_distribution, DistributionNotFound
import os.path

try:
    _dist = get_distribution("stumpy")
    # Normalize case for Windows systems
    dist_loc = os.path.normcase(_dist.location)
    here = os.path.normcase(__file__)
    if not here.startswith(os.path.join(dist_loc, "stumpy")):
        # not installed, but there is another version that *is*
        raise DistributionNotFound
except DistributionNotFound:
    __version__ = "Please install this project with setup.py"
else:
    __version__ = _dist.version
