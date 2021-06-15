import numpy as np
from stumpy import config


def test_change_excl_zone_denom():
    assert config.STUMPY_EXCL_ZONE_DENOM == 4

    config.STUMPY_EXCL_ZONE_DENOM = np.inf
    assert config.STUMPY_EXCL_ZONE_DENOM == np.inf

    config.STUMPY_EXCL_ZONE_DENOM = 4
    assert config.STUMPY_EXCL_ZONE_DENOM == 4
