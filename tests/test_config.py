from stumpy import core, config


def test_change_excl_zone_denom():
    assert core.get_max_window_size(10) == 7

    config.STUMPY_EXCL_ZONE_DENOM = 2
    assert core.get_max_window_size(10) == 6

    config.STUMPY_EXCL_ZONE_DENOM = 4
    assert core.get_max_window_size(10) == 7
