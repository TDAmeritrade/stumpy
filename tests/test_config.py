from stumpy import config, core


def test_change_excl_zone_denom():
    assert core.get_max_window_size(10) == 7

    config.STUMPY_EXCL_ZONE_DENOM = 2
    assert core.get_max_window_size(10) == 6

    config.STUMPY_EXCL_ZONE_DENOM = 4
    assert core.get_max_window_size(10) == 7


def test_reset_one_var():
    ref = config.STUMPY_EXCL_ZONE_DENOM

    config.STUMPY_EXCL_ZONE_DENOM += 1
    config._reset("STUMPY_EXCL_ZONE_DENOM")

    assert config.STUMPY_EXCL_ZONE_DENOM == ref


def test_reset_all_vars():
    ref_fastmath = config.STUMPY_FASTMATH_TRUE
    ref_excl_zone_denom = config.STUMPY_EXCL_ZONE_DENOM

    config.STUMPY_FASTMATH_TRUE = not config.STUMPY_FASTMATH_TRUE
    config.STUMPY_EXCL_ZONE_DENOM += 1

    config._reset()
    assert config.STUMPY_FASTMATH_TRUE == ref_fastmath
    assert config.STUMPY_EXCL_ZONE_DENOM == ref_excl_zone_denom
