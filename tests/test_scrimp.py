import numpy as np
import numpy.testing as npt
import pandas as pd
from stumpy import scrimp, core, stump
import pytest
import utils


test_data = [
    np.array([584, -11, 23, 79, 1001, 0, -19], dtype=np.float64),
    np.random.uniform(-1000, 1000, [64]).astype(np.float64),
]

substitution_locations = [(slice(0, 0), 0, -1, slice(1, 3), [0, 3])]
substitution_values = [np.nan, np.inf]


@pytest.mark.parametrize("T", test_data)
def test_scrimp_self_join(T):
    m = 3
    left = np.zeros(T.shape[0])
    right = scrimp(T, m, percentage=0.0)
    utils.replace_inf(right)
    npt.assert_almost_equal(left, right[:, 0])


@pytest.mark.parametrize("T", test_data)
def test_scrimp_self_join(T):
    m = 3
    zone = int(np.ceil(m / 4))
    left = np.array(
        [
            utils.naive_mass(Q, T, m, i, zone, True)
            for i, Q in enumerate(core.rolling_window(T, m))
        ],
        dtype=object,
    )
    right = scrimp(T, m, percentage=1.0)
    utils.replace_inf(left)
    utils.replace_inf(right)
    npt.assert_almost_equal(left[:, 0], right[:, 0])

    right = scrimp(pd.Series(T), m, percentage=1.0)
    utils.replace_inf(right)
    npt.assert_almost_equal(left[:, 0], right[:, 0])


@pytest.mark.parametrize("T", test_data)
def test_scrimp_self_join_larger_window(T):
    for m in [8, 16, 32]:
        if len(T) > m:
            zone = int(np.ceil(m / 4))
            left = np.array(
                [
                    utils.naive_mass(Q, T, m, i, zone, True)
                    for i, Q in enumerate(core.rolling_window(T, m))
                ],
                dtype=object,
            )
            right = scrimp(T, m, percentage=1.0)
            utils.replace_inf(left)
            utils.replace_inf(right)
            npt.assert_almost_equal(left[:, 0], right[:, 0])

            right = scrimp(pd.Series(T), m, percentage=1.0)
            utils.replace_inf(right)
            npt.assert_almost_equal(left[:, 0], right[:, 0])


def test_scrimp_constant_subsequence_self_join():
    T = np.concatenate((np.zeros(20, dtype=np.float64), np.ones(5, dtype=np.float64)))
    m = 3
    zone = int(np.ceil(m / 4))
    left = np.array(
        [
            utils.naive_mass(Q, T, m, i, zone, True)
            for i, Q in enumerate(core.rolling_window(T, m))
        ],
        dtype=object,
    )
    right = scrimp(T, m, percentage=1.0)
    utils.replace_inf(left)
    utils.replace_inf(right)
    npt.assert_almost_equal(left[:, 0], right[:, 0])

    right = scrimp(pd.Series(T), m, percentage=1.0)
    utils.replace_inf(right)
    npt.assert_almost_equal(left[:, 0], right[:, 0])


@pytest.mark.parametrize("T", test_data)
@pytest.mark.parametrize("substitute", substitution_values)
@pytest.mark.parametrize("substitution_locations", substitution_locations)
def test_scrimp_nan_inf_self_join(T, substitute, substitution_locations):
    m = 3

    T_sub = T.copy()

    for substitution_location in substitution_locations:
        T_sub[:] = T[:]
        T_sub[substitution_location] = substitute
        print(substitute, substitution_location, T_sub)

        zone = int(np.ceil(m / 4))
        left = np.array(
            [
                utils.naive_mass(Q, T_sub, m, i, zone, True)
                for i, Q in enumerate(core.rolling_window(T_sub, m))
            ],
            dtype=object,
        )
        right = scrimp(T_sub, m, percentage=1.0)
        utils.replace_inf(left)
        utils.replace_inf(right)
        npt.assert_almost_equal(left[:, 0], right[:, 0])

        right = scrimp(pd.Series(T_sub), m, percentage=1.0)
        utils.replace_inf(right)
        npt.assert_almost_equal(left[:, 0], right[:, 0])


def test_scrimp_nan_zero_mean_self_join():
    T = np.array([-1, 0, 1, np.inf, 1, 0, -1])
    m = 3

    zone = int(np.ceil(m / 4))
    left = np.array(
        [
            utils.naive_mass(Q, T, m, i, zone, True)
            for i, Q in enumerate(core.rolling_window(T, m))
        ],
        dtype=object,
    )
    right = scrimp(T, m, percentage=1.0)

    utils.replace_inf(left)
    utils.replace_inf(right)
    npt.assert_almost_equal(left[:, 0], right[:, 0])
