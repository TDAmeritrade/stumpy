import numpy as np
import numpy.testing as npt
import pandas as pd
from stumpy import config, aamped
from dask.distributed import Client, LocalCluster
import pytest
import naive


@pytest.fixture(scope="module")
def dask_cluster():
    cluster = LocalCluster(n_workers=2, threads_per_worker=2)
    yield cluster
    cluster.close()


test_data = [
    (
        np.array([9, 8100, -60, 7], dtype=np.float64),
        np.array([584, -11, 23, 79, 1001, 0, -19], dtype=np.float64),
    ),
    (
        np.random.uniform(-1000, 1000, [8]).astype(np.float64),
        np.random.uniform(-1000, 1000, [64]).astype(np.float64),
    ),
]

one_substitution_locations = [0, -1, slice(1, 3), [0, 3]]
two_substitution_locations = [(0, 1), (-1, slice(1, 3)), (slice(1, 3), 1), ([0, 3], 1)]
window_size = [8, 16, 32]


def test_aamp_int_input(dask_cluster):
    with pytest.raises(TypeError):
        with Client(dask_cluster) as dask_client:
            aamped(dask_client, np.arange(10), 5)


@pytest.mark.filterwarnings("ignore:numpy.dtype size changed")
@pytest.mark.filterwarnings("ignore:numpy.ufunc size changed")
@pytest.mark.filterwarnings("ignore:numpy.ndarray size changed")
@pytest.mark.filterwarnings("ignore:\\s+Port 8787 is already in use:UserWarning")
@pytest.mark.parametrize("T_A, T_B", test_data)
def test_aamped_self_join(T_A, T_B, dask_cluster):
    with Client(dask_cluster) as dask_client:
        m = 3
        for p in [1.0, 2.0, 3.0]:
            ref_mp = naive.aamp(T_B, m, p=p)
            comp_mp = aamped(dask_client, T_B, m, p=p)
            naive.replace_inf(ref_mp)
            naive.replace_inf(comp_mp)
            npt.assert_almost_equal(ref_mp, comp_mp)


@pytest.mark.filterwarnings("ignore:numpy.dtype size changed")
@pytest.mark.filterwarnings("ignore:numpy.ufunc size changed")
@pytest.mark.filterwarnings("ignore:numpy.ndarray size changed")
@pytest.mark.filterwarnings("ignore:\\s+Port 8787 is already in use:UserWarning")
@pytest.mark.parametrize("T_A, T_B", test_data)
def test_aamped_self_join_df(T_A, T_B, dask_cluster):
    with Client(dask_cluster) as dask_client:
        m = 3
        ref_mp = naive.aamp(T_B, m)
        comp_mp = aamped(dask_client, pd.Series(T_B), m)
        naive.replace_inf(ref_mp)
        naive.replace_inf(comp_mp)
        npt.assert_almost_equal(ref_mp, comp_mp)


@pytest.mark.filterwarnings("ignore:numpy.dtype size changed")
@pytest.mark.filterwarnings("ignore:numpy.ufunc size changed")
@pytest.mark.filterwarnings("ignore:numpy.ndarray size changed")
@pytest.mark.filterwarnings("ignore:\\s+Port 8787 is already in use:UserWarning")
@pytest.mark.parametrize("T_A, T_B", test_data)
@pytest.mark.parametrize("m", window_size)
def test_aamped_self_join_larger_window(T_A, T_B, m, dask_cluster):
    with Client(dask_cluster) as dask_client:
        if len(T_B) > m:
            ref_mp = naive.aamp(T_B, m)
            comp_mp = aamped(dask_client, T_B, m)
            naive.replace_inf(ref_mp)
            naive.replace_inf(comp_mp)

            npt.assert_almost_equal(ref_mp, comp_mp)


@pytest.mark.filterwarnings("ignore:numpy.dtype size changed")
@pytest.mark.filterwarnings("ignore:numpy.ufunc size changed")
@pytest.mark.filterwarnings("ignore:numpy.ndarray size changed")
@pytest.mark.filterwarnings("ignore:\\s+Port 8787 is already in use:UserWarning")
@pytest.mark.parametrize("T_A, T_B", test_data)
@pytest.mark.parametrize("m", window_size)
def test_aamped_self_join_larger_window_df(T_A, T_B, m, dask_cluster):
    with Client(dask_cluster) as dask_client:
        if len(T_B) > m:
            ref_mp = naive.aamp(T_B, m)
            comp_mp = aamped(dask_client, pd.Series(T_B), m)
            naive.replace_inf(ref_mp)
            naive.replace_inf(comp_mp)

            npt.assert_almost_equal(ref_mp, comp_mp)


@pytest.mark.filterwarnings("ignore:numpy.dtype size changed")
@pytest.mark.filterwarnings("ignore:numpy.ufunc size changed")
@pytest.mark.filterwarnings("ignore:numpy.ndarray size changed")
@pytest.mark.filterwarnings("ignore:\\s+Port 8787 is already in use:UserWarning")
@pytest.mark.parametrize("T_A, T_B", test_data)
def test_aamped_A_B_join(T_A, T_B, dask_cluster):
    with Client(dask_cluster) as dask_client:
        m = 3
        for p in [1.0, 2.0, 3.0]:
            ref_mp = naive.aamp(T_A, m, T_B=T_B, p=p)
            comp_mp = aamped(dask_client, T_A, m, T_B, ignore_trivial=False, p=p)
            naive.replace_inf(ref_mp)
            naive.replace_inf(comp_mp)
            npt.assert_almost_equal(ref_mp, comp_mp)


@pytest.mark.filterwarnings("ignore:numpy.dtype size changed")
@pytest.mark.filterwarnings("ignore:numpy.ufunc size changed")
@pytest.mark.filterwarnings("ignore:numpy.ndarray size changed")
@pytest.mark.filterwarnings("ignore:\\s+Port 8787 is already in use:UserWarning")
@pytest.mark.parametrize("T_A, T_B", test_data)
def test_aamped_A_B_join_df(T_A, T_B, dask_cluster):
    with Client(dask_cluster) as dask_client:
        m = 3
        ref_mp = naive.aamp(T_A, m, T_B=T_B)
        comp_mp = aamped(
            dask_client, pd.Series(T_A), m, pd.Series(T_B), ignore_trivial=False
        )
        naive.replace_inf(ref_mp)
        naive.replace_inf(comp_mp)
        npt.assert_almost_equal(ref_mp, comp_mp)


@pytest.mark.filterwarnings("ignore:A large number of values are smaller")
@pytest.mark.filterwarnings("ignore:For a self-join")
@pytest.mark.filterwarnings("ignore:numpy.dtype size changed")
@pytest.mark.filterwarnings("ignore:numpy.ufunc size changed")
@pytest.mark.filterwarnings("ignore:numpy.ndarray size changed")
@pytest.mark.filterwarnings("ignore:\\s+Port 8787 is already in use:UserWarning")
def test_aamped_one_constant_subsequence_self_join(dask_cluster):
    with Client(dask_cluster) as dask_client:
        T_A = np.concatenate(
            (np.zeros(20, dtype=np.float64), np.ones(5, dtype=np.float64))
        )
        m = 3
        ref_mp = naive.aamp(T_A, m)
        comp_mp = aamped(dask_client, T_A, m, ignore_trivial=True)
        naive.replace_inf(ref_mp)
        naive.replace_inf(comp_mp)
        npt.assert_almost_equal(ref_mp[:, 0], comp_mp[:, 0])  # ignore indices


@pytest.mark.filterwarnings("ignore:A large number of values are smaller")
@pytest.mark.filterwarnings("ignore:For a self-join")
@pytest.mark.filterwarnings("ignore:numpy.dtype size changed")
@pytest.mark.filterwarnings("ignore:numpy.ufunc size changed")
@pytest.mark.filterwarnings("ignore:numpy.ndarray size changed")
@pytest.mark.filterwarnings("ignore:\\s+Port 8787 is already in use:UserWarning")
def test_aamped_one_constant_subsequence_self_join_df(dask_cluster):
    with Client(dask_cluster) as dask_client:
        T_A = np.concatenate(
            (np.zeros(20, dtype=np.float64), np.ones(5, dtype=np.float64))
        )
        m = 3
        ref_mp = naive.aamp(T_A, m)
        comp_mp = aamped(dask_client, pd.Series(T_A), m, ignore_trivial=True)
        naive.replace_inf(ref_mp)
        naive.replace_inf(comp_mp)
        npt.assert_almost_equal(ref_mp[:, 0], comp_mp[:, 0])  # ignore indices


@pytest.mark.filterwarnings("ignore:\\s+A large number of values are smaller")
@pytest.mark.filterwarnings("ignore:\\s+For a self-join")
@pytest.mark.filterwarnings("ignore:numpy.dtype size changed")
@pytest.mark.filterwarnings("ignore:numpy.ufunc size changed")
@pytest.mark.filterwarnings("ignore:numpy.ndarray size changed")
@pytest.mark.filterwarnings("ignore:\\s+Port 8787 is already in use:UserWarning")
def test_aamped_one_constant_subsequence_A_B_join(dask_cluster):
    with Client(dask_cluster) as dask_client:
        T_A = np.random.rand(20)
        T_B = np.concatenate(
            (np.zeros(20, dtype=np.float64), np.ones(5, dtype=np.float64))
        )
        m = 3
        ref_mp = naive.aamp(T_A, m, T_B=T_B)
        comp_mp = aamped(dask_client, T_A, m, T_B, ignore_trivial=False)
        naive.replace_inf(ref_mp)
        naive.replace_inf(comp_mp)
        npt.assert_almost_equal(ref_mp[:, 0], comp_mp[:, 0])  # ignore indices


@pytest.mark.filterwarnings("ignore:\\s+A large number of values are smaller")
@pytest.mark.filterwarnings("ignore:\\s+For a self-join")
@pytest.mark.filterwarnings("ignore:numpy.dtype size changed")
@pytest.mark.filterwarnings("ignore:numpy.ufunc size changed")
@pytest.mark.filterwarnings("ignore:numpy.ndarray size changed")
@pytest.mark.filterwarnings("ignore:\\s+Port 8787 is already in use:UserWarning")
def test_aamped_one_constant_subsequence_A_B_join_df(dask_cluster):
    with Client(dask_cluster) as dask_client:
        T_A = np.random.rand(20)
        T_B = np.concatenate(
            (np.zeros(20, dtype=np.float64), np.ones(5, dtype=np.float64))
        )
        m = 3
        ref_mp = naive.aamp(T_A, m, T_B=T_B)
        comp_mp = aamped(
            dask_client, pd.Series(T_A), m, pd.Series(T_B), ignore_trivial=False
        )
        naive.replace_inf(ref_mp)
        naive.replace_inf(comp_mp)
        npt.assert_almost_equal(ref_mp[:, 0], comp_mp[:, 0])  # ignore indices


@pytest.mark.filterwarnings("ignore:\\s+A large number of values are smaller")
@pytest.mark.filterwarnings("ignore:\\s+For a self-join")
@pytest.mark.filterwarnings("ignore:numpy.dtype size changed")
@pytest.mark.filterwarnings("ignore:numpy.ufunc size changed")
@pytest.mark.filterwarnings("ignore:numpy.ndarray size changed")
@pytest.mark.filterwarnings("ignore:\\s+Port 8787 is already in use:UserWarning")
def test_aamped_one_constant_subsequence_A_B_join_swap(dask_cluster):
    with Client(dask_cluster) as dask_client:
        T_A = np.random.rand(20)
        T_B = np.concatenate(
            (np.zeros(20, dtype=np.float64), np.ones(5, dtype=np.float64))
        )
        m = 3
        ref_mp = naive.aamp(T_A, m, T_B=T_B)
        comp_mp = aamped(dask_client, T_A, m, T_B, ignore_trivial=False)
        naive.replace_inf(ref_mp)
        naive.replace_inf(comp_mp)
        npt.assert_almost_equal(ref_mp[:, 0], comp_mp[:, 0])  # ignore indices


@pytest.mark.filterwarnings("ignore:\\s+A large number of values are smaller")
@pytest.mark.filterwarnings("ignore:\\s+For a self-join")
@pytest.mark.filterwarnings("ignore:numpy.dtype size changed")
@pytest.mark.filterwarnings("ignore:numpy.ufunc size changed")
@pytest.mark.filterwarnings("ignore:numpy.ndarray size changed")
@pytest.mark.filterwarnings("ignore:\\s+Port 8787 is already in use:UserWarning")
def test_aamped_one_constant_subsequence_A_B_join_df_swap(dask_cluster):
    with Client(dask_cluster) as dask_client:
        T_A = np.random.rand(20)
        T_B = np.concatenate(
            (np.zeros(20, dtype=np.float64), np.ones(5, dtype=np.float64))
        )
        m = 3
        ref_mp = naive.aamp(T_A, m, T_B=T_B)
        comp_mp = aamped(
            dask_client, pd.Series(T_A), m, pd.Series(T_B), ignore_trivial=False
        )
        naive.replace_inf(ref_mp)
        naive.replace_inf(comp_mp)
        npt.assert_almost_equal(ref_mp[:, 0], comp_mp[:, 0])  # ignore indices


@pytest.mark.filterwarnings("ignore:A large number of values are smaller")
@pytest.mark.filterwarnings("ignore:For a self-join")
@pytest.mark.filterwarnings("ignore:numpy.dtype size changed")
@pytest.mark.filterwarnings("ignore:numpy.ufunc size changed")
@pytest.mark.filterwarnings("ignore:numpy.ndarray size changed")
@pytest.mark.filterwarnings("ignore:\\s+Port 8787 is already in use:UserWarning")
def test_aamped_identical_subsequence_self_join(dask_cluster):
    with Client(dask_cluster) as dask_client:
        identical = np.random.rand(8)
        T_A = np.random.rand(20)
        T_A[1 : 1 + identical.shape[0]] = identical
        T_A[11 : 11 + identical.shape[0]] = identical
        m = 3
        ref_mp = naive.aamp(T_A, m)
        comp_mp = aamped(dask_client, T_A, m, ignore_trivial=True)
        naive.replace_inf(ref_mp)
        naive.replace_inf(comp_mp)
        npt.assert_almost_equal(
            ref_mp[:, 0], comp_mp[:, 0], decimal=config.STUMPY_TEST_PRECISION
        )  # ignore indices


@pytest.mark.filterwarnings("ignore:\\s+A large number of values are smaller")
@pytest.mark.filterwarnings("ignore:\\s+For a self-join")
@pytest.mark.filterwarnings("ignore:numpy.dtype size changed")
@pytest.mark.filterwarnings("ignore:numpy.ufunc size changed")
@pytest.mark.filterwarnings("ignore:numpy.ndarray size changed")
@pytest.mark.filterwarnings("ignore:\\s+Port 8787 is already in use:UserWarning")
def test_aamped_identical_subsequence_A_B_join(dask_cluster):
    with Client(dask_cluster) as dask_client:
        identical = np.random.rand(8)
        T_A = np.random.rand(20)
        T_B = np.random.rand(20)
        T_A[1 : 1 + identical.shape[0]] = identical
        T_B[11 : 11 + identical.shape[0]] = identical
        m = 3
        ref_mp = naive.aamp(T_A, m, T_B=T_B)
        comp_mp = aamped(dask_client, T_A, m, T_B, ignore_trivial=False)
        naive.replace_inf(ref_mp)
        naive.replace_inf(comp_mp)
        npt.assert_almost_equal(
            ref_mp[:, 0], comp_mp[:, 0], decimal=config.STUMPY_TEST_PRECISION
        )  # ignore indices


@pytest.mark.filterwarnings("ignore:numpy.dtype size changed")
@pytest.mark.filterwarnings("ignore:numpy.ufunc size changed")
@pytest.mark.filterwarnings("ignore:numpy.ndarray size changed")
@pytest.mark.filterwarnings("ignore:\\s+Port 8787 is already in use:UserWarning")
@pytest.mark.parametrize("T_A, T_B", test_data)
@pytest.mark.parametrize("substitution_location_B", one_substitution_locations)
def test_aamped_one_subsequence_inf_A_B_join(
    T_A, T_B, substitution_location_B, dask_cluster
):
    with Client(dask_cluster) as dask_client:
        m = 3

        T_B_sub = T_B.copy()
        T_B_sub[substitution_location_B] = np.inf

        ref_mp = naive.aamp(T_A, m, T_B=T_B_sub)
        comp_mp = aamped(dask_client, T_A, m, T_B_sub, ignore_trivial=False)
        naive.replace_inf(ref_mp)
        naive.replace_inf(comp_mp)
        npt.assert_almost_equal(ref_mp, comp_mp)


@pytest.mark.filterwarnings("ignore:numpy.dtype size changed")
@pytest.mark.filterwarnings("ignore:numpy.ufunc size changed")
@pytest.mark.filterwarnings("ignore:numpy.ndarray size changed")
@pytest.mark.filterwarnings("ignore:\\s+Port 8787 is already in use:UserWarning")
@pytest.mark.parametrize("T_A, T_B", test_data)
@pytest.mark.parametrize("substitution_location_B", one_substitution_locations)
def test_aamped_one_subsequence_inf_self_join(
    T_A, T_B, substitution_location_B, dask_cluster
):
    with Client(dask_cluster) as dask_client:
        m = 3

        T_B_sub = T_B.copy()
        T_B_sub[substitution_location_B] = np.inf

        ref_mp = naive.aamp(T_B_sub, m)
        comp_mp = aamped(dask_client, T_B_sub, m, ignore_trivial=True)
        naive.replace_inf(ref_mp)
        naive.replace_inf(comp_mp)
        npt.assert_almost_equal(ref_mp, comp_mp)


@pytest.mark.filterwarnings("ignore:numpy.dtype size changed")
@pytest.mark.filterwarnings("ignore:numpy.ufunc size changed")
@pytest.mark.filterwarnings("ignore:numpy.ndarray size changed")
@pytest.mark.filterwarnings("ignore:\\s+Port 8787 is already in use:UserWarning")
def test_aamped_nan_zero_mean_self_join(dask_cluster):
    with Client(dask_cluster) as dask_client:
        T = np.array([-1, 0, 1, np.inf, 1, 0, -1])
        m = 3

        ref_mp = naive.aamp(T, m)
        comp_mp = aamped(dask_client, T, m, ignore_trivial=True)

        naive.replace_inf(ref_mp)
        naive.replace_inf(comp_mp)
        npt.assert_almost_equal(ref_mp, comp_mp)


@pytest.mark.filterwarnings("ignore:numpy.dtype size changed")
@pytest.mark.filterwarnings("ignore:numpy.ufunc size changed")
@pytest.mark.filterwarnings("ignore:numpy.ndarray size changed")
@pytest.mark.filterwarnings("ignore:\\s+Port 8787 is already in use:UserWarning")
@pytest.mark.parametrize("T_A, T_B", test_data)
@pytest.mark.parametrize("substitution_location_B", one_substitution_locations)
def test_aamped_one_subsequence_nan_A_B_join(
    T_A, T_B, substitution_location_B, dask_cluster
):
    with Client(dask_cluster) as dask_client:
        m = 3

        T_B_sub = T_B.copy()
        T_B_sub[substitution_location_B] = np.nan

        ref_mp = naive.aamp(T_A, m, T_B=T_B_sub)
        comp_mp = aamped(dask_client, T_A, m, T_B_sub, ignore_trivial=False)
        naive.replace_inf(ref_mp)
        naive.replace_inf(comp_mp)
        npt.assert_almost_equal(ref_mp, comp_mp)


@pytest.mark.filterwarnings("ignore:numpy.dtype size changed")
@pytest.mark.filterwarnings("ignore:numpy.ufunc size changed")
@pytest.mark.filterwarnings("ignore:numpy.ndarray size changed")
@pytest.mark.filterwarnings("ignore:\\s+Port 8787 is already in use:UserWarning")
@pytest.mark.parametrize("T_A, T_B", test_data)
@pytest.mark.parametrize("substitution_location_B", one_substitution_locations)
def test_aamped_one_subsequence_nan_self_join(
    T_A, T_B, substitution_location_B, dask_cluster
):
    with Client(dask_cluster) as dask_client:
        m = 3

        T_B_sub = T_B.copy()
        T_B_sub[substitution_location_B] = np.nan

        ref_mp = naive.aamp(T_B_sub, m)
        comp_mp = aamped(dask_client, T_B_sub, m, ignore_trivial=True)
        naive.replace_inf(ref_mp)
        naive.replace_inf(comp_mp)
        npt.assert_almost_equal(ref_mp, comp_mp)


@pytest.mark.filterwarnings("ignore:\\s+A large number of values are smaller")
@pytest.mark.filterwarnings("ignore:\\s+For a self-join")
@pytest.mark.filterwarnings("ignore:numpy.dtype size changed")
@pytest.mark.filterwarnings("ignore:numpy.ufunc size changed")
@pytest.mark.filterwarnings("ignore:numpy.ndarray size changed")
@pytest.mark.filterwarnings("ignore:\\s+Port 8787 is already in use:UserWarning")
# def test_two_constant_subsequences_A_B_join_swap(dask_cluster):
def test_two_constant_subsequences_A_B_join_swap(dask_cluster):
    with Client(dask_cluster) as dask_client:
        T_A = np.concatenate(
            (np.zeros(10, dtype=np.float64), np.ones(10, dtype=np.float64))
        )
        T_B = np.concatenate(
            (np.zeros(20, dtype=np.float64), np.ones(5, dtype=np.float64))
        )
        m = 3
        ref_mp = naive.aamp(T_B, m, T_B=T_A)
        comp_mp = aamped(dask_client, T_B, m, T_A, ignore_trivial=False)
        naive.replace_inf(ref_mp)
        naive.replace_inf(comp_mp)
        npt.assert_almost_equal(ref_mp[:, 0], comp_mp[:, 0])  # ignore indices


@pytest.mark.filterwarnings("ignore:\\s+A large number of values are smaller")
@pytest.mark.filterwarnings("ignore:\\s+For a self-join")
@pytest.mark.filterwarnings("ignore:numpy.dtype size changed")
@pytest.mark.filterwarnings("ignore:numpy.ufunc size changed")
@pytest.mark.filterwarnings("ignore:numpy.ndarray size changed")
@pytest.mark.filterwarnings("ignore:\\s+Port 8787 is already in use:UserWarning")
# def test_constant_subsequence_A_B_join_df_swap(dask_cluster):
def test_constant_subsequence_A_B_join_df_swap(dask_cluster):
    with Client(dask_cluster) as dask_client:
        T_A = np.concatenate(
            (np.zeros(10, dtype=np.float64), np.ones(10, dtype=np.float64))
        )
        T_B = np.concatenate(
            (np.zeros(20, dtype=np.float64), np.ones(5, dtype=np.float64))
        )
        m = 3
        ref_mp = naive.aamp(T_B, m, T_B=T_A)
        comp_mp = aamped(
            dask_client, pd.Series(T_B), m, pd.Series(T_A), ignore_trivial=False
        )
        naive.replace_inf(ref_mp)
        naive.replace_inf(comp_mp)
        npt.assert_almost_equal(ref_mp[:, 0], comp_mp[:, 0])  # ignore indices


@pytest.mark.filterwarnings("ignore:\\s+A large number of values are smaller")
@pytest.mark.filterwarnings("ignore:\\s+For a self-join")
@pytest.mark.filterwarnings("ignore:numpy.dtype size changed")
@pytest.mark.filterwarnings("ignore:numpy.ufunc size changed")
@pytest.mark.filterwarnings("ignore:numpy.ndarray size changed")
@pytest.mark.filterwarnings("ignore:\\s+Port 8787 is already in use:UserWarning")
def test_two_constant_subsequences_A_B_join(dask_cluster):
    with Client(dask_cluster) as dask_client:
        T_A = np.concatenate(
            (np.zeros(10, dtype=np.float64), np.ones(10, dtype=np.float64))
        )
        T_B = np.concatenate(
            (np.zeros(20, dtype=np.float64), np.ones(5, dtype=np.float64))
        )
        m = 3
        ref_mp = naive.aamp(T_A, m, T_B=T_B)
        comp_mp = aamped(dask_client, T_A, m, T_B, ignore_trivial=False)
        naive.replace_inf(ref_mp)
        naive.replace_inf(comp_mp)
        npt.assert_almost_equal(ref_mp[:, 0], comp_mp[:, 0])  # ignore indices


@pytest.mark.filterwarnings("ignore:\\s+A large number of values are smaller")
@pytest.mark.filterwarnings("ignore:\\s+For a self-join")
@pytest.mark.filterwarnings("ignore:numpy.dtype size changed")
@pytest.mark.filterwarnings("ignore:numpy.ufunc size changed")
@pytest.mark.filterwarnings("ignore:numpy.ndarray size changed")
@pytest.mark.filterwarnings("ignore:\\s+Port 8787 is already in use:UserWarning")
def test_two_constant_subsequences_A_B_join_df(dask_cluster):
    with Client(dask_cluster) as dask_client:
        T_A = np.concatenate(
            (np.zeros(10, dtype=np.float64), np.ones(10, dtype=np.float64))
        )
        T_B = np.concatenate(
            (np.zeros(20, dtype=np.float64), np.ones(5, dtype=np.float64))
        )
        m = 3
        ref_mp = naive.aamp(T_A, m, T_B=T_B)
        comp_mp = aamped(
            dask_client, pd.Series(T_A), m, pd.Series(T_B), ignore_trivial=False
        )
        naive.replace_inf(ref_mp)
        naive.replace_inf(comp_mp)
        npt.assert_almost_equal(ref_mp[:, 0], comp_mp[:, 0])  # ignore indices


@pytest.mark.filterwarnings("ignore:numpy.dtype size changed")
@pytest.mark.filterwarnings("ignore:numpy.ufunc size changed")
@pytest.mark.filterwarnings("ignore:numpy.ndarray size changed")
@pytest.mark.filterwarnings("ignore:\\s+Port 8787 is already in use:UserWarning")
@pytest.mark.parametrize("T_A, T_B", test_data)
@pytest.mark.parametrize(
    "substitution_location_A, substitution_location_B", two_substitution_locations
)
def test_aamped_two_subsequences_inf_A_B_join(
    T_A, T_B, substitution_location_A, substitution_location_B, dask_cluster
):
    with Client(dask_cluster) as dask_client:
        m = 3

        T_A_sub = T_A.copy()
        T_B_sub = T_B.copy()
        T_A_sub[substitution_location_A] = np.inf
        T_B_sub[substitution_location_B] = np.inf

        ref_mp = naive.aamp(T_A_sub, m, T_B=T_B_sub)
        comp_mp = aamped(dask_client, T_A_sub, m, T_B_sub, ignore_trivial=False)
        naive.replace_inf(ref_mp)
        naive.replace_inf(comp_mp)
        npt.assert_almost_equal(ref_mp, comp_mp)


@pytest.mark.filterwarnings("ignore:numpy.dtype size changed")
@pytest.mark.filterwarnings("ignore:numpy.ufunc size changed")
@pytest.mark.filterwarnings("ignore:numpy.ndarray size changed")
@pytest.mark.filterwarnings("ignore:\\s+Port 8787 is already in use:UserWarning")
@pytest.mark.parametrize("T_A, T_B", test_data)
@pytest.mark.parametrize(
    "substitution_location_A, substitution_location_B", two_substitution_locations
)
def test_aamped_two_subsequences_nan_A_B_join(
    T_A, T_B, substitution_location_A, substitution_location_B, dask_cluster
):
    with Client(dask_cluster) as dask_client:
        m = 3

        T_A_sub = T_A.copy()
        T_B_sub = T_B.copy()
        T_A_sub[substitution_location_A] = np.nan
        T_B_sub[substitution_location_B] = np.nan

        ref_mp = naive.aamp(T_A_sub, m, T_B=T_B_sub)
        comp_mp = aamped(dask_client, T_A_sub, m, T_B_sub, ignore_trivial=False)
        naive.replace_inf(ref_mp)
        naive.replace_inf(comp_mp)
        npt.assert_almost_equal(ref_mp, comp_mp)


@pytest.mark.filterwarnings("ignore:numpy.dtype size changed")
@pytest.mark.filterwarnings("ignore:numpy.ufunc size changed")
@pytest.mark.filterwarnings("ignore:numpy.ndarray size changed")
@pytest.mark.filterwarnings("ignore:\\s+Port 8787 is already in use:UserWarning")
@pytest.mark.parametrize("T_A, T_B", test_data)
@pytest.mark.parametrize(
    "substitution_location_A, substitution_location_B", two_substitution_locations
)
def test_aamped_two_subsequences_nan_inf_A_B_join(
    T_A, T_B, substitution_location_A, substitution_location_B, dask_cluster
):
    with Client(dask_cluster) as dask_client:
        m = 3

        T_A_sub = T_A.copy()
        T_B_sub = T_B.copy()
        T_A_sub[substitution_location_A] = np.nan
        T_B_sub[substitution_location_B] = np.inf

        ref_mp = naive.aamp(T_A_sub, m, T_B=T_B_sub)
        comp_mp = aamped(dask_client, T_A_sub, m, T_B_sub, ignore_trivial=False)
        naive.replace_inf(ref_mp)
        naive.replace_inf(comp_mp)
        npt.assert_almost_equal(ref_mp, comp_mp)


@pytest.mark.filterwarnings("ignore:numpy.dtype size changed")
@pytest.mark.filterwarnings("ignore:numpy.ufunc size changed")
@pytest.mark.filterwarnings("ignore:numpy.ndarray size changed")
@pytest.mark.filterwarnings("ignore:\\s+Port 8787 is already in use:UserWarning")
@pytest.mark.parametrize("T_A, T_B", test_data)
@pytest.mark.parametrize(
    "substitution_location_A, substitution_location_B", two_substitution_locations
)
def test_aamped_two_subsequences_nan_inf_A_B_join_swap(
    T_A, T_B, substitution_location_A, substitution_location_B, dask_cluster
):
    with Client(dask_cluster) as dask_client:
        m = 3

        T_A_sub = T_A.copy()
        T_B_sub = T_B.copy()
        T_A_sub[substitution_location_A] = np.inf
        T_B_sub[substitution_location_B] = np.nan

        ref_mp = naive.aamp(T_A_sub, m, T_B=T_B_sub)
        comp_mp = aamped(dask_client, T_A_sub, m, T_B_sub, ignore_trivial=False)
        naive.replace_inf(ref_mp)
        naive.replace_inf(comp_mp)
        npt.assert_almost_equal(ref_mp, comp_mp)


@pytest.mark.filterwarnings("ignore:numpy.dtype size changed")
@pytest.mark.filterwarnings("ignore:numpy.ufunc size changed")
@pytest.mark.filterwarnings("ignore:numpy.ndarray size changed")
@pytest.mark.filterwarnings("ignore:\\s+Port 8787 is already in use:UserWarning")
@pytest.mark.parametrize("T_A, T_B", test_data)
def test_aamped_self_join_KNN(T_A, T_B, dask_cluster):
    with Client(dask_cluster) as dask_client:
        m = 3
        for k in range(2, 4):
            for p in [1.0, 2.0, 3.0]:
                ref_mp = naive.aamp(T_B, m, p=p, k=k)
                comp_mp = aamped(dask_client, T_B, m, p=p, k=k)
                naive.replace_inf(ref_mp)
                naive.replace_inf(comp_mp)
                npt.assert_almost_equal(ref_mp, comp_mp)


@pytest.mark.filterwarnings("ignore:numpy.dtype size changed")
@pytest.mark.filterwarnings("ignore:numpy.ufunc size changed")
@pytest.mark.filterwarnings("ignore:numpy.ndarray size changed")
@pytest.mark.filterwarnings("ignore:\\s+Port 8787 is already in use:UserWarning")
@pytest.mark.parametrize("T_A, T_B", test_data)
def test_aamped_A_B_join_KNN(T_A, T_B, dask_cluster):
    with Client(dask_cluster) as dask_client:
        m = 3
        for k in range(2, 4):
            for p in [1.0, 2.0, 3.0]:
                ref_mp = naive.aamp(T_A, m, T_B=T_B, p=p, k=k)
                comp_mp = aamped(
                    dask_client, T_A, m, T_B, ignore_trivial=False, p=p, k=k
                )
                naive.replace_inf(ref_mp)
                naive.replace_inf(comp_mp)
                npt.assert_almost_equal(ref_mp, comp_mp)
