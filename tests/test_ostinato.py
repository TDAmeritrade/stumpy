import functools

import naive
import numpy as np
import numpy.testing as npt
import pytest
from dask.distributed import Client, LocalCluster

import stumpy


@pytest.fixture(scope="module")
def dask_cluster():
    cluster = LocalCluster(n_workers=2, threads_per_worker=2)
    yield cluster
    cluster.close()


@pytest.mark.parametrize(
    "seed", np.random.choice(np.arange(10000), size=25, replace=False)
)
def test_random_ostinato(seed):
    m = 50
    np.random.seed(seed)
    Ts = [np.random.rand(n) for n in [64, 128, 256]]

    ref_radius, ref_Ts_idx, ref_subseq_idx = naive.ostinato(Ts, m)
    comp_radius, comp_Ts_idx, comp_subseq_idx = stumpy.ostinato(Ts, m)

    npt.assert_almost_equal(ref_radius, comp_radius)
    npt.assert_almost_equal(ref_Ts_idx, comp_Ts_idx)
    npt.assert_almost_equal(ref_subseq_idx, comp_subseq_idx)


@pytest.mark.parametrize("seed", [79, 109, 112, 133, 151, 161, 251, 275, 309, 355])
def test_deterministic_ostinato(seed):
    m = 50
    np.random.seed(seed)
    Ts = [np.random.rand(n) for n in [64, 128, 256]]

    ref_radius, ref_Ts_idx, ref_subseq_idx = naive.ostinato(Ts, m)
    comp_radius, comp_Ts_idx, comp_subseq_idx = stumpy.ostinato(Ts, m)

    npt.assert_almost_equal(ref_radius, comp_radius)
    npt.assert_almost_equal(ref_Ts_idx, comp_Ts_idx)
    npt.assert_almost_equal(ref_subseq_idx, comp_subseq_idx)


@pytest.mark.parametrize(
    "seed", np.random.choice(np.arange(10000), size=25, replace=False)
)
def test_random_ostinatoed(seed, dask_cluster):
    with Client(dask_cluster) as dask_client:
        m = 50
        np.random.seed(seed)
        Ts = [np.random.rand(n) for n in [64, 128, 256]]

        ref_radius, ref_Ts_idx, ref_subseq_idx = naive.ostinato(Ts, m)
        comp_radius, comp_Ts_idx, comp_subseq_idx = stumpy.ostinatoed(
            dask_client, Ts, m
        )

        npt.assert_almost_equal(ref_radius, comp_radius)
        npt.assert_almost_equal(ref_Ts_idx, comp_Ts_idx)
        npt.assert_almost_equal(ref_subseq_idx, comp_subseq_idx)


@pytest.mark.parametrize("seed", [79, 109, 112, 133, 151, 161, 251, 275, 309, 355])
def test_deterministic_ostinatoed(seed, dask_cluster):
    with Client(dask_cluster) as dask_client:
        m = 50
        np.random.seed(seed)
        Ts = [np.random.rand(n) for n in [64, 128, 256]]

        ref_radius, ref_Ts_idx, ref_subseq_idx = naive.ostinato(Ts, m)
        comp_radius, comp_Ts_idx, comp_subseq_idx = stumpy.ostinatoed(
            dask_client, Ts, m
        )

        npt.assert_almost_equal(ref_radius, comp_radius)
        npt.assert_almost_equal(ref_Ts_idx, comp_Ts_idx)
        npt.assert_almost_equal(ref_subseq_idx, comp_subseq_idx)


@pytest.mark.parametrize(
    "seed", np.random.choice(np.arange(10000), size=25, replace=False)
)
def test_random_ostinato_with_isconstant(seed):
    isconstant_custom_func = functools.partial(
        naive.isconstant_func_stddev_threshold, quantile_threshold=0.05
    )

    m = 50
    np.random.seed(seed)
    Ts = [np.random.rand(n) for n in [64, 128, 256]]
    Ts_subseq_isconstant = [isconstant_custom_func for _ in range(len(Ts))]

    ref_radius, ref_Ts_idx, ref_subseq_idx = naive.ostinato(
        Ts, m, Ts_subseq_isconstant=Ts_subseq_isconstant
    )
    comp_radius, comp_Ts_idx, comp_subseq_idx = stumpy.ostinato(
        Ts, m, Ts_subseq_isconstant=Ts_subseq_isconstant
    )

    npt.assert_almost_equal(ref_radius, comp_radius)
    npt.assert_almost_equal(ref_Ts_idx, comp_Ts_idx)
    npt.assert_almost_equal(ref_subseq_idx, comp_subseq_idx)


@pytest.mark.parametrize("seed", [79, 109, 112, 133, 151, 161, 251, 275, 309, 355])
def test_deterministic_ostinatoed_with_isconstant(seed, dask_cluster):
    isconstant_custom_func = functools.partial(
        naive.isconstant_func_stddev_threshold, quantile_threshold=0.05
    )

    with Client(dask_cluster) as dask_client:
        m = 50
        np.random.seed(seed)
        Ts = [np.random.rand(n) for n in [64, 128, 256]]

        l = 64 - m + 1
        subseq_isconsant = np.full(l, 0, dtype=bool)
        subseq_isconsant[np.random.randint(0, l)] = True
        Ts_subseq_isconstant = [
            subseq_isconsant,
            None,
            isconstant_custom_func,
        ]

        ref_radius, ref_Ts_idx, ref_subseq_idx = naive.ostinato(
            Ts, m, Ts_subseq_isconstant=Ts_subseq_isconstant
        )
        comp_radius, comp_Ts_idx, comp_subseq_idx = stumpy.ostinatoed(
            dask_client, Ts, m, Ts_subseq_isconstant=Ts_subseq_isconstant
        )

        npt.assert_almost_equal(ref_radius, comp_radius)
        npt.assert_almost_equal(ref_Ts_idx, comp_Ts_idx)
        npt.assert_almost_equal(ref_subseq_idx, comp_subseq_idx)


@pytest.mark.parametrize(
    "seed", np.random.choice(np.arange(10000), size=25, replace=False)
)
def test_input_not_overwritten(seed):
    m = 50
    np.random.seed(seed)
    Ts = [np.random.rand(n) for n in [64, 128, 256]]

    # without nan values
    Ts_ref = [T.copy() for T in Ts]
    Ts_comp = [T.copy() for T in Ts]
    stumpy.ostinato(Ts_comp, m)
    for i in range(len(Ts)):
        npt.assert_almost_equal(Ts_ref[i], Ts_comp[i])

    # with nan values
    nan_size = [np.random.choice(np.arange(1, len(T) + 1)) for T in Ts]
    for i in range(len(Ts)):
        IDX = np.random.choice(np.arange(len(Ts[i])), size=nan_size[i], replace=False)
        Ts[i][IDX] = np.nan

    Ts_ref = [T.copy() for T in Ts]
    Ts_comp = [T.copy() for T in Ts]
    stumpy.ostinato(Ts_comp, m)
    for i in range(len(Ts)):
        T_ref = Ts_ref[i]
        T_comp = Ts_comp[i]
        npt.assert_almost_equal(T_ref[np.isfinite(T_ref)], T_comp[np.isfinite(T_comp)])


@pytest.mark.parametrize(
    "seed", np.random.choice(np.arange(10000), size=25, replace=False)
)
def test_extract_second_consensus(seed):
    m = 50
    np.random.seed(seed)
    Ts = [np.random.rand(n) for n in [256, 512, 1024]]
    Ts_ref = [T.copy() for T in Ts]
    Ts_comp = [T.copy() for T in Ts]

    # obtain first consensus motif
    central_radius, central_Ts_idx, central_subseq_idx = stumpy.ostinato(Ts_comp, m)

    consensus_motif = Ts_comp[central_Ts_idx][
        central_subseq_idx : central_subseq_idx + m
    ].copy()
    Ts_comp[central_Ts_idx][central_subseq_idx : central_subseq_idx + m] = np.nan
    Ts_ref[central_Ts_idx][
        central_subseq_idx : central_subseq_idx + m
    ] = np.nan  # apply same changes to Ts_ref
    for i in range(len(Ts)):
        if i != central_Ts_idx:
            D = stumpy.core.mass(consensus_motif, Ts_comp[i])
            idx = np.argmin(D)
            Ts_comp[i][idx : idx + m] = np.nan
            Ts_ref[i][idx : idx + m] = np.nan  # apply same changes to Ts_ref

    # obtain second consensus motif
    consensus_radius_comp, consensus_Ts_idx_comp, consensus_subseq_idx_comp = (
        stumpy.ostinato(Ts_comp, m)
    )

    # obtain first consensus motif from Ts_ref where some subsequences
    # are removed based on the first consensus motif
    consensus_radius_ref, consensus_Ts_idx_ref, consensus_subseq_idx_ref = (
        stumpy.ostinato(Ts_ref, m)
    )

    np.testing.assert_almost_equal(consensus_radius_ref, consensus_radius_comp)
    np.testing.assert_almost_equal(consensus_Ts_idx_ref, consensus_Ts_idx_comp)
    np.testing.assert_almost_equal(consensus_subseq_idx_ref, consensus_subseq_idx_comp)
