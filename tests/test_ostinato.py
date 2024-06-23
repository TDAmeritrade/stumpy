import functools

import naive
import numpy as np
import numpy.testing as npt
import pytest
from dask.distributed import Client, LocalCluster

import stumpy


@pytest.fixture(scope="module")
def dask_cluster():
    cluster = LocalCluster(
        n_workers=2,
        threads_per_worker=2,
        dashboard_address=None,
        worker_dashboard_address=None,
    )
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


def test_input_not_overwritten_ostinato():
    # ostinato preprocesses its input, a list of time series,
    # by replacing nan value with 0 in each time series.
    # This test ensures that the original input is not overwritten
    m = 50
    Ts = [np.random.rand(n) for n in [64, 128, 256]]
    for T in Ts:
        T[0] = np.nan

    # raise error if ostinato overwrite its input
    Ts_input = [T.copy() for T in Ts]
    stumpy.ostinato(Ts_input, m)
    for i in range(len(Ts)):
        T_ref = Ts[i]
        T_comp = Ts_input[i]
        npt.assert_almost_equal(T_ref[np.isfinite(T_ref)], T_comp[np.isfinite(T_comp)])


def test_extract_several_consensus_ostinato():
    # This test is to further ensure that the function `ostinato`
    # does not tamper with the original data.
    Ts = [np.random.rand(n) for n in [256, 512, 1024]]
    Ts_ref = [T.copy() for T in Ts]
    Ts_comp = [T.copy() for T in Ts]

    m = 20

    k = 5  # Get the first `k` consensus motifs
    for _ in range(k):
        # Find consensus motif and its NN in each time series in Ts_comp
        # Remove them from Ts_comp as well as Ts_ref, and assert that the
        # two time series are the same
        radius, Ts_idx, subseq_idx = stumpy.ostinato(Ts_comp, m)
        consensus_motif = Ts_comp[Ts_idx][subseq_idx : subseq_idx + m].copy()
        for i in range(len(Ts_comp)):
            if i == Ts_idx:
                query_idx = subseq_idx
            else:
                query_idx = None

            idx = np.argmin(
                stumpy.core.mass(consensus_motif, Ts_comp[i], query_idx=query_idx)
            )
            Ts_comp[i][idx : idx + m] = np.nan
            Ts_ref[i][idx : idx + m] = np.nan

            npt.assert_almost_equal(
                Ts_ref[i][np.isfinite(Ts_ref[i])], Ts_comp[i][np.isfinite(Ts_comp[i])]
            )


def test_input_not_overwritten_ostinatoed(dask_cluster):
    # ostinatoed preprocesses its input, a list of time series,
    # by replacing nan value with 0 in each time series.
    # This test ensures that the original input is not overwritten
    with Client(dask_cluster) as dask_client:
        m = 50
        Ts = [np.random.rand(n) for n in [64, 128, 256]]
        for T in Ts:
            T[0] = np.nan

        # raise error if ostinato overwrite its input
        Ts_input = [T.copy() for T in Ts]
        stumpy.ostinatoed(dask_client, Ts_input, m)
        for i in range(len(Ts)):
            T_ref = Ts[i]
            T_comp = Ts_input[i]
            npt.assert_almost_equal(
                T_ref[np.isfinite(T_ref)], T_comp[np.isfinite(T_comp)]
            )


def test_extract_several_consensus_ostinatoed(dask_cluster):
    # This test is to further ensure that the function `ostinatoed`
    # does not tamper with the original data.
    Ts = [np.random.rand(n) for n in [256, 512, 1024]]
    Ts_ref = [T.copy() for T in Ts]
    Ts_comp = [T.copy() for T in Ts]

    m = 20

    with Client(dask_cluster) as dask_client:
        k = 5  # Get the first `k` consensus motifs
        for _ in range(k):
            # Find consensus motif and its NN in each time series in Ts_comp
            # Remove them from Ts_comp as well as Ts_ref, and assert that the
            # two time series are the same
            radius, Ts_idx, subseq_idx = stumpy.ostinatoed(dask_client, Ts_comp, m)
            consensus_motif = Ts_comp[Ts_idx][subseq_idx : subseq_idx + m].copy()
            for i in range(len(Ts_comp)):
                if i == Ts_idx:
                    query_idx = subseq_idx
                else:
                    query_idx = None

                idx = np.argmin(
                    stumpy.core.mass(consensus_motif, Ts_comp[i], query_idx=query_idx)
                )
                Ts_comp[i][idx : idx + m] = np.nan
                Ts_ref[i][idx : idx + m] = np.nan

                npt.assert_almost_equal(
                    Ts_ref[i][np.isfinite(Ts_ref[i])],
                    Ts_comp[i][np.isfinite(Ts_comp[i])],
                )
