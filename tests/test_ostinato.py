import numpy as np
import numpy.testing as npt
import stumpy
import naive
from stumpy.ostinato import _get_central_motif
import pytest


def naive_across_series_nearest_neighbors(Ts, Ts_idx, subseq_idx, m):
    """
    For multiple time series find, per individual time series, the subsequences closest
    to a query.

    Parameters
    ----------
    Ts : list
        A list of time series for which to find the nearest neighbor subsequences that
        are closest to the query subsequence `Ts[Ts_idx][subseq_idx : subseq_idx + m]`

    Ts_idx : int
        The index of time series in `Ts` which contains the query subsequence
        `Ts[Ts_idx][subseq_idx : subseq_idx + m]`

    subseq_idx : int
        The subsequence index in the time series `Ts[Ts_idx]` that contains the query
        subsequence `Ts[Ts_idx][subseq_idx : subseq_idx + m]`

    m : int
        Subsequence window size

    Returns
    -------
    nns_radii : ndarray
        Nearest neighbor radii to subsequences in `Ts` that are closest to the query
        `Ts[Ts_idx][subseq_idx : subseq_idx + m]`

    nns_subseq_idx : ndarray
        Nearest neighbor indices to subsequences in `Ts` that are closest to the query
        `Ts[Ts_idx][subseq_idx : subseq_idx + m]`
    """
    k = len(Ts)
    Q = Ts[Ts_idx][subseq_idx : subseq_idx + m]
    nns_radii = np.zeros(k, dtype=np.float64)
    nns_subseq_idx = np.zeros(k, dtype=np.int64)

    for i in range(k):
        distance_profile = naive.distance_profile(Q, Ts[i], len(Q))
        nns_subseq_idx[i] = np.argmin(distance_profile)
        nns_radii[i] = distance_profile[nns_subseq_idx[i]]

    return nns_radii, nns_subseq_idx


def naive_get_central_motif(Ts, bsf_radius, bsf_Ts_idx, bsf_subseq_idx, m):
    """
    Compare subsequences with the same radius and return the most central motif

    Parameters
    ----------
    Ts : list
        List of time series for which to find the most central motif

    bsf_radius : float
        Best radius found by a consensus search algorithm

    bsf_Ts_idx : int
        Index of time series in which `radius` was first found

    bsf_subseq_idx : int
        Start index of the subsequence in `Ts[Ts_idx]` that has radius `radius`

    m : int
        Window size

    Returns
    -------
    bsf_radius : float
        The updated radius of the most central consensus motif

    bsf_Ts_idx : int
        The updated index of time series which contains the most central consensus motif

    bsf_subseq_idx : int
        The update subsequence index of most central consensus motif within the time
        series `bsf_Ts_idx` that contains it
    """
    bsf_nns_radii, bsf_nns_subseq_idx = naive_across_series_nearest_neighbors(
        Ts, bsf_Ts_idx, bsf_subseq_idx, m
    )
    bsf_nns_mean_radii = bsf_nns_radii.mean()

    candidate_nns_Ts_idx = np.flatnonzero(np.isclose(bsf_nns_radii, bsf_radius))
    candidate_nns_subseq_idx = bsf_nns_subseq_idx[candidate_nns_Ts_idx]

    for Ts_idx, subseq_idx in zip(candidate_nns_Ts_idx, candidate_nns_subseq_idx):
        candidate_nns_radii, _ = naive_across_series_nearest_neighbors(
            Ts, Ts_idx, subseq_idx, m
        )
        if (
            np.isclose(candidate_nns_radii.max(), bsf_radius)
            and candidate_nns_radii.mean() < bsf_nns_mean_radii
        ):
            bsf_Ts_idx = Ts_idx
            bsf_subseq_idx = subseq_idx
            bsf_nns_mean_radii = candidate_nns_radii.mean()

    return bsf_radius, bsf_Ts_idx, bsf_subseq_idx


def naive_consensus_search(Ts, m):
    """
    Brute force consensus motif from
    <https://www.cs.ucr.edu/~eamonn/consensus_Motif_ICDM_Long_version.pdf>

    See Table 1

    Note that there is a bug in the pseudocode at line 8 where `i` should be `j`.
    This implementation fixes it.
    """
    k = len(Ts)

    bsf_radius = np.inf
    bsf_Ts_idx = 0
    bsf_subseq_idx = 0

    for j in range(k):
        radii = np.zeros(len(Ts[j]) - m + 1)
        for i in range(k):
            if i != j:
                mp = naive.stump(Ts[j], m, Ts[i])
                radii = np.maximum(radii, mp[:, 0])
        min_radius_idx = np.argmin(radii)
        min_radius = radii[min_radius_idx]
        if min_radius < bsf_radius:
            bsf_radius = min_radius
            bsf_Ts_idx = j
            bsf_subseq_idx = min_radius_idx

    return bsf_radius, bsf_Ts_idx, bsf_subseq_idx


@pytest.mark.parametrize(
    "seed", np.random.choice(np.arange(10000), size=100, replace=False)
)
def test_random_ostinato(seed):
    m = 50
    np.random.seed(seed)
    Ts = [np.random.rand(n) for n in [64, 128, 256]]

    bsf_radius, bsf_Ts_idx, bsf_subseq_idx = naive_consensus_search(Ts, m)
    ref_radius, ref_Ts_idx, ref_subseq_idx = naive_get_central_motif(
        Ts, bsf_radius, bsf_Ts_idx, bsf_subseq_idx, m
    )
    comp_radius, comp_Ts_idx, comp_subseq_idx = stumpy.ostinato(Ts, m)

    npt.assert_almost_equal(ref_radius, comp_radius)
    npt.assert_almost_equal(ref_Ts_idx, comp_Ts_idx)
    npt.assert_almost_equal(ref_subseq_idx, comp_subseq_idx)


@pytest.mark.parametrize("seed", [79, 109, 112, 133, 151, 161, 251, 275, 309, 355])
def test_deterministic_ostinato(seed):
    m = 50
    np.random.seed(seed)
    Ts = [np.random.rand(n) for n in [64, 128, 256]]

    bsf_radius, bsf_Ts_idx, bsf_subseq_idx = naive_consensus_search(Ts, m)
    ref_radius, ref_Ts_idx, ref_subseq_idx = naive_get_central_motif(
        Ts, bsf_radius, bsf_Ts_idx, bsf_subseq_idx, m
    )
    comp_radius, comp_Ts_idx, comp_subseq_idx = stumpy.ostinato(Ts, m)

    npt.assert_almost_equal(ref_radius, comp_radius)
    npt.assert_almost_equal(ref_Ts_idx, comp_Ts_idx)
    npt.assert_almost_equal(ref_subseq_idx, comp_subseq_idx)
