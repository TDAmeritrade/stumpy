import numpy as np
import logging
from . import core, config, mdl, match

logger = logging.getLogger(__name__)


def mmotifs(
    T: np.ndarray,
    P: np.ndarray,
    I: np.ndarray,
    max_matches: int = 10,
    max_motifs: int = 1,
    cutoff: np.ndarray = None,
    max_distance: float = None,
    min_neighbors: int = 1,
    atol: float = 1e-8,
):
    """
    Discover the top motifs for the multi-dimensional time series `T`

    Parameters
    ----------
    T: numpy.ndarray
        The multi-dimensional time series or sequence

    P: numpy.ndarray
        Multi-dimensional Matrix Profile of T

    I: numpy.ndarray
        Multi-dimensional Matrix Profile indices

    min_neighbors : int, default 1
        The minimum number of similar matches a subsequence needs to have in order
        to be considered a motif. This defaults to `1`, which means that a subsequence
        must have at least one similar match in order to be considered a motif.

    max_distance: flaot, default None
        Maximal distance that is allowed between a query subsequence
        (a candidate motif) and all subsequences in T to be considered as a match.
        If None, this defaults to
        `np.nanmax([np.nanmean(D) - 2 * np.nanstd(D), np.nanmin(D)])`
        (i.e. at least the closest match will be returned).

    cutoff: numpy.ndarray or float, default None
        The largest matrix profile value (distance) for each dimension of the
        multidimensional matrix profile that a multidimenisonal candidate motif is
        allowed to have.
        If cutoff is only one value, these value will be applied to every dimension.

    max_matches: int, default 10
        The maximum amount of similar matches (nearest neighbors) of a motif
        representative to be returned.
        The first match is always the self-match for each motif.

    max_motifs: int, default 1
        The maximum number of motifs to return

    atol : float, default 1e-8
        The absolute tolerance parameter. This value will be added to `max_distance`
        when comparing distances between subsequences.

    Returns
    -------
    motif_matches_distances: numpy.ndarray
        The distances corresponding to a set of subsequence matches for each motif.

    motif_matches_indices: numpy.ndarray
        The indices corresponding to a set of subsequences matches for each motif.

    motif_subspaces: numpy.ndarray
        A numpy.ndarray consisting of arrays that contain the `k`-dimensional
        subspace for each motif.

    motif_mdls: numpy.ndarray
        A numpy.ndarray consisting of arrays that contain the mdl results for
        finding the dimension of each motif
    """
    if max_motifs < 1:  # pragma: no cover
        logger.warning(
            "The maximum number of motifs, `max_motifs`, "
            "must be greater than or equal to 1"
        )
        logger.warning("`max_motifs` has been set to `1`")
        max_motifs = 1

    # Calculate subsequence length
    T = core._preprocess(T)  # convert dataframe to array if necessary
    m = T.shape[-1] - P.shape[-1] + 1

    # Calculate exclusion zone
    excl_zone = int(np.ceil(m / config.STUMPY_EXCL_ZONE_DENOM))

    # Precompute rolling means and standard deviations
    # T, mean_T, sigma_T = core.preprocess(T, m)

    if max_matches is None:
        max_matches = np.inf

    motif_matches_distances = []
    motif_matches_indices = []
    motif_subspaces = []
    motif_mdls = []

    # Identify the first multidimensional motif as candidate motif and get its
    # nearest neighbor
    candidate_idx = np.argsort(P, axis=1)[:, 0]
    nn_idx = I[np.arange(len(candidate_idx)), candidate_idx]

    # Iterate over each motif
    while len(motif_matches_distances) < max_motifs:

        # Choose dimension k using MDL
        mdls, subspaces = mdl(T, m, candidate_idx, nn_idx)
        k = np.argmin(mdls)
        motif_mdls.append(mdls)

        # Calculate default cutoff value
        if cutoff is None:
            P_copy = P.copy().astype(np.float64)
            P_copy[np.isinf(P_copy)] = np.nan
            P_copy = P_copy[k, :]
            cutoff_var = np.nanmax(
                [np.nanmean(P_copy - 2.0 * np.nanstd(P_copy)), np.nanmin(P_copy)]
            )
            cutoff = None
        else:
            if isinstance(cutoff, np.ndarray) or isinstance(cutoff, list):
                cutoff_var = cutoff[k]
            else:
                cutoff_var = cutoff

        # Get k-dimensional motif
        motif_idx = candidate_idx[k]
        motif_value = P[k, motif_idx]
        if motif_value > cutoff_var or not np.isfinite(motif_value):
            break

        # Compute subspace and find related dimensions
        motif_subspaces.append(subspaces[k])
        sub_dims = T[subspaces[k]]

        # Stop iteration if max_distance is a constant and the k-dim.
        # matrix profile value is larger than the maximum distance.
        if isinstance(max_distance, float) and motif_value > max_distance:
            break

        # Get multidimensional Distance Profile to find the max_matches
        # nearest neighbors
        T_sub, mean_T, sigma_T = core.preprocess(sub_dims, m)

        Q = sub_dims[:, motif_idx : motif_idx + m]
        query_matches = match(
            Q=Q,
            T=sub_dims,
            M_T=mean_T,
            Σ_T=sigma_T,
            max_matches=max_matches,
            max_distance=max_distance,
            atol=atol,
        )

        if len(query_matches) > min_neighbors:
            motif_matches_distances.append(query_matches[:, 0])
            motif_matches_indices.append(query_matches[:, 1])

        # Set exclusion zones and find new candidate_idx an nn_idx for
        # the next motif
        for idx in query_matches[:, 1]:
            core.apply_exclusion_zone(P, idx, excl_zone, np.inf)
        candidate_idx = np.argsort(P, axis=1)[:, 0]
        nn_idx = I[np.arange(len(candidate_idx)), candidate_idx]

    motif_matches_distances = core._jagged_list_to_array(
        motif_matches_distances, fill_value=np.nan, dtype=np.float64
    )
    motif_matches_indices = core._jagged_list_to_array(
        motif_matches_indices, fill_value=-1, dtype=np.int64
    )

    return (
        motif_matches_distances,
        motif_matches_indices,
        np.array(motif_subspaces, dtype=object),
        np.array(motif_mdls, dtype=object),
    )
