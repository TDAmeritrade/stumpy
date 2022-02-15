import numpy as np
import logging

# from .aamp_motifs import aamp_motifs
from . import core, config, mdl, match

logger = logging.getLogger(__name__)


# @core.non_normalized(aamp_motifs)
def mmotifs(
    T: np.ndarray,
    P: np.ndarray,
    I: np.ndarray,
    min_neighbors: int = 1,
    max_distance: float = None,
    cutoffs: np.ndarray = None,
    max_matches: int = 10,
    max_motifs: int = 1,
    atol: float = 1e-8,
    # normalize: bool = True,
    # p: float = 2.0
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

    cutoffs: numpy.ndarray or float, default None
        The largest matrix profile value (distance) for each dimension of the
        multidimensional matrix profile that a multidimenisonal candidate motif is
        allowed to have.
        If cutoffs is only one value, these value will be applied to every dimension.

    max_matches: int, default 10
        The maximum amount of similar matches (nearest neighbors) of a motif
        representative to be returned.
        The first match is always the self-match for each motif.

    max_motifs: int, default 1
        The maximum number of motifs to return

    atol : float, default 1e-8
        The absolute tolerance parameter. This value will be added to `max_distance`
        when comparing distances between subsequences.

    normalize : bool, default True
        When set to `True`, this z-normalizes subsequences prior to computing distances.
        Otherwise, this function gets re-routed to its complementary non-normalized
        equivalent set in the `@core.non_normalized` function decorator.

    p : float, default 2.0
        The p-norm to apply for computing the Minkowski distance. This parameter is
        ignored when `normalize == False`.

    Returns
    -------
    motif_distances: numpy.ndarray
        The distances corresponding to a set of subsequence matches for each motif.

    motif_indices: numpy.ndarray
        The indices corresponding to a set of subsequences matches for each motif.

    motif_subspaces: numpy.ndarray
        A numpy.ndarray consisting of arrays that contain the `k`-dimensional
        subspace for each motif.

    motif_mdls: numpy.ndarray
        A numpy.ndarray consisting of arrays that contain the mdl results for
        finding the dimension of each motif
    """
    # Convert dataframe to array if necessary
    T = core._preprocess(T)

    if max_motifs < 1:  # pragma: no cover
        logger.warning(
            "The maximum number of motifs, `max_motifs`, "
            "must be greater than or equal to 1"
        )
        logger.warning("`max_motifs` has been set to `1`")
        max_motifs = 1

    # Calculate subsequence length
    m = T.shape[-1] - P.shape[-1] + 1

    # Calculate exclusion zone
    excl_zone = int(np.ceil(m / config.STUMPY_EXCL_ZONE_DENOM))

    # Set default max_matches value
    if max_matches is None:
        max_matches = np.inf

    # Calculate default cutoff value
    if cutoffs is None:
        cutoffs = []
        P_copy = P.copy().astype(np.float64)
        P_copy[np.isinf(P_copy)] = np.nan
        for i in range(P_copy.shape[0]):
            cutoffs.append(
                np.nanmax(
                    [
                        np.nanmean(P_copy[i, :] - 2.0 * np.nanstd(P_copy[i, :])),
                        np.nanmin(P_copy[i, :]),
                    ]
                )
            )
    elif isinstance(cutoffs, int) or isinstance(cutoffs, float):
        # cutoffs is a single value -> apply value to each MP dimension
        cutoffs_tmp = cutoffs
        cutoffs = []
        cutoffs.extend(cutoffs_tmp for _ in range(T.shape[0]))

    # Precompute rolling means and standard deviations
    T, M_T, Σ_T = core.preprocess(T, m)

    # Allocate memory for returns
    motif_distances = []
    motif_indices = []
    motif_subspaces = []
    motif_mdls = []

    # Identify the first multidimensional motif as candidate motif and get its
    # nearest neighbor
    candidate_idx = np.argmin(P, axis=1)
    nn_idx = I[np.arange(len(candidate_idx)), candidate_idx]

    # Iterate over each motif
    while len(motif_distances) < max_motifs:

        # Choose dimension k using MDL
        mdls, subspaces = mdl(T, m, candidate_idx, nn_idx)
        k = np.argmin(mdls)
        motif_mdls.append(mdls)

        # Choose cutoff value for the `k`-th dimension
        cutoff = cutoffs[k]

        # Get k-dimensional motif
        motif_idx = candidate_idx[k]
        motif_value = P[k, motif_idx]
        if motif_value > cutoff or not np.isfinite(motif_value):
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
        M_T_sub = M_T[subspaces[k]]
        Σ_T_sub = Σ_T[subspaces[k]]

        Q = sub_dims[:, motif_idx : motif_idx + m]
        query_matches = match(
            Q=Q,
            T=sub_dims,
            M_T=M_T_sub,
            Σ_T=Σ_T_sub,
            max_matches=max_matches,
            max_distance=max_distance,
            atol=atol,
            # normalize=normalize,
            # p=p
        )

        if len(query_matches) > min_neighbors:
            motif_distances.append(query_matches[:, 0])
            motif_indices.append(query_matches[:, 1])

        # Set exclusion zones and find new candidate_idx an nn_idx for
        # the next motif
        for idx in query_matches[:, 1]:
            core.apply_exclusion_zone(P, idx, excl_zone, np.inf)
        candidate_idx = np.argmin(P, axis=1)
        nn_idx = I[np.arange(len(candidate_idx)), candidate_idx]

    motif_distances = core._jagged_list_to_array(
        motif_distances, fill_value=np.nan, dtype=np.float64
    )
    motif_indices = core._jagged_list_to_array(
        motif_indices, fill_value=-1, dtype=np.int64
    )

    return (
        motif_distances,
        motif_indices,
        np.array(motif_subspaces, dtype=object),
        np.array(motif_mdls, dtype=object),
    )
