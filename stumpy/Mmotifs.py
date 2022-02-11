import numpy as np
import logging
import stumpy
import matplotlib.pyplot as plt
from . import core
from . import config
from stumpy.core import _preprocess

logger = logging.getLogger(__name__)


def Mmotifs(
    T: np.ndarray,
    P: np.ndarray,
    I: np.ndarray,
    max_matches: int = 10,
    max_motifs: int = 1,
    cutoff: np.ndarray = None,
    visualize_mdl: bool = False,
    max_distance: float = None,
    min_neighbors: int = 1,
    atol: float = 1e-8
):
    """
    Discover the top k multidimensional motifs for the time series T

    Parameters
    ----------
    T: numpy.ndarray
        The multidimensional time series or sequence

    P: numpy.ndarray
        Multidimensional Matrix Profile of T

    I: numpy.ndarray
        Matrix Profile indices

    max_matches: int, default 10
        The maximum amount of similar matches (nearest neighbors) of a motif
        representative to be returned.
        The first match is always the self-match for each motif.

    max_distance: flaot, default None
        Maximal distance that is allowed between a query subsequence
        (a candidate motif) and all subsequences in T to be considered as a match.
        If None, this defaults to
        `np.nanmax([np.nanmean(D) - 2 * np.nanstd(D), np.nanmin(D)])`
        (i.e. at least the closest match will be returned).

    min_neighbors : int, default 1
        The minimum number of similar matches a subsequence needs to have in order
        to be considered a motif. This defaults to `1`, which means that a subsequence
        must have at least one similar match in order to be considered a motif.    

    atol : float, default 1e-8
        The absolute tolerance parameter. This value will be added to `max_distance`
        when comparing distances between subsequences.

    max_motifs: int, default 1
        The maximum number of motifs to return

    cutoff: numpy.ndarray, default None
    The largest matrix profile value (distance) for each dimension of the
    multidimensional matrix profile that a multidimenisonal candidate motif is
    allowed to have. 

    visualize_mdl: boolean, default 'False'
        Visualize MDL results

    Returns
    -------
    motif_matches_distances: numpy.ndarray
        The distances corresponding to a set of subsequence matches for each motif.

    motif_matches_indices: numpy.ndarray
        The indices corresponding to a set of subsequences matches for each motif.

    subspace: numpy.ndarray
        A numpy.ndarray consisting of arrays that contain the `k`-dimensional
        subspace for each motif.
    """

    if max_motifs < 1:  # pragma: no cover
        logger.warning(
            "The maximum number of motifs, `max_motifs`, "
            "must be greater than or equal to 1"
        )
        logger.warning("`max_motifs` has been set to `1`")
        max_motifs = 1

    # Calculate subsequence length
    T = _preprocess(T)  # convert dataframe to array if necessary
    m = T.shape[-1] - P.shape[-1] + 1

    # Calculate exclusion zone
    excl_zone = int(np.ceil(m / config.STUMPY_EXCL_ZONE_DENOM))

    if max_matches is None:
        max_matches = np.inf

    motif_matches_distances = []
    motif_matches_indices = []
    subspace = []

    # Identify the first multidimensional motif as candidate motif and get its
    # nearest neighbor
    candidate_idx = np.argsort(P, axis=1)[:, 0]
    nn_idx = I[np.arange(len(candidate_idx)), candidate_idx]

    # Iterate over each motif
    while len(motif_matches_distances) < max_motifs:

        # Choose dimension k using MDL
        mdl, subspaces = stumpy.mdl(T, m, candidate_idx, nn_idx)
        k = np.argmin(mdl)

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
            cutoff_var = cutoff[k]

        if visualize_mdl:
            # Visualize MDL results:
            plt.plot(np.arange(len(mdl)), mdl, linewidth="4")
            plt.title("MDL results", fontsize="30")
            plt.xlabel("k (zero-based)", fontsize="20")
            plt.ylabel("Bit Size", fontsize="20")
            plt.show()

        # Get k-dimensional motif
        motif_idx = candidate_idx[k]
        motif_value = P[k, motif_idx]
        if motif_value > cutoff_var or not np.isfinite(motif_value):
            break

        # Compute subspace and find related dimensions
        subspace.append(subspaces[k])
        sub_dims = T[subspaces[k]]

        # Stop iteration if max_distance is a constant and the k-dim.
        # matrix profile value is larger than the maximum distance.
        if isinstance(max_distance, float) and motif_value > max_distance:
            break

        # Get multidimensional Distance Profile to find the max_matches
        # nearest neighbors
        T, mean_T, sigma_T = core.preprocess(sub_dims, m)

        Q = sub_dims[:, motif_idx : motif_idx + m]
        query_matches = Mmatch(
            Q=Q,
            T_sub=sub_dims,
            mean_T=mean_T,
            sigma_T=sigma_T,
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
        np.array(subspace, dtype=object),
    )


def Mmatch(
    Q: np.ndarray,
    T_sub: np.ndarray,
    mean_T: np.ndarray = None,
    sigma_T: np.ndarray = None,
    max_matches: int = None,
    max_distance: float = None,
    atol: float = 1e-8,
):
    """
    Discover the 'max_matches' nearest neighbors of a multidimensional query Q in a time
    series T

    Parameters
    ----------
    Q: numpy.ndarray
        The multidimensional query sequence

    T_sub: numpy.ndarray
        The relevant dimensions of the multidimensional time series (the dimensions
        where the query Q is represented).
        If Q is a k-dimensional motif, then T_sub corresponds to the k subspace
        dimensions in which the motif is represented.

    mean_T: numpy.ndarray, default None
        Sliding mean of the time series T

    sigma_T: numpy.ndarray, default None
        Sliding standard deviation of time series T

    max_matches: int, default None
        The maximum amount of similar matches (nearest neighbors) of a motif
        representative to be returned

    max_distance: flaot, default None
        Maximal distance that is allowed between a query subsequence (a
        candidate motif) and all subsequences in T to be considered as a match.
        If None, this defaults to
        `np.nanmax([np.nanmean(D) - 2 * np.nanstd(D), np.nanmin(D)])`
        (i.e. at least the closest match will be returned).

    atol : float, default 1e-8
        The absolute tolerance parameter. This value will be added to `max_distance`
        when comparing distances between subsequences.

    Returns
    -------
    matches: numpy.ndarray
        The first column consists of sorted distances (lowest to highest) of
        subsequences of `T_sub`.
        The second column consists of the corresponding indices in `T`.
    """
    Q = _preprocess(Q)  # convert dataframe to array if necessary
    T = _preprocess(T_sub)

    d, n = T.shape
    m = Q.shape[1]

    # Calculate exclusion zone
    excl_zone = int(np.ceil(m / config.STUMPY_EXCL_ZONE_DENOM))
    if max_matches is None:
        max_matches = np.inf

    if np.any(np.isnan(Q)) or np.any(np.isinf(Q)):
        raise ValueError("Q contains illegal values (NaN or inf)")

    if max_distance is None:  # pragma: no cover

        def max_distance(D):
            D_copy = D.copy().astype(np.float64)
            D_copy[np.isinf(D_copy)] = np.nan
            return np.nanmax(
                [np.nanmean(D_copy) - 2.0 * np.nanstd(D_copy), np.nanmin(D_copy)]
            )

    # Compute sliding mean and standard deviation
    if mean_T is None or sigma_T is None:
        T, mean_T, sigma_T = core.preprocess(T, m)

    # Compute multidimensional Distance Profile (the k subspace dimensions are used
    # only) and its mean
    D = np.empty((d, n - m + 1))
    for i in range(d):
        # Compute the 1D Distance Profile of each dimension
        D[i, :] = core.mass(Q[i], T[i], mean_T[i], sigma_T[i])
    D = np.mean(D, axis=0)
    if not isinstance(max_distance, float):
        max_distance = max_distance(D)

    matches = []

    nearest_neighbor_idx = np.argmin(D)
    while (
        len(matches) < max_matches
        and D[nearest_neighbor_idx] <= atol + max_distance
        and np.isfinite(D[nearest_neighbor_idx])
    ):
        matches.append([D[nearest_neighbor_idx], nearest_neighbor_idx])
        core.apply_exclusion_zone(D, nearest_neighbor_idx, excl_zone, np.inf)
        # Find the next nerarest neighbor index after setting the exclusion zone
        nearest_neighbor_idx = np.argmin(D)

    # return matches as array
    return np.array(matches, dtype=object)
