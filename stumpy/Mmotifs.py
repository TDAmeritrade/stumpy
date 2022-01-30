import numpy as np
import logging, stumpy
from . import core, config
from stumpy.mstump import _apply_include, _multi_mass, _multi_distance_profile
logger = logging.getLogger(__name__)


def Mmotifs(T: np.ndarray, P: np.ndarray, I: np.ndarray, max_matches: int = 10, max_motifs: int = 1):
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
        The maximum amount of similar matches (nearest neighbors) of a motif representative to be returned

    max_motifs: int, default 1
        The maximum number of motifs to return

    Returns
    -------
    motif_neighbors_distances: numpy.ndarray
        The distances corresponding to a set of subsequence matches for each motif.

    motif_neighbors_indices: numpy.ndarray
        The indices corresponding to a set of subsequences matches for each motif.

    subspace: list
        A list of numpy.ndarrays that contain the `k`th-dimensional subspaces
    """

    if max_motifs < 1:  # pragma: no cover
        logger.warning(
            "The maximum number of motifs, `max_motifs`, "
            "must be greater than or equal to 1"
        )
        logger.warning("`max_motifs` has been set to `1`")
        max_motifs = 1

    # Calculate subsequence length
    m = T.shape[0] - P.shape[1] + 1   # CHECK CASES
    
    # Calculate exclusion zone
    excl_zone = int(
        np.ceil(m / config.STUMPY_EXCL_ZONE_DENOM)
    )

    if max_matches is None:
        max_matches = np.inf

    # Identify the first multidimensional motif as candidate motif and get its nearest neighbor
    candidate_idx = np.argsort(P, axis=1)[:, 0]
    nn_idx = I[np.arange(len(candidate_idx)), candidate_idx]

    # Iterate over each motif
    for motif_number in range(max_motifs):

        # Choose dimension k using MDL and compute subspace
        mdl, subspace = stumpy.mdl(T, m, candidate_idx, nn_idx)
        k = np.argmin(mdl)
        subspace = subspace[k]
        subspace = [T.columns.values[s] for s in subspace]
        sub_dims = T[subspace].copy()
    

        # Get k-dimensional motif
        motif_idx = candidate_idx[k]
        #nearest_neighbor_idx = nn_idx[k]

        # Get multidimensional Distance Profile
        T, mean_T, sigma_T = core.preprocess(sub_dims, m)

        D = _multi_distance_profile(
            motif_idx, T, T, m, excl_zone, mean_T, sigma_T, mean_T, sigma_T
        )

        # Find the max_matches nearest neighbors 
        motif_neighbors_distances = []
        motif_neighbors_distances.append(0)
        motif_neighbors_indices = []
        motif_neighbors_indices.append(motif_idx)
        nn_idx = np.argmin(D[k, :])
        while len(motif_neighbors_distances) < max_matches:
            motif_neighbors_distances.append(D[k, :][nn_idx])
            motif_neighbors_indices.append(nn_idx)
            core.apply_exclusion_zone(D[k, :], nn_idx, excl_zone, np.inf)
            # Find the next nerarest neighbor index after setting the exclusion zone
            nn_idx = np.argmin(D[k, :])

        # Set exclusion zone and find new candidate_idx for the next motif
        core.apply_exclusion_zone(P, motif_idx, excl_zone, np.inf)
        candidate_idx = np.argsort(P, axis=1)[:, 0]


    return motif_neighbors_distances, motif_neighbors_indices, subspace



