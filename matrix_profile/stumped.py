import numpy as np
from . import core, _stump, _get_first_stump_profile, _get_QT
import logging

logger = logging.getLogger(__name__)

def stumped(dask_client, T_A, m, T_B=None, ignore_trivial=False, disclaimer=True):
    """
    DOI: 10.1109/ICDM.2016.0085
    See Table II

    This is a Dask distributed implementation of stump that scales
    across multiple servers and is a convenience wrapper around the 
    parallelized `stump._stump` function

    Timeseries, T_B, will be annotated with the distance location
    (or index) of all its subsequences in another times series, T_A.

    Return: For every subsequence, Q, in T_B, you will get a distance
    and index for the closest subsequence in T_A. Thus, the array
    returned will have length T_B.shape[0]-m+1. Additionally, the 
    left and right matrix profiles are also returned.

    Note: Unlike in the Table II where T_A.shape is expected to be equal 
    to T_B.shape, this implementation is generalized so that the shapes of 
    T_A and T_B can be different. In the case where T_A.shape == T_B.shape, 
    then our algorithm reduces down to the same algorithm found in Table II. 

    Additionally, unlike STAMP where the exclusion zone is m/2, the default 
    exclusion zone for STOMP is m/4 (See Definition 3 and Figure 3).

    For self-joins, set `ignore_trivial = True` in order to avoid the 
    trivial match. 

    Note that left and right matrix profiles are only available for self-joins.
    """

    if disclaimer:
        logger.warning("Stumped is an experimental implementation that is "
                       "still under development and may change in the future. "
                       "Use at your own risk.")

    core.check_dtype(T_A)
    if T_B is None:
        T_B = T_A
    core.check_dtype(T_B)

    if ignore_trivial == False and core.are_arrays_equal(T_A, T_B):  # pragma: no cover
        logger.warning("Arrays T_A, T_B are equal, which implies a self-join.")
        logger.warning("Try setting `ignore_trivial = True`.")
    
    n = T_B.shape[0]
    k = T_A.shape[0]-m+1
    l = n-m+1
    zone = int(np.ceil(m/4))  # See Definition 3 and Figure 3

    M_T, Σ_T = core.compute_mean_std(T_A, m)
    μ_Q, σ_Q = core.compute_mean_std(T_B, m)

    out = np.empty((l, 4), dtype=object)
    profile = np.empty((l,), dtype='float64')
    indices = np.empty((l, 3), dtype='int64')

    nworkers = len(dask_client.ncores())

    # Scatter data to Dask cluster
    T_A_future = dask_client.scatter(T_A, broadcast=True)
    T_B_future = dask_client.scatter(T_B, broadcast=True)
    M_T_future = dask_client.scatter(M_T, broadcast=True)
    Σ_T_future = dask_client.scatter(Σ_T, broadcast=True)
    μ_Q_future = dask_client.scatter(μ_Q, broadcast=True)
    σ_Q_future = dask_client.scatter(σ_Q, broadcast=True)
    
    step = 1+l//nworkers
    QT_futures = []
    QT_first_futures = []
    for start in range(0, l, step):
        stop = min(l, start + step)

        profile[start], indices[start, :] = \
            _get_first_stump_profile(start, T_A, T_B, m, zone, M_T, 
                                     Σ_T, μ_Q, σ_Q, ignore_trivial)

        QT, QT_first = _get_QT(start, T_A, T_B, m)

        QT_future = dask_client.scatter(QT, broadcast=True)
        QT_first_future = dask_client.scatter(QT_first, broadcast=True)
        
        QT_futures.append(QT_future)
        QT_first_futures.append(QT_first_future)
        
    futures = []        
    for i, start in enumerate(range(0, l, step)):
        stop = min(l, start + step)
                        
        futures.append(
            dask_client.submit(_stump, T_A_future, T_B_future, m, stop, 
                               zone, M_T_future, Σ_T_future, QT_futures[i], 
                               QT_first_futures[i], μ_Q_future, σ_Q_future,
                               k, ignore_trivial, start+1
                              )
                      )
        
    results = dask_client.gather(futures)
    for i, start in enumerate(range(0, l, step)):
        stop = min(l, start + step)
        profile[start+1:stop], indices[start+1:stop, :] = results[i]

    out[:, 0] = profile
    out[:, 1:4] = indices
    
    threshold = 10e-6
    if core.are_distances_too_small(out[:, 0], threshold=threshold):  # pragma: no cover
        logger.warning(f"A large number of values are smaller than {threshold}.")
        logger.warning("For a self-join, try setting `ignore_trivial = True`.")
        
    return out
