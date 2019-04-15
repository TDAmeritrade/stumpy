import numpy as np
from . import core
from . import _mstump, _get_first_mstump_profile, _get_multi_QT, multi_compute_mean_std
import logging

logger = logging.getLogger(__name__)

def mstumped(dask_client, T, m):
    """
    This is highly distributed implementation around the Numba JIT-compiled 
    parallelized `_mstump` function which computes the matrix profile according 
    to STOMP.

    Parameters
    ----------
    dask_client : client
        A Dask Distributed client that is connected to a Dask scheduler and 
        Dask workers. Setting up a Dask distributed cluster is beyond the 
        scope of this library. Please refer to the Dask Distributed 
        documentation.

    T : ndarray
        The time series or sequence for which to compute the matrix profile

    m : int
        Window size

    Returns
    -------
    out : ndarray
        The first column consists of the matrix profile, the second column 
        consists of the matrix profile indices, the third column consists of 
        the left matrix profile indices, and the fourth column consists of 
        the right matrix profile indices.

    Notes
    -----

    DOI: 10.1109/ICDM.2016.0085
    See Table II

    This is a Dask distributed implementation of stump that scales
    across multiple servers and is a convenience wrapper around the 
    parallelized `mstump._mstump` function

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

    hosts = list(dask_client.ncores().keys())
    nworkers = len(hosts)

    core.check_dtype(T)
    
    d = T.shape[0]
    n = T.shape[1]
    k = n-m+1
    excl_zone = int(np.ceil(m/4))  # See Definition 3 and Figure 3

    M_T, Σ_T = multi_compute_mean_std(T, m)
    μ_Q, σ_Q = multi_compute_mean_std(T, m)

    P = np.empty((nworkers, d, k), dtype='float64')
    D = np.zeros((nworkers, d, k), dtype='float64')
    D_prime = np.zeros((nworkers, k), dtype='float64')
    I = np.ones((nworkers, d, k), dtype='int64') * -1

    # Scatter data to Dask cluster
    T_future = dask_client.scatter(T, broadcast=True)
    M_T_future = dask_client.scatter(M_T, broadcast=True)
    Σ_T_future = dask_client.scatter(Σ_T, broadcast=True)
    μ_Q_future = dask_client.scatter(μ_Q, broadcast=True)
    σ_Q_future = dask_client.scatter(σ_Q, broadcast=True)
    
    step = 1+k//nworkers
    QT_futures = []
    QT_first_futures = []
    P_futures = []
    I_futures = []
    D_futures = []
    D_prime_futures = []

    for i, start in enumerate(range(0, k, step)):
        P[i], I[i] = _get_first_mstump_profile(start, T, m, excl_zone, M_T, Σ_T)

        P_future = dask_client.scatter(P[i], workers=[hosts[i]])
        I_future = dask_client.scatter(I[i], workers=[hosts[i]])
        D_future = dask_client.scatter(D[i], workers=[hosts[i]])
        D_prime_future = dask_client.scatter(D_prime[i], workers=[hosts[i]])

        P_futures.append(P_future)
        I_futures.append(I_future)
        D_futures.append(D_future)
        D_prime_futures.append(D_prime_future)

        QT, QT_first = _get_multi_QT(start, T, m)

        QT_future = dask_client.scatter(QT, workers=[hosts[i]])
        QT_first_future = dask_client.scatter(QT_first, workers=[hosts[i]])
        
        QT_futures.append(QT_future)
        QT_first_futures.append(QT_first_future)
        
    futures = []
    for i, start in enumerate(range(0, k, step)):
        stop = min(k, start + step)

        futures.append(
            dask_client.submit(_mstump, T_future, m, P_futures[i], I_futures[i], 
                               D_futures[i], D_prime_futures[i], stop, excl_zone, 
                               M_T_future, Σ_T_future, QT_futures[i], 
                               QT_first_futures[i], μ_Q_future, σ_Q_future, k, 
                               start+1
                              )
                       )

    results = dask_client.gather(futures)
    for i, start in enumerate(range(0, k, step)):
        P[i], I[i] = results[i]
        col_mask = P[0] > P[i]
        P[0, col_mask] = P[i, col_mask]
        I[0, col_mask] = I[i, col_mask]

    return P[0], I[0]