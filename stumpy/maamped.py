# STUMPY
# Copyright 2019 TD Ameritrade. Released under the terms of the 3-Clause BSD license.
# STUMPY is a trademark of TD Ameritrade IP Company, Inc. All rights reserved.

import math

import numpy as np

from . import config, core
from .maamp import _get_first_maamp_profile, _get_multi_p_norm, _maamp
from .mmparray import mparray


def _dask_maamped(
    dask_client,
    T_A,
    T_B,
    m,
    excl_zone,
    T_A_subseq_isfinite,
    T_B_subseq_isfinite,
    p,
    include,
    discords,
):
    """
    Compute the multi-dimensional non-normalized (i.e., without z-normalization) matrix
    profile with a `dask` cluster

    This is a highly distributed implementation around the Numba JIT-compiled
    parallelized `_maamp` function which computes the multi-dimensional matrix
    profile according to STOMP. Note that only self-joins are supported.

    Parameters
    ----------
    dask_client : client
        A `dask` client. Setting up a cluster is beyond the scope of this library.
        Please refer to the `dask` documentation.

    T_A : numpy.ndarray
        The time series or sequence for which to compute the multi-dimensional
        matrix profile. Each row in `T_A` represents data from the same
        dimension while each column in `T_A` represents data from a different
        dimension.

    T_B : numpy.ndarray
        The time series or sequence that will be used to annotate T_A. For every
        subsequence in T_A, its nearest neighbor in T_B will be recorded.

    m : int
        Window size

    excl_zone : int
        The half width for the exclusion zone relative to the current
        sliding window

    T_A_subseq_isfinite : numpy.ndarray
        A boolean array that indicates whether a subsequence in `T_A` contains a
        `np.nan`/`np.inf` value (False)

    T_B_subseq_isfinite : numpy.ndarray
        A boolean array that indicates whether a subsequence in `T_B` contains a
        `np.nan`/`np.inf` value (False)

    p : float
        The p-norm to apply for computing the Minkowski distance. Minkowski distance is
        typically used with `p` being 1 or 2, which correspond to the Manhattan distance
        and the Euclidean distance, respectively.

    include : numpy.ndarray
        A list of (zero-based) indices corresponding to the dimensions in `T` that
        must be included in the constrained multidimensional motif search.
        For more information, see Section IV D in:

        `DOI: 10.1109/ICDM.2017.66 \
        <https://www.cs.ucr.edu/~eamonn/Motif_Discovery_ICDM.pdf>`__

    discords : bool
        When set to `True`, this reverses the distance profile to favor discords rather
        than motifs. Note that indices in `include` are still maintained and respected.

    Returns
    -------
    P : numpy.ndarray
        The multi-dimensional matrix profile. Each row of the array corresponds
        to each matrix profile for a given dimension (i.e., the first row is
        the 1-D matrix profile and the second row is the 2-D matrix profile).

    I : numpy.ndarray
        The multi-dimensional matrix profile index where each row of the array
        corresponds to each matrix profile index for a given dimension.
    """
    d, n = T_B.shape
    l = n - m + 1
    P = np.empty((d, l), dtype=np.float64)
    I = np.empty((d, l), dtype=np.int64)

    hosts = list(dask_client.ncores().keys())
    nworkers = len(hosts)

    step = int(math.ceil(l / nworkers))

    for start in range(0, l, step):
        P[:, start], I[:, start] = _get_first_maamp_profile(
            start,
            T_A,
            T_B,
            m,
            excl_zone,
            T_B_subseq_isfinite,
            p,
            include,
            discords,
        )
    # Scatter data to Dask cluster
    T_A_future = dask_client.scatter(T_A, broadcast=True, hash=False)
    T_A_subseq_isfinite_future = dask_client.scatter(
        T_A_subseq_isfinite, broadcast=True, hash=False
    )
    T_B_subseq_isfinite_future = dask_client.scatter(
        T_B_subseq_isfinite, broadcast=True, hash=False
    )

    p_norm_futures = []
    p_norm_first_futures = []

    for i, start in enumerate(range(0, l, step)):
        p_norm, p_norm_first = _get_multi_p_norm(start, T_A, m)

        p_norm_future = dask_client.scatter(p_norm, workers=[hosts[i]], hash=False)
        p_norm_first_future = dask_client.scatter(
            p_norm_first, workers=[hosts[i]], hash=False
        )

        p_norm_futures.append(p_norm_future)
        p_norm_first_futures.append(p_norm_first_future)

    futures = []
    for i, start in enumerate(range(0, l, step)):
        stop = min(l, start + step)

        futures.append(
            dask_client.submit(
                _maamp,
                T_A_future,
                m,
                stop,
                excl_zone,
                T_A_subseq_isfinite_future,
                T_B_subseq_isfinite_future,
                p,
                p_norm_futures[i],
                p_norm_first_futures[i],
                l,
                start + 1,
                include,
                discords,
            )
        )

    results = dask_client.gather(futures)
    for i, start in enumerate(range(0, l, step)):
        stop = min(l, start + step)
        P[:, start + 1 : stop], I[:, start + 1 : stop] = results[i]

    return P, I


def _ray_maamped(
    ray_client,
    T_A,
    T_B,
    m,
    excl_zone,
    T_A_subseq_isfinite,
    T_B_subseq_isfinite,
    p,
    include,
    discords,
):
    """
    Compute the multi-dimensional non-normalized (i.e., without z-normalization) matrix
    profile with a `ray` cluster

    This is a highly distributed implementation around the Numba JIT-compiled
    parallelized `_maamp` function which computes the multi-dimensional matrix
    profile according to STOMP. Note that only self-joins are supported.

    Parameters
    ----------
    ray_client : client
        A `ray` client. Setting up a cluster is beyond the scope of this
        library. Please refer to the `ray` documentation.

    T_A : numpy.ndarray
        The time series or sequence for which to compute the multi-dimensional
        matrix profile. Each row in `T_A` represents data from the same
        dimension while each column in `T_A` represents data from a different
        dimension.

    T_B : numpy.ndarray
        The time series or sequence that will be used to annotate T_A. For every
        subsequence in T_A, its nearest neighbor in T_B will be recorded.

    m : int
        Window size

    excl_zone : int
        The half width for the exclusion zone relative to the current
        sliding window

    T_A_subseq_isfinite : numpy.ndarray
        A boolean array that indicates whether a subsequence in `T_A` contains a
        `np.nan`/`np.inf` value (False)

    T_B_subseq_isfinite : numpy.ndarray
        A boolean array that indicates whether a subsequence in `T_B` contains a
        `np.nan`/`np.inf` value (False)

    p : float
        The p-norm to apply for computing the Minkowski distance. Minkowski distance is
        typically used with `p` being 1 or 2, which correspond to the Manhattan distance
        and the Euclidean distance, respectively.

    include : numpy.ndarray
        A list of (zero-based) indices corresponding to the dimensions in `T` that
        must be included in the constrained multidimensional motif search.
        For more information, see Section IV D in:

        `DOI: 10.1109/ICDM.2017.66 \
        <https://www.cs.ucr.edu/~eamonn/Motif_Discovery_ICDM.pdf>`__

    discords : bool
        When set to `True`, this reverses the distance profile to favor discords rather
        than motifs. Note that indices in `include` are still maintained and respected.

    Returns
    -------
    P : numpy.ndarray
        The multi-dimensional matrix profile. Each row of the array corresponds
        to each matrix profile for a given dimension (i.e., the first row is
        the 1-D matrix profile and the second row is the 2-D matrix profile).

    I : numpy.ndarray
        The multi-dimensional matrix profile index where each row of the array
        corresponds to each matrix profile index for a given dimension.
    """
    core.check_ray(ray_client)

    d, n = T_B.shape
    l = n - m + 1
    P = np.empty((d, l), dtype=np.float64)
    I = np.empty((d, l), dtype=np.int64)

    nworkers = core.get_ray_nworkers(ray_client)

    step = int(math.ceil(l / nworkers))

    for start in range(0, l, step):
        P[:, start], I[:, start] = _get_first_maamp_profile(
            start,
            T_A,
            T_B,
            m,
            excl_zone,
            T_B_subseq_isfinite,
            p,
            include,
            discords,
        )

    # Put data into Ray object storage
    T_A_ref = ray_client.put(T_A)
    T_A_subseq_isfinite_ref = ray_client.put(T_A_subseq_isfinite)
    T_B_subseq_isfinite_ref = ray_client.put(T_B_subseq_isfinite)

    p_norm_refs = []
    p_norm_first_refs = []

    for start in range(0, l, step):
        p_norm, p_norm_first = _get_multi_p_norm(start, T_A, m)

        p_norm_ref = ray_client.put(p_norm)
        p_norm_first_ref = ray_client.put(p_norm_first)

        p_norm_refs.append(p_norm_ref)
        p_norm_first_refs.append(p_norm_first_ref)

    ray_maamp_func = ray_client.remote(core.deco_ray_tor(_maamp))

    refs = []
    for i, start in enumerate(range(0, l, step)):
        stop = min(l, start + step)

        refs.append(
            ray_maamp_func.remote(
                T_A_ref,
                m,
                stop,
                excl_zone,
                T_A_subseq_isfinite_ref,
                T_B_subseq_isfinite_ref,
                p,
                p_norm_refs[i],
                p_norm_first_refs[i],
                l,
                start + 1,
                include,
                discords,
            )
        )

    results = ray_client.get(refs)
    for i, start in enumerate(range(0, l, step)):
        stop = min(l, start + step)
        P[:, start + 1 : stop], I[:, start + 1 : stop] = results[i]

    return P, I


def maamped(client, T, m, include=None, discords=False, p=2.0):
    """
    Compute the multi-dimensional non-normalized (i.e., without z-normalization) matrix
    profile with a `dask`/`ray` cluster

    This is a highly distributed implementation around the Numba JIT-compiled
    parallelized `_maamp` function which computes the multi-dimensional matrix
    profile according to STOMP. Note that only self-joins are supported.

    Parameters
    ----------
    client : client
        A `dask`/`ray` client. Setting up a cluster is beyond the scope of this
        library. Please refer to the `dask`/`ray` documentation.

    T : numpy.ndarray
        The time series or sequence for which to compute the multi-dimensional
        matrix profile. Each row in `T` represents data from the same
        dimension while each column in `T` represents data from a different
        dimension.

    m : int
        Window size

    include : list, numpy.ndarray, default None
        A list of (zero-based) indices corresponding to the dimensions in `T` that
        must be included in the constrained multidimensional motif search.
        For more information, see Section IV D in:

        `DOI: 10.1109/ICDM.2017.66 \
        <https://www.cs.ucr.edu/~eamonn/Motif_Discovery_ICDM.pdf>`__

    discords : bool, default False
        When set to `True`, this reverses the distance matrix which results in a
        multi-dimensional matrix profile that favors larger matrix profile values
        (i.e., discords) rather than smaller values (i.e., motifs). Note that indices
        in `include` are still maintained and respected.

    p : float, default 2.0
        The p-norm to apply for computing the Minkowski distance. Minkowski distance is
        typically used with `p` being 1 or 2, which correspond to the Manhattan distance
        and the Euclidean distance, respectively.

    Returns
    -------
    P : numpy.ndarray
        The multi-dimensional matrix profile. Each row of the array corresponds
        to each matrix profile for a given dimension (i.e., the first row is
        the 1-D matrix profile and the second row is the 2-D matrix profile).

    I : numpy.ndarray
        The multi-dimensional matrix profile index where each row of the array
        corresponds to each matrix profile index for a given dimension.

    Notes
    -----
    `DOI: 10.1109/ICDM.2017.66 \
    <https://www.cs.ucr.edu/~eamonn/Motif_Discovery_ICDM.pdf>`__

    See mSTAMP Algorithm
    """
    T_A = T
    T_B = T_A

    T_A, T_A_subseq_isfinite = core.preprocess_non_normalized(T_A, m)
    T_B, T_B_subseq_isfinite = core.preprocess_non_normalized(T_B, m)

    if T_A.ndim <= 1:  # pragma: no cover
        err = f"T is {T_A.ndim}-dimensional and must be at least 2-dimensional"
        raise ValueError(f"{err}")

    core.check_window_size(m, max_size=min(T_A.shape[1], T_B.shape[1]), n=T_A.shape[1])

    if include is not None:
        include = core._preprocess_include(include)

    excl_zone = int(
        np.ceil(m / config.STUMPY_EXCL_ZONE_DENOM)
    )  # See Definition 3 and Figure 3

    _maamped = core._client_to_func(client)

    P, I = _maamped(
        client,
        T_A,
        T_B,
        m,
        excl_zone,
        T_A_subseq_isfinite,
        T_B_subseq_isfinite,
        p,
        include,
        discords,
    )

    return mparray(P_=P, I_=I)
