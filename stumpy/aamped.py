# STUMPY
# Copyright 2019 TD Ameritrade. Released under the terms of the 3-Clause BSD license.
# STUMPY is a trademark of TD Ameritrade IP Company, Inc. All rights reserved.

# import inspect
import numpy as np

from . import config, core
from .aamp import _aamp
from .mparray import mparray


def _dask_aamped(
    dask_client,
    T_A,
    T_B,
    m,
    T_A_subseq_isfinite,
    T_B_subseq_isfinite,
    p,
    diags,
    ignore_trivial,
    k,
):
    """
    Compute the non-normalized (i.e., without z-normalization) matrix profile with a
    `dask` cluster

    This is a highly distributed implementation around the Numba JIT-compiled
    parallelized `_aamp` function which computes the non-normalized matrix profile
    according to AAMP.

    Parameters
    ----------
    dask_client : client
        A `dask` client. Setting up a cluster is beyond the scope of this library.
        Please refer to the `dask` documentation.

    T_A : numpy.ndarray
        The time series or sequence for which to compute the matrix profile

    T_B : numpy.ndarray
        The time series or sequence that will be used to annotate T_A. For every
        subsequence in T_A, its nearest neighbor in T_B will be recorded.

    m : int
        Window size

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

    diags : numpy.ndarray
        The diagonal indices

    ignore_trivial : bool, default True
        Set to `True` if this is a self-join. Otherwise, for AB-join, set this
        to `False`. Default is `True`.

    k : int, default 1
        The number of top `k` smallest distances used to construct the matrix profile.
        Note that this will increase the total computational time and memory usage
        when k > 1. If you have access to a GPU device, then you may be able to
        leverage `gpu_stump` for better performance and scalability.

    Returns
    -------
    out : numpy.ndarray
        When k = 1 (default), the first column consists of the matrix profile,
        the second column consists of the matrix profile indices, the third column
        consists of the left matrix profile indices, and the fourth column consists
        of the right matrix profile indices. However, when k > 1, the output array
        will contain exactly 2 * k + 2 columns. The first k columns (i.e., out[:, :k])
        consists of the top-k matrix profile, the next set of k columns
        (i.e., out[:, k:2k]) consists of the corresponding top-k matrix profile
        indices, and the last two columns (i.e., out[:, 2k] and out[:, 2k+1] or,
        equivalently, out[:, -2] and out[:, -1]) correspond to the top-1 left
        matrix profile indices and the top-1 right matrix profile indices, respectively.
    """
    n_A = T_A.shape[0]
    n_B = T_B.shape[0]
    l = n_A - m + 1

    hosts = list(dask_client.ncores().keys())
    nworkers = len(hosts)

    ndist_counts = core._count_diagonal_ndist(diags, m, n_A, n_B)
    diags_ranges = core._get_array_ranges(ndist_counts, nworkers, False)
    diags_ranges += diags[0]

    # Scatter data to Dask cluster
    T_A_future = dask_client.scatter(T_A, broadcast=True, hash=False)
    T_B_future = dask_client.scatter(T_B, broadcast=True, hash=False)
    T_A_subseq_isfinite_future = dask_client.scatter(
        T_A_subseq_isfinite, broadcast=True, hash=False
    )
    T_B_subseq_isfinite_future = dask_client.scatter(
        T_B_subseq_isfinite, broadcast=True, hash=False
    )

    diags_futures = []
    for i, host in enumerate(hosts):
        diags_future = dask_client.scatter(
            np.arange(diags_ranges[i, 0], diags_ranges[i, 1], dtype=np.int64),
            workers=[host],
            hash=False,
        )
        diags_futures.append(diags_future)

    futures = []
    for i in range(len(hosts)):
        futures.append(
            dask_client.submit(
                _aamp,
                T_A_future,
                T_B_future,
                m,
                T_A_subseq_isfinite_future,
                T_B_subseq_isfinite_future,
                p,
                diags_futures[i],
                ignore_trivial,
                k,
            )
        )

    results = dask_client.gather(futures)
    profile, profile_L, profile_R, indices, indices_L, indices_R = results[0]
    for i in range(1, len(hosts)):
        P, PL, PR, I, IL, IR = results[i]
        # Update top-k matrix profile and matrix profile indices
        core._merge_topk_PI(profile, P, indices, I)

        # Update top-1 left matrix profile and matrix profile index
        mask = PL < profile_L
        profile_L[mask] = PL[mask]
        indices_L[mask] = IL[mask]

        # Update top-1 right matrix profile and matrix profile index
        mask = PR < profile_R
        profile_R[mask] = PR[mask]
        indices_R[mask] = IR[mask]

    out = np.empty((l, 2 * k + 2), dtype=object)
    out[:, :k] = profile
    out[:, k : 2 * k + 2] = np.column_stack((indices, indices_L, indices_R))

    return out


def _ray_aamped(
    ray_client,
    T_A,
    T_B,
    m,
    T_A_subseq_isfinite,
    T_B_subseq_isfinite,
    p,
    diags,
    ignore_trivial,
    k,
):
    """
    Compute the non-normalized (i.e., without z-normalization) matrix profile with a
    `ray` cluster

    This is a highly distributed implementation around the Numba JIT-compiled
    parallelized `_aamp` function which computes the non-normalized matrix profile
    according to AAMP.

    Parameters
    ----------
    ray_client : client
        A `ray` client. Setting up a cluster is beyond the scope of this library.
        Please refer to the `ray` documentation.

    T_A : numpy.ndarray
        The time series or sequence for which to compute the matrix profile

    T_B : numpy.ndarray
        The time series or sequence that will be used to annotate T_A. For every
        subsequence in T_A, its nearest neighbor in T_B will be recorded.

    m : int
        Window size

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

    diags : numpy.ndarray
        The diagonal indices

    ignore_trivial : bool, default True
        Set to `True` if this is a self-join. Otherwise, for AB-join, set this
        to `False`. Default is `True`.

    k : int, default 1
        The number of top `k` smallest distances used to construct the matrix profile.
        Note that this will increase the total computational time and memory usage
        when k > 1. If you have access to a GPU device, then you may be able to
        leverage `gpu_stump` for better performance and scalability.

    Returns
    -------
    out : numpy.ndarray
        When k = 1 (default), the first column consists of the matrix profile,
        the second column consists of the matrix profile indices, the third column
        consists of the left matrix profile indices, and the fourth column consists
        of the right matrix profile indices. However, when k > 1, the output array
        will contain exactly 2 * k + 2 columns. The first k columns (i.e., out[:, :k])
        consists of the top-k matrix profile, the next set of k columns
        (i.e., out[:, k:2k]) consists of the corresponding top-k matrix profile
        indices, and the last two columns (i.e., out[:, 2k] and out[:, 2k+1] or,
        equivalently, out[:, -2] and out[:, -1]) correspond to the top-1 left
        matrix profile indices and the top-1 right matrix profile indices, respectively.
    """
    core.check_ray(ray_client)

    n_A = T_A.shape[0]
    n_B = T_B.shape[0]
    l = n_A - m + 1

    nworkers = core.get_ray_nworkers(ray_client)

    ndist_counts = core._count_diagonal_ndist(diags, m, n_A, n_B)
    diags_ranges = core._get_array_ranges(ndist_counts, nworkers, False)
    diags_ranges += diags[0]

    # Scatter data to Ray cluster
    T_A_ref = ray_client.put(T_A)
    T_B_ref = ray_client.put(T_B)
    T_A_subseq_isfinite_ref = ray_client.put(T_A_subseq_isfinite)
    T_B_subseq_isfinite_ref = ray_client.put(T_B_subseq_isfinite)

    diags_refs = []
    for i in range(nworkers):
        diags_ref = ray_client.put(
            np.arange(diags_ranges[i, 0], diags_ranges[i, 1], dtype=np.int64)
        )
        diags_refs.append(diags_ref)

    ray_aamp_func = ray_client.remote(core.deco_ray_tor(_aamp))

    refs = []
    for i in range(nworkers):
        refs.append(
            ray_aamp_func.remote(
                T_A_ref,
                T_B_ref,
                m,
                T_A_subseq_isfinite_ref,
                T_B_subseq_isfinite_ref,
                p,
                diags_refs[i],
                ignore_trivial,
                k,
            )
        )

    results = ray_client.get(refs)
    # Must make a mutable copy from Ray's object store (ndarrays are immutable)
    profile, profile_L, profile_R, indices, indices_L, indices_R = [
        arr.copy() for arr in results[0]
    ]

    for i in range(1, nworkers):
        P, PL, PR, I, IL, IR = results[i]  # Read-only variables
        # Update top-k matrix profile and matrix profile indices
        core._merge_topk_PI(profile, P, indices, I)

        # Update top-1 left matrix profile and matrix profile index
        mask = PL < profile_L
        profile_L[mask] = PL[mask]
        indices_L[mask] = IL[mask]

        # Update top-1 right matrix profile and matrix profile index
        mask = PR < profile_R
        profile_R[mask] = PR[mask]
        indices_R[mask] = IR[mask]

    out = np.empty((l, 2 * k + 2), dtype=object)
    out[:, :k] = profile
    out[:, k : 2 * k + 2] = np.column_stack((indices, indices_L, indices_R))

    return out


def aamped(client, T_A, m, T_B=None, ignore_trivial=True, p=2.0, k=1):
    """
    Compute the non-normalized (i.e., without z-normalization) matrix profile
    with a `dask`/`ray` cluster

    This is a highly distributed implementation around the Numba JIT-compiled
    parallelized `_aamp` function which computes the non-normalized matrix profile
    according to AAMP.

    Parameters
    ----------
    client : client
        A `dask`/`ray` client. Setting up a cluster is beyond the scope of this library.
        Please refer to the `dask`/`ray` documentation.

    T_A : numpy.ndarray
        The time series or sequence for which to compute the matrix profile

    m : int
        Window size

    T_B : numpy.ndarray, default None
        The time series or sequence that will be used to annotate T_A. For every
        subsequence in T_A, its nearest neighbor in T_B will be recorded. Default is
        `None` which corresponds to a self-join.

    ignore_trivial : bool, default True
        Set to `True` if this is a self-join. Otherwise, for AB-join, set this
        to `False`. Default is `True`.

    p : float, default 2.0
        The p-norm to apply for computing the Minkowski distance. Minkowski distance is
        typically used with `p` being 1 or 2, which correspond to the Manhattan distance
        and the Euclidean distance, respectively.

    k : int, default 1
        The number of top `k` smallest distances used to construct the matrix profile.
        Note that this will increase the total computational time and memory usage
        when k > 1.

    Returns
    -------
    out : numpy.ndarray
        When k = 1 (default), the first column consists of the matrix profile,
        the second column consists of the matrix profile indices, the third column
        consists of the left matrix profile indices, and the fourth column consists
        of the right matrix profile indices. However, when k > 1, the output array
        will contain exactly 2 * k + 2 columns. The first k columns (i.e., out[:, :k])
        consists of the top-k matrix profile, the next set of k columns
        (i.e., out[:, k:2k]) consists of the corresponding top-k matrix profile
        indices, and the last two columns (i.e., out[:, 2k] and out[:, 2k+1] or,
        equivalently, out[:, -2] and out[:, -1]) correspond to the top-1 left
        matrix profile indices and the top-1 right matrix profile indices, respectively.

        For convenience, the matrix profile (distances) and matrix profile indices can
        also be accessed via their corresponding named array attributes, `.P_` and
        `.I_`,respectively. Similarly, the corresponding left matrix profile indices
        and right matrix profile indices may also be accessed via the `.left_I_` and
        `.right_I_` array attributes.

    Notes
    -----
    `arXiv:1901.05708 \
    <https://arxiv.org/pdf/1901.05708.pdf>`__

    See Algorithm 1

    Note that we have extended this algorithm for AB-joins as well.
    """
    if T_B is None:
        T_B = T_A.copy()
        ignore_trivial = True

    T_A, T_A_subseq_isfinite = core.preprocess_non_normalized(T_A, m)
    T_B, T_B_subseq_isfinite = core.preprocess_non_normalized(T_B, m)

    if T_A.ndim != 1:  # pragma: no cover
        raise ValueError(f"T_A is {T_A.ndim}-dimensional and must be 1-dimensional. ")

    if T_B.ndim != 1:  # pragma: no cover
        raise ValueError(f"T_B is {T_B.ndim}-dimensional and must be 1-dimensional. ")

    n_A = T_A.shape[0]
    n_B = T_B.shape[0]

    ignore_trivial = core.check_ignore_trivial(T_A, T_B, ignore_trivial)
    excl_zone = int(np.ceil(m / config.STUMPY_EXCL_ZONE_DENOM))

    if ignore_trivial:
        core.check_window_size(m, max_size=min(n_A, n_B), n=n_A)
        diags = np.arange(excl_zone + 1, n_A - m + 1, dtype=np.int64)
    else:
        core.check_window_size(m, max_size=min(n_A, n_B))
        diags = np.arange(-(n_A - m + 1) + 1, n_B - m + 1, dtype=np.int64)

    _aamped = core._client_to_func(client)

    out = _aamped(
        client,
        T_A,
        T_B,
        m,
        T_A_subseq_isfinite,
        T_B_subseq_isfinite,
        p,
        diags,
        ignore_trivial,
        k,
    )

    core._check_P(out[:, 0])

    return mparray(out, m, k, config.STUMPY_EXCL_ZONE_DENOM)
