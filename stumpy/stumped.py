# STUMPY
# Copyright 2019 TD Ameritrade. Released under the terms of the 3-Clause BSD license.
# STUMPY is a trademark of TD Ameritrade IP Company, Inc. All rights reserved.

import numpy as np

from . import config, core
from .aamped import aamped
from .mparray import mparray
from .stump import _stump


def _dask_stumped(
    dask_client,
    T_A,
    T_B,
    m,
    M_T,
    μ_Q,
    Σ_T_inverse,
    σ_Q_inverse,
    M_T_m_1,
    μ_Q_m_1,
    T_A_subseq_isfinite,
    T_B_subseq_isfinite,
    T_A_subseq_isconstant,
    T_B_subseq_isconstant,
    diags,
    ignore_trivial,
    k,
):
    """
    Compute the z-normalized (top-k) matrix profile with a `dask` cluster

    This is a highly distributed implementation around the Numba JIT-compiled
    parallelized `_stump` function which computes the (top-k) matrix profile according
    to STOMPopt with Pearson correlations.

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

    M_T : numpy.ndarray
        Sliding mean of time series, `T`

    μ_Q : numpy.ndarray
        Mean of the query sequence, `Q`, relative to the current sliding window

    Σ_T_inverse : numpy.ndarray
        Inverse sliding standard deviation of time series, `T`

    σ_Q_inverse : numpy.ndarray
        Inverse standard deviation of the query sequence, `Q`, relative to the current

    M_T_m_1 : numpy.ndarray
        Sliding mean of time series, `T`, using a window size of `m-1`

    μ_Q_m_1 : numpy.ndarray
        Mean of the query sequence, `Q`, relative to the current sliding window and
        using a window size of `m-1`

    T_A_subseq_isfinite : numpy.ndarray
        A boolean array that indicates whether a subsequence in `T_A` contains a
        `np.nan`/`np.inf` value (False)

    T_B_subseq_isfinite : numpy.ndarray
        A boolean array that indicates whether a subsequence in `T_B` contains a
        `np.nan`/`np.inf` value (False)

    T_A_subseq_isconstant : numpy.ndarray
        A boolean array that indicates whether a subsequence in `T_A` is constant (True)

    T_B_subseq_isconstant : numpy.ndarray
        A boolean array that indicates whether a subsequence in `T_B` is constant (True)

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
    M_T_future = dask_client.scatter(M_T, broadcast=True, hash=False)
    μ_Q_future = dask_client.scatter(μ_Q, broadcast=True, hash=False)
    Σ_T_inverse_future = dask_client.scatter(Σ_T_inverse, broadcast=True, hash=False)
    σ_Q_inverse_future = dask_client.scatter(σ_Q_inverse, broadcast=True, hash=False)
    M_T_m_1_future = dask_client.scatter(M_T_m_1, broadcast=True, hash=False)
    μ_Q_m_1_future = dask_client.scatter(μ_Q_m_1, broadcast=True, hash=False)
    T_A_subseq_isfinite_future = dask_client.scatter(
        T_A_subseq_isfinite, broadcast=True, hash=False
    )
    T_B_subseq_isfinite_future = dask_client.scatter(
        T_B_subseq_isfinite, broadcast=True, hash=False
    )
    T_A_subseq_isconstant_future = dask_client.scatter(
        T_A_subseq_isconstant, broadcast=True, hash=False
    )
    T_B_subseq_isconstant_future = dask_client.scatter(
        T_B_subseq_isconstant, broadcast=True, hash=False
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
    for i in range(nworkers):
        futures.append(
            dask_client.submit(
                _stump,
                T_A_future,
                T_B_future,
                m,
                μ_Q_future,
                M_T_future,
                σ_Q_inverse_future,
                Σ_T_inverse_future,
                μ_Q_m_1_future,
                M_T_m_1_future,
                T_A_subseq_isfinite_future,
                T_B_subseq_isfinite_future,
                T_A_subseq_isconstant_future,
                T_B_subseq_isconstant_future,
                diags_futures[i],
                ignore_trivial,
                k,
            )
        )

    results = dask_client.gather(futures)
    profile, profile_L, profile_R, indices, indices_L, indices_R = results[0]

    for i in range(1, nworkers):
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
    out[:, k:] = np.column_stack((indices, indices_L, indices_R))

    return out


def _ray_stumped(
    ray_client,
    T_A,
    T_B,
    m,
    M_T,
    μ_Q,
    Σ_T_inverse,
    σ_Q_inverse,
    M_T_m_1,
    μ_Q_m_1,
    T_A_subseq_isfinite,
    T_B_subseq_isfinite,
    T_A_subseq_isconstant,
    T_B_subseq_isconstant,
    diags,
    ignore_trivial,
    k,
):
    """
    Compute the z-normalized (top-k) matrix profile with a `ray` cluster

    This is a highly distributed implementation around the Numba JIT-compiled
    parallelized `_stump` function which computes the (top-k) matrix profile according
    to STOMPopt with Pearson correlations.

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

    M_T : numpy.ndarray
        Sliding mean of time series, `T`

    μ_Q : numpy.ndarray
        Mean of the query sequence, `Q`, relative to the current sliding window

    Σ_T_inverse : numpy.ndarray
        Inverse sliding standard deviation of time series, `T`

    σ_Q_inverse : numpy.ndarray
        Inverse standard deviation of the query sequence, `Q`, relative to the current

    M_T_m_1 : numpy.ndarray
        Sliding mean of time series, `T`, using a window size of `m-1`

    μ_Q_m_1 : numpy.ndarray
        Mean of the query sequence, `Q`, relative to the current sliding window and
        using a window size of `m-1`

    T_A_subseq_isfinite : numpy.ndarray
        A boolean array that indicates whether a subsequence in `T_A` contains a
        `np.nan`/`np.inf` value (False)

    T_B_subseq_isfinite : numpy.ndarray
        A boolean array that indicates whether a subsequence in `T_B` contains a
        `np.nan`/`np.inf` value (False)

    T_A_subseq_isconstant : numpy.ndarray
        A boolean array that indicates whether a subsequence in `T_A` is constant (True)

    T_B_subseq_isconstant : numpy.ndarray
        A boolean array that indicates whether a subsequence in `T_B` is constant (True)

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

    # Put data in the Ray object store
    T_A_ref = ray_client.put(T_A)
    T_B_ref = ray_client.put(T_B)
    M_T_ref = ray_client.put(M_T)
    μ_Q_ref = ray_client.put(μ_Q)
    Σ_T_inverse_ref = ray_client.put(Σ_T_inverse)
    σ_Q_inverse_ref = ray_client.put(σ_Q_inverse)
    M_T_m_1_ref = ray_client.put(M_T_m_1)
    μ_Q_m_1_ref = ray_client.put(μ_Q_m_1)
    T_A_subseq_isfinite_ref = ray_client.put(T_A_subseq_isfinite)
    T_B_subseq_isfinite_ref = ray_client.put(T_B_subseq_isfinite)
    T_A_subseq_isconstant_ref = ray_client.put(T_A_subseq_isconstant)
    T_B_subseq_isconstant_ref = ray_client.put(T_B_subseq_isconstant)

    diags_refs = []
    for i in range(nworkers):
        diags_ref = ray_client.put(
            np.arange(diags_ranges[i, 0], diags_ranges[i, 1], dtype=np.int64),
        )
        diags_refs.append(diags_ref)

    ray_stump_func = ray_client.remote(core.deco_ray_tor(_stump))

    refs = []
    for i in range(nworkers):
        refs.append(
            ray_stump_func.remote(
                T_A_ref,
                T_B_ref,
                m,
                μ_Q_ref,
                M_T_ref,
                σ_Q_inverse_ref,
                Σ_T_inverse_ref,
                μ_Q_m_1_ref,
                M_T_m_1_ref,
                T_A_subseq_isfinite_ref,
                T_B_subseq_isfinite_ref,
                T_A_subseq_isconstant_ref,
                T_B_subseq_isconstant_ref,
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
    out[:, k:] = np.column_stack((indices, indices_L, indices_R))

    return out


@core.non_normalized(aamped)
def stumped(
    client,
    T_A,
    m,
    T_B=None,
    ignore_trivial=True,
    normalize=True,
    p=2.0,
    k=1,
    T_A_subseq_isconstant=None,
    T_B_subseq_isconstant=None,
):
    """
    Compute the z-normalized matrix profile with a ``dask``/``ray`` cluster

    This is a highly distributed implementation around the Numba JIT-compiled
    parallelized ``_stump`` function which computes the (top-k) matrix profile
    according to STOMPopt with Pearson correlations.

    Parameters
    ----------
    client : client
        A ``dask``/``ray`` client. Setting up a cluster is beyond the scope of this
        library. Please refer to the ``dask``/``ray`` documentation.

    T_A : numpy.ndarray
        The time series or sequence for which to compute the matrix profile.

    m : int
        Window size.

    T_B : numpy.ndarray, default None
        The time series or sequence that will be used to annotate ``T_A``. For every
        subsequence in ``T_A``, its nearest neighbor in ``T_B`` will be recorded.
        Default is ``None`` which corresponds to a self-join.

    ignore_trivial : bool, default True
        Set to ``True`` if this is a self-join. Otherwise, for AB-join, set this
        to ``False``.

    normalize : bool, default True
        When set to ``True``, this z-normalizes subsequences prior to computing
        distances. Otherwise, this function gets re-routed to its complementary
        non-normalized equivalent set in the ``@core.non_normalized`` function
        decorator.

    p : float, default 2.0
        The p-norm to apply for computing the Minkowski distance. Minkowski distance is
        typically used with ``p`` being ``1`` or ``2``, which correspond to the
        Manhattan distance and the Euclidean distance, respectively. This parameter is
        ignored when ``normalize == True``.

    k : int, default 1
        The number of top ``k`` smallest distances used to construct the matrix
        profile. Note that this will increase the total computational time and memory
        usage when ``k > 1``. If you have access to a GPU device, then you may be able
        to leverage ``gpu_stump`` for better performance and scalability.

    T_A_subseq_isconstant : numpy.ndarray or function, default None
        A boolean array that indicates whether a subsequence in ``T_A`` is constant
        (``True``). Alternatively, a custom, user-defined function that returns a
        boolean array that indicates whether a subsequence in ``T_A`` is constant
        (``True``). The function must only take two arguments, ``a``, a 1-D array,
        and ``w``, the window size, while additional arguments may be specified
        by currying the user-defined function using ``functools.partial``. Any
        subsequence with at least one ``np.nan``/``np.inf`` will automatically have
        its corresponding value set to ``False`` in this boolean array.

    T_B_subseq_isconstant : numpy.ndarray or function, default None
        A boolean array that indicates whether a subsequence in ``T_B`` is constant
        (``True``). Alternatively, a custom, user-defined function that returns a
        boolean array that indicates whether a subsequence in ``T_B`` is constant
        (``True``). The function must only take two arguments, ``a``, a 1-D array,
        and ``w``, the window size, while additional arguments may be specified
        by currying the user-defined function using ``functools.partial``. Any
        subsequence with at least one ``np.nan``/``np.inf`` will automatically have
        its corresponding value set to ``False`` in this boolean array.

    Returns
    -------
    out : numpy.ndarray
        When ``k = 1`` (default), the first column consists of the matrix profile,
        the second column consists of the matrix profile indices, the third column
        consists of the left matrix profile indices, and the fourth column consists
        of the right matrix profile indices. However, when ``k > 1``, the output array
        will contain exactly ``2 * k + 2`` columns. The first ``k`` columns (i.e.,
        ``out[:, :k]``) consists of the top-k matrix profile, the next set of ``k``
        columns (i.e., ``out[:, k : 2 * k]``) consists of the corresponding top-k matrix
        profile indices, and the last two columns (i.e., ``out[:, 2 * k]`` and
        ``out[:, 2 * k + 1]`` or, equivalently, ``out[:, -2]`` and ``out[:, -1]``)
        correspond to the top-1 left matrix profile indices and the top-1 right matrix
        profile indices, respectively.

        |

        For convenience, the matrix profile (distances) and matrix profile indices can
        also be accessed via their corresponding named array attributes, ``.P_`` and
        ``.I_``,respectively. Similarly, the corresponding left matrix profile indices
        and right matrix profile indices may also be accessed via the ``.left_I_`` and
        ``.right_I_`` array attributes. See examples below.

    See Also
    --------
    stumpy.stump : Compute the z-normalized matrix profile
        cluster
    stumpy.gpu_stump : Compute the z-normalized matrix profile with one or more GPU
        devices
    stumpy.scrump : Compute an approximate z-normalized matrix profile

    Notes
    -----
    `DOI: 10.1007/s10115-017-1138-x \
    <https://www.cs.ucr.edu/~eamonn/ten_quadrillion.pdf>`__

    See Section 4.5

    The above reference outlines a general approach for traversing the distance
    matrix in a diagonal fashion rather than in a row-wise fashion.

    `DOI: 10.1145/3357223.3362721 \
    <https://www.cs.ucr.edu/~eamonn/public/GPU_Matrix_profile_VLDB_30DraftOnly.pdf>`__

    See Section 3.1 and Section 3.3

    The above reference outlines the use of the Pearson correlation via Welford's
    centered sum-of-products along each diagonal of the distance matrix in place of the
    sliding window dot product found in the original STOMP method.

    `DOI: 10.1109/ICDM.2016.0085 \
    <https://www.cs.ucr.edu/~eamonn/STOMP_GPU_final_submission_camera_ready.pdf>`__

    See Table II

    This is a ``dask``/``ray`` implementation of stump that scales
    across multiple servers and is a convenience wrapper around the
    parallelized ``stump._stump`` function

    Timeseries, ``T_A``, will be annotated with the distance location
    (or index) of all its subsequences in another times series, ``T_B``.

    Return: For every subsequence, ``Q``, in ``T_A``, you will get a distance
    and index for the closest subsequence in ``T_B``. Thus, the array
    returned will have length ``T_A.shape[0] - m + 1``. Additionally, the
    left and right matrix profiles are also returned.

    Note: Unlike in the Table II where ``T_A.shape`` is expected to be equal
    to ``T_B.shape``, this implementation is generalized so that the shapes of
    ``T_A`` and ``T_B`` can be different. In the case where ``T_A.shape == T_B.shape``,
    then our algorithm reduces down to the same algorithm found in Table II.

    Additionally, unlike STAMP where the exclusion zone is ``m``/2, the default
    exclusion zone for STOMP is ``m``/4 (See Definition 3 and Figure 3).

    For self-joins, set ``ignore_trivial = True`` in order to avoid the
    trivial match.

    Note that left and right matrix profiles are only available for self-joins.

    Examples
    --------
    >>> import stumpy
    >>> import numpy as np
    >>> from dask.distributed import Client
    >>> if __name__ == "__main__":
    ...     with Client() as dask_client:
    ...         stumpy.stumped(
    ...             dask_client,
    ...             np.array([584., -11., 23., 79., 1001., 0., -19.]),
    ...             m=3)
    mparray([[0.11633857113691416, 4, -1, 4],
             [2.694073918063438, 3, -1, 3],
             [3.0000926340485923, 0, 0, 4],
             [2.694073918063438, 1, 1, -1],
             [0.11633857113691416, 0, 0, -1]], dtype=object)
    >>>
    >>>         mp.P_
    mparray([0.11633857, 2.69407392, 3.00009263, 2.69407392, 0.11633857])
    >>>         mp.I_
    mparray([4, 3, 0, 1, 0])

    Alternatively, you can also use `ray`

    >>> import ray
    >>> if __name__ == "__main__":
    >>>     ray.init()
    >>>     stumpy.stumped(
    ...             ray,
    ...             np.array([584., -11., 23., 79., 1001., 0., -19.]),
    ...             m=3)
    >>>     ray.shutdown()
    """
    if T_B is None:
        T_B = T_A
        ignore_trivial = True
        T_B_subseq_isconstant = T_A_subseq_isconstant

    (
        T_A,
        μ_Q,
        σ_Q_inverse,
        μ_Q_m_1,
        T_A_subseq_isfinite,
        T_A_subseq_isconstant,
    ) = core.preprocess_diagonal(T_A, m, T_subseq_isconstant=T_A_subseq_isconstant)

    (
        T_B,
        M_T,
        Σ_T_inverse,
        M_T_m_1,
        T_B_subseq_isfinite,
        T_B_subseq_isconstant,
    ) = core.preprocess_diagonal(T_B, m, T_subseq_isconstant=T_B_subseq_isconstant)

    if T_A.ndim != 1:  # pragma: no cover
        raise ValueError(
            f"T_A is {T_A.ndim}-dimensional and must be 1-dimensional. "
            "For multidimensional STUMP use `stumpy.mstump` or `stumpy.mstumped`"
        )

    if T_B.ndim != 1:  # pragma: no cover
        raise ValueError(
            f"T_B is {T_B.ndim}-dimensional and must be 1-dimensional. "
            "For multidimensional STUMP use `stumpy.mstump` or `stumpy.mstumped`"
        )

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

    _stumped = core._client_to_func(client)

    out = _stumped(
        client,
        T_A,
        T_B,
        m,
        M_T,
        μ_Q,
        Σ_T_inverse,
        σ_Q_inverse,
        M_T_m_1,
        μ_Q_m_1,
        T_A_subseq_isfinite,
        T_B_subseq_isfinite,
        T_A_subseq_isconstant,
        T_B_subseq_isconstant,
        diags,
        ignore_trivial,
        k,
    )

    core._check_P(out[:, 0])

    return mparray(out, m, k, config.STUMPY_EXCL_ZONE_DENOM)
