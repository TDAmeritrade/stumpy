# STUMPY
# Copyright 2019 TD Ameritrade. Released under the terms of the 3-Clause BSD license.
# STUMPY is a trademark of TD Ameritrade IP Company, Inc. All rights reserved.

import math

import numpy as np

from . import config, core
from .maamped import maamped
from .mmparray import mparray
from .mstump import _get_first_mstump_profile, _get_multi_QT, _mstump


def _dask_mstumped(
    dask_client,
    T_A,
    T_B,
    m,
    excl_zone,
    M_T,
    Σ_T,
    μ_Q,
    σ_Q,
    T_subseq_isconstant,
    Q_subseq_isconstant,
    include,
    discords,
):
    """
    Compute the multi-dimensional z-normalized matrix profile with a `dask` cluster

    This is a highly distributed implementation around the Numba JIT-compiled
    parallelized `_mstump` function which computes the multi-dimensional matrix
    profile according to STOMP. Note that only self-joins are supported.

    Parameters
    ----------
    dask_client : client
        A ``dask`` client. Setting up a ``dask`` cluster is beyond
        the scope of this library. Please refer to the ``dask``
        documentation.

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

    M_T : numpy.ndarray
        Sliding mean of time series, `T`

    Σ_T : numpy.ndarray
        Sliding standard deviation of time series, `T`

    μ_Q : numpy.ndarray
        Mean of the query sequence, `Q`, relative to the current sliding window

    σ_Q : numpy.ndarray
        Standard deviation of the query sequence, `Q`, relative to the current
        sliding window

    T_subseq_isconstant : numpy.ndarray
        A boolearn array representing Rolling isconstant for `T`

    Q_subseq_isconstant : numpy.ndarray
        A boolearn array representing Rolling isconstant for `Q`

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
        P[:, start], I[:, start] = _get_first_mstump_profile(
            start,
            T_A,
            T_B,
            m,
            excl_zone,
            M_T,
            Σ_T,
            μ_Q,
            σ_Q,
            T_subseq_isconstant,
            Q_subseq_isconstant,
            include,
            discords,
        )

    # Scatter data to Dask cluster
    T_A_future = dask_client.scatter(T_A, broadcast=True, hash=False)
    M_T_future = dask_client.scatter(M_T, broadcast=True, hash=False)
    Σ_T_future = dask_client.scatter(Σ_T, broadcast=True, hash=False)
    μ_Q_future = dask_client.scatter(μ_Q, broadcast=True, hash=False)
    σ_Q_future = dask_client.scatter(σ_Q, broadcast=True, hash=False)
    T_subseq_isconstant_future = dask_client.scatter(
        T_subseq_isconstant, broadcast=True, hash=False
    )
    Q_subseq_isconstant_future = dask_client.scatter(
        Q_subseq_isconstant, broadcast=True, hash=False
    )

    QT_futures = []
    QT_first_futures = []

    for i, start in enumerate(range(0, l, step)):
        QT, QT_first = _get_multi_QT(start, T_A, m)

        QT_future = dask_client.scatter(QT, workers=[hosts[i]], hash=False)
        QT_first_future = dask_client.scatter(QT_first, workers=[hosts[i]], hash=False)

        QT_futures.append(QT_future)
        QT_first_futures.append(QT_first_future)

    futures = []
    for i, start in enumerate(range(0, l, step)):
        stop = min(l, start + step)

        futures.append(
            dask_client.submit(
                _mstump,
                T_A_future,
                m,
                stop,
                excl_zone,
                M_T_future,
                Σ_T_future,
                QT_futures[i],
                QT_first_futures[i],
                μ_Q_future,
                σ_Q_future,
                T_subseq_isconstant_future,
                Q_subseq_isconstant_future,
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


def _ray_mstumped(
    ray_client,
    T_A,
    T_B,
    m,
    excl_zone,
    M_T,
    Σ_T,
    μ_Q,
    σ_Q,
    T_subseq_isconstant,
    Q_subseq_isconstant,
    include,
    discords,
):
    """
    Compute the multi-dimensional z-normalized matrix profile with a `ray` cluster

    This is a highly distributed implementation around the Numba JIT-compiled
    parallelized `_mstump` function which computes the multi-dimensional matrix
    profile according to STOMP. Note that only self-joins are supported.

    Parameters
    ----------
    ray_client : client
        A `ray` client. Setting up a cluster is beyond the scope of this library.
        Please refer to the `ray` documentation.

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

    M_T : numpy.ndarray
        Sliding mean of time series, `T`

    Σ_T : numpy.ndarray
        Sliding standard deviation of time series, `T`

    μ_Q : numpy.ndarray
        Mean of the query sequence, `Q`, relative to the current sliding window

    σ_Q : numpy.ndarray
        Standard deviation of the query sequence, `Q`, relative to the current
        sliding window

    T_subseq_isconstant : numpy.ndarray
        A boolearn array representing Rolling isconstant for `T`

    Q_subseq_isconstant : numpy.ndarray
        A boolearn array representing Rolling isconstant for `Q`

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
        P[:, start], I[:, start] = _get_first_mstump_profile(
            start,
            T_A,
            T_B,
            m,
            excl_zone,
            M_T,
            Σ_T,
            μ_Q,
            σ_Q,
            T_subseq_isconstant,
            Q_subseq_isconstant,
            include,
            discords,
        )

    # Put data into Ray object storage
    T_A_ref = ray_client.put(T_A)
    M_T_ref = ray_client.put(M_T)
    Σ_T_ref = ray_client.put(Σ_T)
    μ_Q_ref = ray_client.put(μ_Q)
    σ_Q_ref = ray_client.put(σ_Q)
    T_subseq_isconstant_ref = ray_client.put(T_subseq_isconstant)
    Q_subseq_isconstant_ref = ray_client.put(Q_subseq_isconstant)

    QT_refs = []
    QT_first_refs = []

    for start in range(0, l, step):
        QT, QT_first = _get_multi_QT(start, T_A, m)

        QT_ref = ray_client.put(QT)
        QT_first_ref = ray_client.put(QT_first)

        QT_refs.append(QT_ref)
        QT_first_refs.append(QT_first_ref)

    ray_mstump_func = ray_client.remote(core.deco_ray_tor(_mstump))

    refs = []
    for i, start in enumerate(range(0, l, step)):
        stop = min(l, start + step)

        refs.append(
            ray_mstump_func.remote(
                T_A_ref,
                m,
                stop,
                excl_zone,
                M_T_ref,
                Σ_T_ref,
                QT_refs[i],
                QT_first_refs[i],
                μ_Q_ref,
                σ_Q_ref,
                T_subseq_isconstant_ref,
                Q_subseq_isconstant_ref,
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


@core.non_normalized(
    maamped,
    exclude=["normalize", "T_subseq_isconstant"],
)
def mstumped(
    client,
    T,
    m,
    include=None,
    discords=False,
    p=2.0,
    normalize=True,
    T_subseq_isconstant=None,
):
    """
    Compute the multi-dimensional z-normalized matrix profile with a
    ``dask``/``ray`` cluster

    This is a highly distributed implementation around the Numba JIT-compiled
    parallelized ``_mstump`` function which computes the multi-dimensional matrix
    profile according to STOMP. Note that only self-joins are supported.

    Parameters
    ----------
    client : client
        A ``dask``/``ray`` client. Setting up a cluster is beyond the scope of this
        library. Please refer to the ``dask``/``ray`` documentation.

    T : numpy.ndarray
        The time series or sequence for which to compute the multi-dimensional
        matrix profile. Each row in ``T`` represents data from the same
        dimension while each column in ``T`` represents data from a different
        dimension.

    m : int
        Window size.

    include : list, numpy.ndarray, default None
        A list of (zero-based) indices corresponding to the dimensions in ``T`` that
        must be included in the constrained multidimensional motif search.
        For more information, see Section IV D in:

        `DOI: 10.1109/ICDM.2017.66 \
        <https://www.cs.ucr.edu/~eamonn/Motif_Discovery_ICDM.pdf>`__

    discords : bool, default False
        When set to ``True``, this reverses the distance matrix which results in a
        multi-dimensional matrix profile that favors larger matrix profile values
        (i.e., discords) rather than smaller values (i.e., motifs). Note that indices
        in `include` are still maintained and respected.

    p : float, default 2.0
        The p-norm to apply for computing the Minkowski distance. Minkowski distance is
        typically used with ``p`` being ``1`` or ``2``, which correspond to the
        Manhattan distance and the Euclidean distance, respectively.

    normalize : bool, default True
        When set to ``True``, this z-normalizes subsequences prior to computing
        distances. Otherwise, this function gets re-routed to its complementary
        non-normalized equivalent set in the ``@core.non_normalized`` function
        decorator.

    T_subseq_isconstant : numpy.ndarray, function, or list, default None
        A parameter that is used to show whether a subsequence of a time series in ``T``
        is constant (``True``) or not. ``T_subseq_isconstant`` can be a 2D boolean
        ``numpy.ndarray`` or a function that can be applied to each time series in
        ``T``. Alternatively, for maximum flexibility, a list (with length equal to the
        total number of time series) may also be used. In this case,
        ``T_subseq_isconstant[i]`` corresponds to the ``i``-th time series ``T[i]`` and
        each element in the list can either be a 1D boolean ``numpy.ndarray``, a
        function, or ``None``.

    Returns
    -------
    P : numpy.ndarray
        The multi-dimensional matrix profile. Each row of the array corresponds
        to each matrix profile for a given dimension (i.e., the first row is
        the 1-D matrix profile and the second row is the 2-D matrix profile).

    I : numpy.ndarray
        The multi-dimensional matrix profile index where each row of the array
        corresponds to each matrix profile index for a given dimension.

    See Also
    --------
    stumpy.mstump : Compute the multi-dimensional z-normalized matrix profile
    stumpy.subspace : Compute the k-dimensional matrix profile subspace for a given
        subsequence index and its nearest neighbor index
    stumpy.mdl : Compute the number of bits needed to compress one array with another
        using the minimum description length (MDL)

    Notes
    -----
    `DOI: 10.1109/ICDM.2017.66 \
    <https://www.cs.ucr.edu/~eamonn/Motif_Discovery_ICDM.pdf>`__

    See mSTAMP Algorithm

    Examples
    --------
    >>> import stumpy
    >>> import numpy as np
    >>> from dask.distributed import Client
    >>> if __name__ == "__main__":
    ...     with Client() as dask_client:
    ...         stumpy.mstumped(
    ...             dask_client,
    ...             np.array([[584., -11., 23., 79., 1001., 0., -19.],
    ...                       [  1.,   2.,  4.,  8.,   16., 0.,  32.]]),
    ...             m=3)
    (array([[0.        , 1.43947142, 0.        , 2.69407392, 0.11633857],
            [0.777905  , 2.36179922, 1.50004632, 2.92246722, 0.777905  ]]),
     array([[2, 4, 0, 1, 0],
            [4, 4, 0, 1, 0]]))

    Alternatively, you can also use `ray`

    >>> import ray
    >>> if __name__ == "__main__":
    >>>     ray.init()
    >>>     stumpy.mstumped(
    ...         ray,
    ...         np.array([[584., -11., 23., 79., 1001., 0., -19.],
    ...                   [  1.,   2.,  4.,  8.,   16., 0.,  32.]]),
    ...         m=3)
    >>>     ray.shutdown()
    """
    T_A = T
    T_B = T_A

    T_A = core._preprocess(T_A)
    T_B = core._preprocess(T_B)

    T_A_subseq_isconstant = T_subseq_isconstant
    T_A_subseq_isconstant = core.process_isconstant(T_A, m, T_A_subseq_isconstant)
    T_B_subseq_isconstant = T_A_subseq_isconstant

    T_A, M_T, Σ_T, T_subseq_isconstant = core.preprocess(
        T_A, m, T_subseq_isconstant=T_A_subseq_isconstant
    )
    T_B, μ_Q, σ_Q, Q_subseq_isconstant = core.preprocess(
        T_B, m, T_subseq_isconstant=T_B_subseq_isconstant
    )

    if T_A.ndim <= 1:  # pragma: no cover
        err = f"T is {T_A.ndim}-dimensional and must be at least 2-dimensional"
        raise ValueError(f"{err}")

    # mstump currently only supports self-join. Therefore, the argument `n=T_A.shape[1]`
    # must be passed to the function `core.check_window_size`.
    core.check_window_size(m, max_size=min(T_A.shape[1], T_B.shape[1]), n=T_A.shape[1])

    if include is not None:
        include = core._preprocess_include(include)

    excl_zone = int(
        np.ceil(m / config.STUMPY_EXCL_ZONE_DENOM)
    )  # See Definition 3 and Figure 3

    _mstumped = core._client_to_func(client)

    P, I = _mstumped(
        client,
        T_A,
        T_B,
        m,
        excl_zone,
        M_T,
        Σ_T,
        μ_Q,
        σ_Q,
        T_subseq_isconstant,
        Q_subseq_isconstant,
        include,
        discords,
    )

    return mparray(P_=P, I_=I)
