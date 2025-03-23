# STUMPY
# Copyright 2019 TD Ameritrade. Released under the terms of the 3-Clause BSD license.
# STUMPY is a trademark of TD Ameritrade IP Company, Inc. All rights reserved.

import warnings

import numpy as np

from . import config, core
from .aamp_mmotifs import aamp_mmotifs
from .motifs import match
from .mstump import mdl


@core.non_normalized(
    aamp_mmotifs,
    exclude=[
        "normalize",
        "T_subseq_isconstant",
    ],
)
def mmotifs(
    T,
    P,
    I,
    min_neighbors=1,
    max_distance=None,
    cutoffs=None,
    max_matches=10,
    max_motifs=1,
    atol=1e-8,
    k=None,
    include=None,
    normalize=True,
    p=2.0,
    T_subseq_isconstant=None,
):
    """
    Discover the top motifs for the multi-dimensional time series ``T``.

    Parameters
    ----------
    T : numpy.ndarray
        The multi-dimensional time series or sequence.

    P : numpy.ndarray
        Multi-dimensional Matrix Profile of ``T``.

    I : numpy.ndarray
        Multi-dimensional Matrix Profile indices.

    min_neighbors : int, default 1
        The minimum number of similar matches a subsequence needs to have in order
        to be considered a motif. This defaults to ``1``, which means that a
        subsequence must have at least one similar match in order to be considered a
        motif.

    max_distance : float, default None
        Maximal distance that is allowed between a query subsequence
        (a candidate motif) and all subsequences in ``T`` to be considered as a
        match. If ``None``, this defaults to
        ``np.nanmax([np.nanmean(D) - 2 * np.nanstd(D), np.nanmin(D)])``
        (i.e. at least the closest match will be returned).

    cutoffs : numpy.ndarray or float, default None
        The largest matrix profile value (distance) for each dimension of the
        multidimensional matrix profile that a multidimenisonal candidate motif is
        allowed to have. If ``cutoffs`` is a scalar value, then this value will be
        applied to every dimension.

    max_matches : int, default 10
        The maximum number of similar matches (nearest neighbors) to return for each
        motif. The first match is always the self/trivial-match for each motif.

    max_motifs : int, default 1
        The maximum number of motifs to return. To consider returning all possible
        valid motifs, try setting `max_motifs` to the length of your input matrix
        profile (i.e., ``max_motifs=len(P)``)

    atol : float, default 1e-8
        The absolute tolerance parameter. This value will be added to ``max_distance``
        when comparing distances between subsequences.

    k : int, default None
        The number of dimensions (``k + 1``) required for discovering all motifs. This
        value is available for doing guided search or, together with ``include``, for
        constrained search. If ``k is None``, then this will be automatically be
        computed for each motif using MDL (unconstrained search).

    include : numpy.ndarray, default None
        A list of (zero based) indices corresponding to the dimensions in ``T`` that
        must be included in the constrained multidimensional motif search.

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
    motif_distances: numpy.ndarray
        The distances corresponding to a set of subsequence matches for each motif.

    motif_indices: numpy.ndarray
        The indices corresponding to a set of subsequences matches for each motif.

    motif_subspaces: list
        A list consisting of arrays that contain the ``k``-dimensional
        subspace for each motif.

    motif_mdls: list
        A list consisting of arrays that contain the mdl results for
        finding the dimension of each motif.

    See Also
    --------
    stumpy.motifs : Find the top motifs for time series ``T``
    stumpy.match : Find all matches of a query ``Q`` in a time series ``T``
    stumpy.mstump : Compute the multi-dimensional z-normalized matrix profile
    stumpy.mstumped : Compute the multi-dimensional z-normalized matrix profile with
        a ``dask``/``ray`` cluster
    stumpy.subspace : Compute the ``k``-dimensional matrix profile subspace for a given
        subsequence index and its nearest neighbor index
    stumpy.mdl : Compute the number of bits needed to compress one array with another
        using the minimum description length (MDL)

    Notes
    -----
    `DOI: 10.1109/ICDM.2017.66 \
    <https://www.cs.ucr.edu/~eamonn/Motif_Discovery_ICDM.pdf>`__

    For more information on ``include`` and search types, see Section IV D and IV E

    Examples
    --------
    >>> import stumpy
    >>> import numpy as np
    >>> mps, indices = stumpy.mstump(
    ...     np.array([[584., -11., 23., 79., 1001., 0., -19.],
    ...               [  1.,   2.,  4.,  8.,   16., 0.,  32.]]),
    ...     m=3)
    >>> stumpy.mmotifs(
    ...     np.array([[584., -11., 23., 79., 1001., 0., -19.],
    ...               [  1.,   2.,  4.,  8.,   16., 0.,  32.]]),
    ...     mps,
    ...     indices)
    (array([[4.47034836e-08, 4.47034836e-08]]),  array([[0, 2]]), [array([1])],
     [array([ 80.      , 111.509775])])
    """
    T = core._preprocess(T)
    m = T.shape[-1] - P.shape[-1] + 1
    reset_k = False

    if max_motifs < 1:  # pragma: no cover
        msg = "The maximum number of motifs, `max_motifs`, "
        msg += "must be greater than or equal to 1.\n"
        msg += "`max_motifs` has been set to `1`"
        warnings.warn(msg)
        max_motifs = 1

    T, M_T, Σ_T, T_subseq_isconstant = core.preprocess(
        T, m, T_subseq_isconstant=T_subseq_isconstant
    )
    P = P.copy()

    excl_zone = int(np.ceil(m / config.STUMPY_EXCL_ZONE_DENOM))

    if max_matches is None:
        max_matches = np.inf

    if cutoffs is None:
        P_copy = P.copy().astype(np.float64)
        P_copy[np.isinf(P_copy)] = np.nan
        cutoffs = np.nanmax(
            [
                np.nanmean(P_copy, axis=1) - 2.0 * np.nanstd(P_copy, axis=1),
                np.nanmin(P_copy, axis=1),
            ],
            axis=0,
        )
    if np.isscalar(cutoffs):
        cutoffs = np.full(P.shape[0], cutoffs)

    motif_distances = []
    motif_indices = []
    motif_subspaces = []
    motif_mdls = []

    candidate_idx = np.argmin(P, axis=1)
    nn_idx = I[np.arange(len(candidate_idx)), candidate_idx]

    while len(motif_distances) < max_motifs:
        mdls, subspaces = mdl(T, m, candidate_idx, nn_idx, include)
        if k is None:
            k = np.argmin(mdls)
            reset_k = True
        subspace_k = subspaces[k]

        motif_idx = candidate_idx[k]
        motif_value = P[k, motif_idx]

        if (
            motif_value > cutoffs[k]
            or not np.isfinite(motif_value)
            or motif_idx < 0
            or nn_idx[k] < 0
            or (isinstance(max_distance, float) and motif_value > max_distance)
        ):  # pragma: no cover
            break

        query_matches = match(
            Q=T[subspace_k, motif_idx : motif_idx + m],
            T=T[subspace_k],
            M_T=M_T[subspace_k],
            Σ_T=Σ_T[subspace_k],
            max_matches=max_matches,
            max_distance=max_distance,
            atol=atol,
            query_idx=motif_idx,
            normalize=normalize,
            p=p,
            T_subseq_isconstant=T_subseq_isconstant[subspace_k],
            Q_subseq_isconstant=np.expand_dims(
                T_subseq_isconstant[subspace_k, motif_idx], axis=1
            ),
        )

        if len(query_matches) > min_neighbors:
            motif_distances.append(query_matches[:, 0])
            motif_indices.append(query_matches[:, 1])
            motif_subspaces.append(subspace_k)
            motif_mdls.append(mdls)

        for idx in query_matches[:, 1]:
            core.apply_exclusion_zone(P, idx, excl_zone, np.inf)
        candidate_idx = np.argmin(P, axis=1)
        nn_idx = I[np.arange(len(candidate_idx)), candidate_idx]
        if reset_k:
            k = None

    motif_distances = core._jagged_list_to_array(
        motif_distances, fill_value=np.nan, dtype=np.float64
    )
    motif_indices = core._jagged_list_to_array(
        motif_indices, fill_value=-1, dtype=np.int64
    )

    return motif_distances, motif_indices, motif_subspaces, motif_mdls
