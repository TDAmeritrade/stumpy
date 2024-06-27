# STUMPY
# Copyright 2019 TD Ameritrade. Released under the terms of the 3-Clause BSD license.
# STUMPY is a trademark of TD Ameritrade IP Company, Inc. All rights reserved.

import math

import numpy as np

from . import core
from .aampdist_snippets import aampdist_snippets
from .core import _get_mask_slices, check_window_size
from .mpdist import _mpdist_vect


def _get_all_profiles(
    T,
    m,
    percentage=1.0,
    s=None,
    mpdist_percentage=0.05,
    mpdist_k=None,
    mpdist_custom_func=None,
    mpdist_T_subseq_isconstant=None,
):
    """
    For each non-overlapping subsequence, `S[i]`, in `T`, compute the matrix profile
    distance measure vector between the `i`th non-overlapping subsequence and each
    sliding window subsequence, `T[j : j + m]`, within `T` where `j < len(T) - m + 1`.

    Parameters
    ----------
    T : numpy.ndarray
        The time series or sequence for which to find the snippets

    m : int
        The window size for each non-overlapping subsequence, `S[i]`.

    percentage : float, default 1.0
        With the length of each non-overlapping subsequence, `S[i]`, set to `m`, this
        is the percentage of `S[i]` (i.e., `percentage * m`) to set the `s` to. When
        `percentage == 1.0`, then the full length of `S[i]` is used to compute the
        `mpdist_vect`. When `percentage < 1.0`, then shorter subsequences from `S[i]`
        is used to compute `mpdist_vect`.

    s : int, default None
        With the length of each non-overlapping subsequence, `S[i]`, set to `m`, this
        is essentially the sub-subsequence length (i.e., a shorter part of `S[i]`).
        When `s == m`, then the full length of `S[i]` is used to compute the
        `mpdist_vect`. When `s < m`, then shorter subsequences with length `s` from
        each `S[i]` is used to compute `mpdist_vect`. When `s` is not `None`, then
        the `percentage` parameter is ignored.

    mpdist_percentage : float, default 0.05
        The percentage of distances that will be used to report `mpdist`. The value
        is between 0.0 and 1.0.

    mpdist_k : int
        Specify the `k`th value in the concatenated matrix profiles to return. When
        `mpdist_k` is not `None`, then the `mpdist_percentage` parameter is ignored.

    mpdist_custom_func : func, default None
        A custom user defined function for selecting the desired value from the
        sorted `P_ABBA` array. This function may need to leverage `functools.partial`
        and should take `P_ABBA` as its only input parameter and return a single
        `MPdist` value. The `percentage` and `k` parameters are ignored when
        `mpdist_custom_func` is not None.

    mpdist_T_subseq_isconstant : numpy.ndarray or function, default None
        A boolean array that indicates whether a subsequence (of length `s`) in `T`
        is constant (True). Alternatively, a custom, user-defined function that
        returns a boolean array that indicates whether a subsequence in `T` is
        constant (True). The function must only take two arguments, `a`, a 1-D array,
        and `w`, the window size, while additional arguments may be specified
        by currying the user-defined function using `functools.partial`. Any
        subsequence with at least one np.nan/np.inf will automatically have its
        corresponding value set to False in this boolean array.

    Returns
    -------
    D : numpy.ndarray
        MPdist profiles

    Notes
    -----
    `DOI: 10.1109/ICBK.2018.00058 \
    <https://www.cs.ucr.edu/~eamonn/Time_Series_Snippets_10pages.pdf>`__

    See Table II
    """
    if m > T.shape[0] // 2:  # pragma: no cover
        raise ValueError(
            f"The window size {m} for each non-overlapping subsequence is too large "
            f"for a time series with length {T.shape[0]}. "
            f"Please try `m <= len(T) // 2`."
        )

    if s is not None:
        s = min(int(s), m)
    else:
        percentage = np.clip(percentage, 0.0, 1.0)
        s = min(math.ceil(percentage * m), m)

    right_pad = 0
    T_subseq_isconstant = core.process_isconstant(T, s, mpdist_T_subseq_isconstant)
    n_contiguous_windows = int(T.shape[0] // m)
    if T.shape[0] % m != 0:
        right_pad = int(m * np.ceil(T.shape[0] / m) - T.shape[0])
        pad_width = (0, right_pad)
        T = np.pad(T, pad_width, mode="constant", constant_values=np.nan)
        T_subseq_isconstant = np.pad(
            T_subseq_isconstant, pad_width, mode="constant", constant_values=False
        )

    n_padded = T.shape[0]
    D = np.empty((n_contiguous_windows, n_padded - m + 1), dtype=np.float64)

    M_T, Σ_T = core.compute_mean_std(T, s)

    # Iterate over non-overlapping subsequences, see Definition 3
    for i in range(n_contiguous_windows):
        start = i * m
        stop = (i + 1) * m
        S_i = T[start:stop]
        D[i, :] = _mpdist_vect(
            S_i,
            T,
            s,
            M_T[start : stop - s + 1],
            Σ_T[start : stop - s + 1],
            M_T,
            Σ_T,
            T_subseq_isconstant[start : stop - s + 1],
            T_subseq_isconstant,
            percentage=mpdist_percentage,
            k=mpdist_k,
            custom_func=mpdist_custom_func,
            query_idx=start,
        )

    stop_idx = n_padded - m + 1 - right_pad
    D = D[:, :stop_idx]

    return D


@core.non_normalized(
    aampdist_snippets,
    exclude=[
        "normalize",
        "mpdist_T_subseq_isconstant",
    ],
)
def snippets(
    T,
    m,
    k,
    percentage=1.0,
    s=None,
    mpdist_percentage=0.05,
    mpdist_k=None,
    normalize=True,
    p=2.0,
    mpdist_T_subseq_isconstant=None,
):
    """
    Identify the top ``k`` snippets that best represent the time series, ``T``

    Parameters
    ----------
    T : numpy.ndarray
        The time series or sequence for which to find the snippets.

    m : int
        The snippet window size.

    k : int
        The desired number of snippets.

    percentage : float, default 1.0
        With the length of each non-overlapping subsequence, ``S[i]``, set to ``m``,
        this is the percentage of ``S[i]`` (i.e., ``percentage * m``) to set ``s`` (the
        sub-subsequence length) to. When ``percentage == 1.0``, then the full length of
        ``S[i]`` is used to compute the ``mpdist_vect``. When ``percentage < 1.0``,
        then a shorter sub-subsequence length of
        ``s = min(math.ceil(percentage * m), m)`` from each ``S[i]`` is used to compute
        ``mpdist_vect``. When ``s`` is not ``None``, then the ``percentage`` parameter
        is ignored.

    s : int, default None
        With the length of each non-overlapping subsequence, ``S[i]``, set to ``m``,
        this is essentially the sub-subsequence length (i.e., a shorter part of
        ``S[i]``). When ``s == m``, then the full length of ``S[i]`` is used to compute
        the ``mpdist_vect``. When ``s < m``, then shorter subsequences with length
        ``s`` from each ``S[i]`` is used to compute ``mpdist_vect``. When ``s`` is not
        ``None``, then the ``percentage`` parameter is ignored.

    mpdist_percentage : float, default 0.05
        The percentage of distances that will be used to report ``mpdist``. The value
        is between ``0.0`` and ``1.0``.

    mpdist_k : int
        Specify the ``k``-th value in the concatenated matrix profiles to return. When
        ``mpdist_k`` is not ``None``, then the ``mpdist_percentage`` parameter is
        ignored.

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

    mpdist_T_subseq_isconstant : numpy.ndarray or function, default None
        A boolean array that indicates whether a subsequence (of length equal to
        ``len(s)``) in ``T`` is constant (``True``). Alternatively, a custom,
        user-defined function that returns a boolean array that indicates whether a
        subsequence in ``T`` is constant (``True``). The function must only take two
        arguments, ``a``, a 1-D array, and ``w``, the window size, while additional
        arguments may be specified by currying the user-defined function using
        ``functools.partial``. Any subsequence with at least one ``np.nan``/``np.inf``
        will automatically have its corresponding value set to ``False`` in this
        boolean array.

    Returns
    -------
    snippets : numpy.ndarray
        The top ``k`` snippets.

    snippets_indices : numpy.ndarray
        The index locations for each of top ``k`` snippets.

    snippets_profiles : numpy.ndarray
        The MPdist profiles for each of the top  ``k`` snippets.

    snippets_fractions : numpy.ndarray
        The fraction of data that each of the top ``k`` snippets represents.

    snippets_areas : numpy.ndarray
        The area under the curve corresponding to each profile for each of the top
        ``k`` snippets.

    snippets_regimes: numpy.ndarray
        The index slices corresponding to the set of regimes for each of the top ``k``
        snippets. The first column is the (zero-based) snippet index while the second
        and third columns correspond to the (inclusive) regime start indices and the
        (exclusive) regime stop indices, respectively.

    Notes
    -----
    `DOI: 10.1109/ICBK.2018.00058 \
    <https://www.cs.ucr.edu/~eamonn/Time_Series_Snippets_10pages.pdf>`__

    See Table I

    Examples
    --------
    >>> import stumpy
    >>> import numpy as np
    >>> stumpy.snippets(np.array([584., -11., 23., 79., 1001., 0., -19.]), m=3, k=2)
    (array([[ 584.,  -11.,   23.],
            [  79., 1001.,    0.]]),
     array([0, 3]),
     array([[0.        , 3.2452632 , 3.00009263, 2.982409  , 0.11633857],
            [2.982409  , 2.69407392, 3.01719586, 0.        , 2.92154586]]),
    array([0.6, 0.4]),
    array([9.3441034 , 5.81050512]),
    array([[0, 0, 1],
           [0, 2, 3],
           [0, 4, 5],
           [1, 1, 2],
           [1, 3, 4]]))
    """
    T = core._preprocess(T)

    if m > T.shape[0] // 2:  # pragma: no cover
        raise ValueError(
            f"The snippet window size of {m} is too large for a time series with "
            f"length {T.shape[0]}. Please try `m <= len(T) // 2`."
        )

    check_window_size(m, max_size=T.shape[0] // 2)

    D = _get_all_profiles(
        T,
        m,
        percentage=percentage,
        s=s,
        mpdist_percentage=mpdist_percentage,
        mpdist_k=mpdist_k,
        mpdist_T_subseq_isconstant=mpdist_T_subseq_isconstant,
    )

    snippets = np.empty((k, m), dtype=np.float64)
    snippets_indices = np.empty(k, dtype=np.int64)
    snippets_profiles = np.empty((k, D.shape[-1]), dtype=np.float64)
    snippets_fractions = np.empty(k, dtype=np.float64)
    snippets_areas = np.empty(k, dtype=np.float64)
    Q = np.full(D.shape[-1], np.inf, dtype=np.float64)
    indices = np.arange(D.shape[0], dtype=np.int64) * m
    snippets_regimes_list = []

    for i in range(k):
        profile_areas = np.sum(np.minimum(D, Q), axis=1)
        idx = np.argmin(profile_areas)

        snippets[i] = T[indices[idx] : indices[idx] + m]
        snippets_indices[i] = indices[idx]
        snippets_profiles[i] = D[idx]
        snippets_areas[i] = np.sum(np.minimum(D[idx], Q))

        Q[:] = np.minimum(D[idx], Q)

    total_min = np.min(snippets_profiles, axis=0)

    for i in range(k):
        mask = snippets_profiles[i] <= total_min
        snippets_fractions[i] = np.sum(mask) / total_min.shape[0]
        total_min = total_min - mask.astype(np.float64)
        slices = _get_mask_slices(mask)
        snippets_regimes_list.append(slices)

    n_slices = [regime.shape[0] for regime in snippets_regimes_list]
    snippets_regimes = np.empty((sum(n_slices), 3), dtype=np.int64)
    snippets_regimes[:, 0] = np.repeat(np.arange(len(snippets_regimes_list)), n_slices)
    snippets_regimes[:, 1:] = np.vstack(snippets_regimes_list)

    return (
        snippets,
        snippets_indices,
        snippets_profiles,
        snippets_fractions,
        snippets_areas,
        snippets_regimes,
    )
