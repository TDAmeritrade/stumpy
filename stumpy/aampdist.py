# STUMPY
# Copyright 2019 TD Ameritrade. Released under the terms of the 3-Clause BSD license.
# STUMPY is a trademark of TD Ameritrade IP Company, Inc. All rights reserved.

from . import aamp, aamped, mpdist
from .core import _mass_absolute_distance_matrix


def _aampdist_vect(
    Q,
    T,
    m,
    percentage=0.05,
    k=None,
    custom_func=None,
):
    """
    Compute the non-normalized (i.e., without z-normalization) matrix profile distance
    measure vector between `Q` and each subsequence, `T[i : i + len(Q)]`, within `T`.

    Parameters
    ----------
    Q : ndarray
        Query array

    T : ndarray
        Time series or sequence

    m : int
        Window size

    percentage : float, 0.05
        The percentage of distances that will be used to report `mpdist`. The value
        is between 0.0 and 1.0. This parameter is ignored when `k` is not `None` or when
        `k_func` is not None.

    k : int, default None
        Specify the `k`th value in the concatenated matrix profiles to return. When `k`
        is not `None`, then the `percentage` parameter is ignored. This parameter is
        ignored when `custom_func` is not None.

    custom_func : object, default None
        A custom user defined function for selecting the desired value from the
        unsorted `P_ABBA` array. This function may need to leverage `functools.partial`
        and should take `P_ABBA` as its only input parameter and return a single
        `MPdist` value. The `percentage` and `k` parameters are ignored when
        `custom_func` is not None.
    """
    return mpdist._mpdist_vect(
        Q,
        T,
        m,
        percentage=percentage,
        k=k,
        custom_func=custom_func,
        distance_matrix_func=_mass_absolute_distance_matrix,
    )


def aampdist(T_A, T_B, m, percentage=0.05, k=None):
    """
    Compute the non-normalized (i.e., without z-normalization) matrix profile distance
    (MPdist) measure between any two time series with `stumpy.aamp`.

    The MPdist distance measure considers two time series to be similar if they share
    many subsequences, regardless of the order of matching subsequences. MPdist
    concatenates the output of an AB-join and a BA-join and returns the `k`th smallest
    value as the reported distance. Note that MPdist is a measure and not a metric.
    Therefore, it does not obey the triangular inequality but the method is highly
    scalable.

    Parameters
    ----------
    T_A : ndarray
        The first time series or sequence for which to compute the matrix profile

    T_B : ndarray
        The second time series or sequence for which to compute the matrix profile

    m : int
        Window size

    percentage : float, default 0.05
        The percentage of distances that will be used to report `mpdist`. The value
        is between 0.0 and 1.0.

    k : int
        Specify the `k`th value in the concatenated matrix profiles to return. When `k`
        is not `None`, then the `percentage` parameter is ignored.

    Returns
    -------
    MPdist : float
        The matrix profile distance

    Notes
    -----
    `DOI: 10.1109/ICDM.2018.00119 \
    <https://www.cs.ucr.edu/~eamonn/MPdist_Expanded.pdf>`__

    See Section III
    """
    return mpdist._mpdist(T_A, T_B, m, percentage, k, mp_func=aamp)


def aampdisted(dask_client, T_A, T_B, m, percentage=0.05, k=None):
    """
    Compute the non-normalized (i.e., without z-normalization) matrix profile distance
    (MPdist) measure between any two time series with a distributed dask cluster and
    `stumpy.aamped`.

    The MPdist distance measure considers two time series to be similar if they share
    many subsequences, regardless of the order of matching subsequences. MPdist
    concatenates the output of an AB-join and a BA-join and returns the `k`th smallest
    value as the reported distance. Note that MPdist is a measure and not a metric.
    Therefore, it does not obey the triangular inequality but the method is highly
    scalable.

    Parameters
    ----------
    dask_client : client
        A Dask Distributed client that is connected to a Dask scheduler and
        Dask workers. Setting up a Dask distributed cluster is beyond the
        scope of this library. Please refer to the Dask Distributed
        documentation.

    T_A : ndarray
        The first time series or sequence for which to compute the matrix profile

    T_B : ndarray
        The second time series or sequence for which to compute the matrix profile

    m : int
        Window size

    percentage : float, default 0.05
        The percentage of distances that will be used to report `mpdist`. The value
        is between 0.0 and 1.0. This parameter is ignored when `k` is not `None`.

    k : int
        Specify the `k`th value in the concatenated matrix profiles to return. When `k`
        is not `None`, then the `percentage` parameter is ignored.

    Returns
    -------
    MPdist : float
        The matrix profile distance

    Notes
    -----
    `DOI: 10.1109/ICDM.2018.00119 \
    <https://www.cs.ucr.edu/~eamonn/MPdist_Expanded.pdf>`__

    See Section III
    """
    return mpdist._mpdist(
        T_A, T_B, m, percentage, k, dask_client=dask_client, mp_func=aamped
    )
