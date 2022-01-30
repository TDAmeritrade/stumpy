import numpy as np

from . import core, config


def Mmotifs(T: np.ndarray, P: np.ndarray, max_matches: int = 10, max_motifs: int = 1):
    """
    Discover the top k multidimensional motifs for the time series T

    Parameters
    ----------
    T: numpy.ndarray
        The multidimensional time series or sequence

    P: numpy.ndarray
        Matrix Profile of T

    max_matches: int, default 10
        The maximum amount of similar matches (nearest neighbors) of a motif representative to be returned

    max_motifs: int, default 1
        The maximum number of motifs to return

    Returns
    -------

    """
