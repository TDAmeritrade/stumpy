import numpy as np


class mparray(np.ndarray):
    """
    A matrix profile convenience class that subclasses the numpy ndarray

    Parameters
    ----------
    cls : class
        The base class

    input_array : ndarray
        The input `numpy` array to be subclassed

    m : int
        Window size

    k : int
        The number of top `k` smallest distances used to construct the
        matrix profile.

    excl_zone_denom : int
        The denominator used in computing the exclusion zone

    Attributes
    ----------
    P_ : numpy.ndarray
        The (top-k) matrix profile for `T`. When `k=1`, the first
        (and only) column in this 2D array, which consists of the matrix profile,
        is returned. When `k > 1`, the output has exactly `k` columns consisting of
        the top-k matrix profile.

    I_ : numpy.ndarray
        The(top-k) matrix profile indices for `T`. When `k=1`, the first
        (and only) column in this 2D array, which consists of the matrix profile,
        indices is returned. When `k > 1`, the output has exactly `k` columns
        consisting of the top-k matrix profile indices.

    left_I_ : numpy.ndarray
        The left (top-1) matrix profile indices for `T`

    right_I_ : numpy.ndarray
        The right (top-1) matrix profile indices for `T`
    """

    def __new__(cls, input_array, m, k, excl_zone_denom):
        """
        Create the ndarray instance of our type, given the usual
        ndarray input arguments.  This will call the standard
        ndarray constructor, but return an object of our type.
        It also triggers a call mparray.__array_finalize__

        Parameters
        ----------
        cls : class
            The base class

        input_array : ndarray
            The input `numpy` array to be subclassed

        m : int
            Window size

        k : int
            The number of top `k` smallest distances used to construct the
            matrix profile

        excl_zone_denom : int
            The denominator used in computing the exclusion zone
        """
        obj = np.asarray(input_array).view(cls)
        obj._m = m
        obj._k = k
        obj._excl_zone_denom = excl_zone_denom
        # All new attributes will also need to be added to the `__array_finalize__`
        # function below so that "new-from-template" objects (e.g., an array slice)
        # will also contain the same new attributes
        return obj

    def __array_finalize__(self, obj):
        """
        Finalize the array

        Parameters
        ----------
        obj : object
            This is the class object
        """
        if obj is None:  # pragma: no cover
            return
        # The lines below ensure that child objects that are created from a slice
        # of an `mparray` will also inherit the attributes from the parent `mparray`
        self._m = getattr(obj, "_m", None)
        self._k = getattr(obj, "_k", None)
        self._excl_zone_denom = getattr(obj, "_excl_zone_denom", None)

    def _P(self):
        """
        Matrix profile values

        Parameters
        ----------
        None
        """
        if self._k == 1:
            return self[:, : self._k].flatten().astype(np.float64)
        else:
            return self[:, : self._k].astype(np.float64)

    def _I(self):
        """
        Nearest neighbor indices

        Parameters
        ----------
        None
        """
        if self._k == 1:
            return self[:, self._k : 2 * self._k].flatten().astype(np.int64)
        else:
            return self[:, self._k : 2 * self._k].astype(np.int64)

    def _left_I(self):
        """
        Left nearest neighbor indices

        Parameters
        ----------
        None
        """
        if self._k == 1:
            return self[:, 2 * self._k].flatten().astype(np.int64)
        else:
            return self[:, 2 * self._k].astype(np.int64)

    def _right_I(self):
        """
        Right nearest neighbor indices

        Parameters
        ----------
        None
        """
        if self._k == 1:
            return self[:, 2 * self._k + 1].flatten().astype(np.int64)
        else:
            return self[:, 2 * self._k + 1].astype(np.int64)

    @property
    def P_(self):
        """
        Matrix profile values

        Parameters
        ----------
        None
        """
        return self._P()

    @property
    def I_(self):
        """
        Nearest neighbor indices

        Parameters
        ----------
        None
        """
        return self._I()

    @property
    def left_I_(self):
        """
        Left nearest neighbor indices

        Parameters
        ----------
        None
        """
        return self._left_I()

    @property
    def right_I_(self):
        """
        Right nearest neighbor indices

        Parameters
        ----------
        None
        """
        return self._right_I()
