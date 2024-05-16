import numpy as np


class mparray(np.ndarray):
    """
    A matrix profile convenience class that subclasses the numpy ndarray
    """

    def __new__(cls, input_array, m, k=1):
        """
        Create the ndarray instance of our type, given the usual
        ndarray input arguments.  This will call the standard
        ndarray constructor, but return an object of our type.
        It also triggers a call mparrau.__array_finalize__

        Parameters
        ----------
        cls : class
            THe base class

        input_array : ndarray
            The input `numpy` array to be sublcassed

        m : int
            Window size

        k : int, default 1
            The number of top `k` smallest distances used to construct the
            matrix profile.
        """
        obj = np.asarray(input_array).view(cls)
        obj.m = m
        obj.k = k
        # All additional attributes need to also be
        # appended in the `__array_finalize` method
        return obj

    def __array_finalize__(self, obj):
        """
        Finalize the array

        Parameters
        ----------
        obj : object
            This is the class object
        """
        if obj is None:
            return
        self._m = getattr(obj, "m", None)
        self._k = getattr(obj, "k", None)

    def _P(self):
        """
        Matrix profile values

        Parameters
        ----------
        None
        """
        return self[:, : self._k]

    def _I(self):
        """
        Nearest neighbor indices

        Parameters
        ----------
        None
        """
        return self[:, self._k : 2 * self._k]

    def _left_I(self):
        """
        Left nearest neighbor indices

        Parameters
        ----------
        None
        """
        return self[:, 2 * self._k]

    def _right_I(self):
        """
        Right nearest neighbor indices

        Parameters
        ----------
        None
        """
        return self[:, 2 * self._k + 1]

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
        return self._IL()

    @property
    def right_I_(self):
        """
        Right nearest neighbor indices

        Parameters
        ----------
        None
        """
        return self._IR()
