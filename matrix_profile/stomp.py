#!/usr/bin/env python

from . import core
from . import stamp

class STOMP(stamp.STAMP):

    def get_matrix_profile(self):
        """
        """
        return


if __name__ == '__main__':
    core.check_python_version()
    parser = core.get_parser()
    args = parser.parse_args()
    mp = STOMP()
    mp.get_matrix_profile()
