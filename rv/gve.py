"""Gauss's Variational Equations equivalents for position-velocity elements."""

import numpy as np

__author__ = "Nathan I. Budd"
__email__ = "nibudd@gmail.com"
__copyright__ = "Copyright 2017, LASR Lab"
__license__ = "MIT"
__version__ = "0.1"
__status__ = "Production"
__date__ = "02 Mar 2017"


class GVE(object):
    """Creates mapping of LVLH accelerations into RV time derivatives.

    Attributes:
        mu: float
            Standard Gravitational Parameter
    """

    def __init__(self, mu=1.0):
        self.mu = mu

    def __call__(T, X):
        """Create GVE mapping.

        Args:
            T: ndarray
                (m, 1) array of times.
            X: ndarray
                (m, 6) array of states.
                Columns are ordered as (rx, ry, rz, vx, vy, vz).

        Returns:
            Xdot: ndarray
                (m, 6) array of state time derivatives in the same order as X.
        """
