"""Keplerian trajectory for position-velocity elements."""

import orbital_mechanics.orbit as orb
from orbital_elements.coe.keplerian_path import KeplerianPath as coeKepPath

__author__ = "Nathan I. Budd"
__email__ = "nibudd@gmail.com"
__copyright__ = "Copyright 2017, LASR Lab"
__license__ = "MIT"
__version__ = "0.1"
__status__ = "Production"
__date__ = "02 Mar 2017"


class KeplerianPath(object):
    """Calculate Keplerian trajectory in RV elements.

    Attributes:
        X0: ndarray
            (1, 6) array of initial states ordered as (rx, ry, rz, vx, vy, vz).
        mu: float
            Standard Gravitational Parameter
    """

    def __init__(self, X0, mu=1.0):
        self.X0 = X0
        self.mu = mu

    def __call__(self, T):
        """Calculate position-velocity based on the initial state.

        Args:
            T: ndarray
                (m, 1) array of times.

        Returns:
            X: ndarray
                (m, 6) array of states.
        """
        X_coe = coeKepPath(orb.rv2coe(self.X0), mu=self.mu)(T)
        return orb.coe2rv(X_coe)
