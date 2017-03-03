"""Keplerian trajectory for classical orbital elements."""

import numpy as np


__author__ = "Nathan I. Budd"
__email__ = "nibudd@gmail.com"
__copyright__ = "Copyright 2017, LASR Lab"
__license__ = "MIT"
__version__ = "0.1"
__status__ = "Production"
__date__ = "02 Mar 2017"


class KeplerianPath(object):
    """Calculate Keplerian trajectory in classical orbital elements.

    Attributes:
        X0: ndarray
            (1, 6) array of initial states ordered as (a, e, i, W, w, f), where
            a = semi-major axis
            e = eccentricity
            i = inclination
            W = right ascension of the ascending node
            w = argument of perigee
            f = true anomaly
        mu: float
            Standard Gravitational Parameter
    """

    def __init__(self, X0, mu=1.0):
        self.X0 = X0
        self.mu = mu

    def __call__(self, T):
        """Calculate classical orbital elements based on the initial state.

        Args:
            T: ndarray
                (m, 1) array of times.

        Returns:
            X: ndarray
                (m, 6) array of states.
        """
        T0 = T[0, 0]
        a = self.X0[0, 0]
        n = (self.mu / a**3)**.5

        dT = T - T0
        dM = dT * n

        X = np.tile(self.X0, T.shape)
        X[0:, -1:] = X[0:, -1:] + dM

        return X
