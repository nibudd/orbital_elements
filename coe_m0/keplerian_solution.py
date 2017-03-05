import numpy as np


__author__ = "Nathan I. Budd"
__email__ = "nibudd@gmail.com"
__copyright__ = "Copyright 2017, LASR Lab"
__license__ = "MIT"
__version__ = "0.1"
__status__ = "Production"
__date__ = "04 Mar 2017"


class KeplerianSolution(object):
    """Keplerian solution for classical orbital elements w/ epoch mean anomaly.

    Attributes:
        X0: ndarray
            (1, 6) array of initial states ordered as (a, e, i, W, w, M0),
            where
            a = semi-major axis
            e = eccentricity
            i = inclination
            W = right ascension of the ascending node
            w = argument of perigee
            M0 = epoch mean anomaly
        mu: float, optional
            Standard Gravitational Parameter. Defaults to 1.0, the standard
            value in canonical units.
    """

    def __init__(self, X0, mu=1.0):
        self.X0 = X0
        self.mu = mu

    def __call__(self, T):
        """Calculate classical orbital elements w/ epoch mean anomaly solution.

        Args:
            T: ndarray
                (m, 1) array of times.

        Returns:
            X: ndarray
                (m, 6) array of states.
        """
        return np.tile(self.X0, T.shape)
