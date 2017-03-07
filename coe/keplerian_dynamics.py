import numpy as np

__author__ = "Nathan I. Budd"
__email__ = "nibudd@gmail.com"
__copyright__ = "Copyright 2017, LASR Lab"
__license__ = "MIT"
__version__ = "0.1"
__status__ = "Production"
__date__ = "04 Mar 2017"


class KeplerianDynamics(object):
    """Keplerian dynamics for classical orbital elements.

    Attributes:
        mu: float, optional
            Standard Gravitational Parameter. Defaults to 1.0, the standard
            value in canonical units.
    """

    def __init__(self, mu=1.0):
        self.mu = mu

    def __call__(self, T, X):
        """Calculate classical orbital elements solution.

        Args:
            T: ndarray
                (m, 1) array of times.
            X: ndarray
                (m, 6) array of states ordered as (a, e, i, W, w, f), where
                a = semi-major axis
                e = eccentricity
                i = inclination
                W = right ascension of the ascending node
                w = argument of perigee
                f = true anomaly

        Returns:
            Xdot: ndarray
                (m, 6) array of state derivatives.
        """
        a = X[:, 0:1]
        e = X[:, 1:2]
        f = X[:, 5:6]
        p = a * (1-e**2)
        r = p / (1. + e * np.cos(f))
        h = (self.mu * p)**0.5
        f_dot = h / r**2

        return np.concatenate(
            (np.zeros((T.shape[0], 5)), f_dot),
            axis=1)
