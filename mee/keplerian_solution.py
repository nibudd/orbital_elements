import numpy as np


__author__ = "Nathan I. Budd"
__email__ = "nibudd@gmail.com"
__copyright__ = "Copyright 2017, LASR Lab"
__license__ = "MIT"
__version__ = "0.1"
__status__ = "Production"
__date__ = "06 Mar 2017"


class KeplerianSolution(object):
    """Keplerian solution for modified equinoctial elements.

    Attributes:
        X0: ndarray
            (m, 6) array of position-velocity elements ordered as
            (p, f, g, h, k, L), where
            p = semi-latus rectum
            f = 1-component of eccentricity vector in perifocal frame
            g = 2-component of eccentricity vector in perifocal frame
            h = 1-component of the ascending node vector in equinoctial frame
            k = 2-component of the ascending node vector in equinoctial frame
            L = true longitude
        mu: float, optional
            Standard Gravitational Parameter. Defaults to 1.0, the standard
            value in canonical units.
    """

    def __init__(self, X0, mu=1.0):
        self.X0 = X0
        self.mu = mu

    def __call__(self, T):
        """Calculate classical orbital elements solution.

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
