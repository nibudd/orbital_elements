import numpy as np


__author__ = "Nathan I. Budd"
__email__ = "nibudd@gmail.com"
__copyright__ = "Copyright 2017, LASR Lab"
__license__ = "MIT"
__version__ = "0.1"
__status__ = "Production"
__date__ = "27 Mar 2017"


class KeplerianSolution(object):
    """Keplerian solution for MEEs with mean longitude at epoch.

    Attributes:
        X0: ndarray
            (m, 6) array of modified equinoctial elements ordered as
            (p, f, g, h, k, Ml0), where
            p = semi-latus rectum
            f = 1-component of eccentricity vector in perifocal frame
            g = 2-component of eccentricity vector in perifocal frame
            h = 1-component of the ascending node vector in equ. frame
            k = 2-component of the ascending node vector in equ. frame
            Ml0 = mean longitude at epoch
        mu: float, optional
            Standard Gravitational Parameter. Defaults to 1.0, the standard
            value in canonical units.
    """

    def __init__(self, X0, mu=1.0):
        self.X0 = X0
        self.mu = mu

    def __call__(self, T):
        """Calculate modified equinocital elements solution.

        Args:
            T: ndarray
                (m, 1) array of times.

        Returns:
            X: ndarray
                (m, 6) array of states.
        """
        return np.tile(self.X0, T.shape)
