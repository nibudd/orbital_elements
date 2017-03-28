import numpy as np

__author__ = "Nathan I. Budd"
__email__ = "nibudd@gmail.com"
__copyright__ = "Copyright 2017, LASR Lab"
__license__ = "MIT"
__version__ = "0.1"
__status__ = "Production"
__date__ = "27 Mar 2017"


class KeplerianDynamics(object):
    """Keplerian dynamics for MEEs with mean longitude at epoch.

    Attributes:
        mu: float, optional
            Standard Gravitational Parameter. Defaults to 1.0, the standard
            value in canonical units.
    """

    def __init__(self, mu=1.0):
        self.mu = mu

    def __call__(self, T, X):
        """Calculate modified equinoctial elements dynamics.

        Args:
            T: ndarray
                (m, 1) array of times.
            X: ndarray
                (m, 6) array of modified equinoctial elements ordered as
                (p, f, g, h, k, Ml0), where
                p = semi-latus rectum
                f = 1-component of eccentricity vector in perifocal frame
                g = 2-component of eccentricity vector in perifocal frame
                h = 1-component of the ascending node vector in equ. frame
                k = 2-component of the ascending node vector in equ. frame
                Ml0 = mean longitude at epoch

        Returns:
            Xdot: ndarray
                (m, 6) array of state derivatives.
        """
        return np.zeros(X.shape)
