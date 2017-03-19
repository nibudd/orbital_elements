import numpy as np

__author__ = "Nathan I. Budd"
__email__ = "nibudd@gmail.com"
__copyright__ = "Copyright 2017, LASR Lab"
__license__ = "MIT"
__version__ = "0.1"
__status__ = "Production"
__date__ = "16 Mar 2017"


class KeplerianDynamics(object):
    """Keplerian dynamics for modified equinoctial elements.

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
                (p, f, g, h, k, L), where
                p = semi-latus rectum
                f = 1-component of eccentricity vector in perifocal frame
                g = 2-component of eccentricity vector in perifocal frame
                h = 1-component of ascending node vector in equinoctial frame
                k = 2-component of ascending node vector in equinoctial frame
                L = true longitude

        Returns:
            Xdot: ndarray
                (m, 6) array of state derivatives.
        """
        p = X[:, 0:1]
        f = X[:, 1:2]
        g = X[:, 2:3]
        L = X[:, 5:6]
        L_dot = (self.mu / p**3)**0.5 * (1 + f*np.cos(L) + g*np.sin(L))**2

        return np.concatenate(
            (np.zeros((T.shape[0], 5)), L_dot),
            axis=1)
