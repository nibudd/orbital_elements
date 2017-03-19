import numpy as np
from orbital_elements.coe.gve import GVE

__author__ = "Nathan I. Budd"
__email__ = "nibudd@gmail.com"
__copyright__ = "Copyright 2017, LASR Lab"
__license__ = "MIT"
__version__ = "0.1"
__status__ = "Production"
__date__ = "18 Mar 2017"


class ConstantThrust(object):
    """Constant LVLH acceleration as COE time derivatives.

    Attributes:
        u: ndarray
            3-element array representing the LVLH acceleration in the radial,
            theta, and normal directions.
        mu: float, optional
            Standard Gravitational Parameter. Defaults to 1.0, the standard
            value in canonical units.
    """

    def __init__(self, u, mu=1.0):
        self.u = u.reshape((1, 3, 1))
        self.mu = mu

    def __call__(self, T, X):
        """Calculate constant acceleration as COE time derivatives.

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
        m = T.shape[0]
        u = np.tile(self.u, (m, 1, 1))
        G = GVE()(T, X)

        return (G @ u).reshape((m, 6))
