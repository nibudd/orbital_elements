import numpy as np
from orbital_elements.mee.gve import GVE

__author__ = "Nathan I. Budd"
__email__ = "nibudd@gmail.com"
__copyright__ = "Copyright 2017, LASR Lab"
__license__ = "MIT"
__version__ = "0.1"
__status__ = "Production"
__date__ = "19 Mar 2017"


class ConstantThrust(object):
    """Constant LVLH acceleration as constant MEE time derivatives.

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
        """Calculate constant acceleration as MEE time derivatives.

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
        m = T.shape[0]
        u = np.tile(self.u, (m, 1, 1))
        G = GVE()(T, X)

        return (G @ u).reshape((m, 6))
