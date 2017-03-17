import numpy as np

__author__ = "Nathan I. Budd"
__email__ = "nibudd@gmail.com"
__copyright__ = "Copyright 2017, LASR Lab"
__license__ = "MIT"
__version__ = "0.1"
__status__ = "Production"
__date__ = "03 Mar 2017"


class KeplerianDynamics(object):
    """Keplerian dynamics for position-velocity elements.

    Attributes:
        mu: float, optional
            Standard Gravitational Parameter. Defaults to 1.0, the standard
            value in canonical units.
    """

    def __init__(self, mu=1.0):
        self.mu = mu

    def __call__(self, T, X):
        """Calculate position-velocity dynamics.

        Args:
            T: ndarray
                (m, 1) array of times.
            X: ndarray
                (m, 6) array of states ordered as (rx, ry, rz, vx, vy, vz),
                where
                rx = position x-component
                ry = position y-component
                rz = position z-component
                vx = velocity x-component
                vy = velocity y-component
                vz = velocity z-component

        Returns:
            Xdot: ndarray
                (m, 6) array of state derivatives.
        """
        R = X[:, 0:3]
        V = X[:, 3:6]
        r = np.linalg.norm(R, axis=1, keepdims=True)

        neg_mu_by_r3 = (-self.mu / r**3)
        Vdot = np.tile(neg_mu_by_r3, (1, 3)) * R

        return np.concatenate((V, Vdot), axis=1)
