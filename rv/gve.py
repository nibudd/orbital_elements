import numpy as np

__author__ = "Nathan I. Budd"
__email__ = "nibudd@gmail.com"
__copyright__ = "Copyright 2017, LASR Lab"
__license__ = "MIT"
__version__ = "0.1"
__status__ = "Production"
__date__ = "03 Mar 2017"


class GVE(object):
    """Gauss's Variational Equations equivalent for position-velocity elements.

    Attributes:
        mu: float, optional
            Standard Gravitational Parameter. Defaults to 1.0, the standard
            value in canonical units.
    """

    def __init__(self, mu=1.0):
        self.mu = mu

    def __call__(self, T, X):
        """Calculate GVE matrices for position-velocity elements.

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
            G: ndarray
                (m, 6, 3) array of GVE matrices.
        """
        m = T.shape[0]
        dims = (m, 3, 1)

        r = X[:, 0:3]
        v = X[:, 3:6]
        h = np.cross(r, v)

        i_r = r / np.linalg.norm(r)
        i_h = h / np.linalg.norm(h)
        i_t = np.cross(i_h, i_r).reshape(dims)
        i_r = i_r.reshape(dims)
        i_h = i_h.reshape(dims)

        # G's bottom rotates acceleration vectors from LVLH to ECI frame
        G_bottom = np.concatenate((i_r, i_t, i_h), axis=2)

        return np.concatenate((np.zeros((m, 3, 3)), G_bottom), axis=1)
