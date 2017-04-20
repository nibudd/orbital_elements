import numpy as np
from orbital_elements.meeMl0.gve import GVE


__author__ = "Nathan I. Budd"
__email__ = "nibudd@gmail.com"
__copyright__ = "Copyright 2017, LASR Lab"
__license__ = "MIT"
__version__ = "0.1"
__status__ = "Production"
__date__ = "19 Apr 2017"


class ErrorDirectionLSQ(object):
    """Least squares-derived control in the direction of the error vector.

    Given Keplerian dynamics and an LVLH-frame control (Xdot = a_t G(X) u)
    the control is calculated via least squares, where Xdot is meeMl0
    derivatives and is set equal to the current state error, a_t is the maximum
    thrust magnitude, G is Gauss's Variational Equations for meeMl0, X is the
    meeMl0 states, and u is the control with a magnitude no greater than 1.

    Attributes:
        Xf: ndarray
            Takes an (1, 6) array of target meeMl0 values
        a_t: float, opttional
            Maximum thrust magnitude.
        mu: float, optional
            Standard Gravitational Parameter. Defaults to 1.0, the standard
            value in canonical units.
        E: ndarray
            (m, 6) array of state errors.
        u: ndarray
            (m, 3) array of control accelerations in the LVLH frame.
    """

    def __init__(self, Xf, a_t=1.0, mu=1.0):
        self.Xf = Xf
        self.a_t = a_t
        self.mu = mu
        self.E = np.array([])
        self.u = np.array([])

    def __call__(self, T, X):
        """Calculate control.

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
            u: ndarray
                (m, 3) array of control vectors.
        """
        m = T.shape[0]
        G = GVE(mu=self.mu)(T, X)

        GTG_inv = np.array([np.linalg.inv(g.T @ g)
                            for g in G])

        self.E = np.tile(self.Xf, T.shape) - X

        u = (
            GTG_inv @ G.transpose((0, 2, 1)) @ self.E.reshape((m, 6, 1))
            ).reshape((m, 3))

        # u = np.array([u_i / np.linalg.norm(u_i) if np.linalg.norm(u_i) > 1
        #            else u_i for u_i in u])
        u = u / np.tile(np.linalg.norm(u, axis=1, keepdims=True), (1, 3))
        self.u = u

        return self.a_t * (G @ u.reshape((m, 3, 1))).reshape((m, 6))
