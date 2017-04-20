import numpy as np
from orbital_elements.meeMl0.gve import GVE


__author__ = "Nathan I. Budd"
__email__ = "nibudd@gmail.com"
__copyright__ = "Copyright 2017, LASR Lab"
__license__ = "MIT"
__version__ = "0.1"
__status__ = "Production"
__date__ = "14 Apr 2017"


class LeastSquaresControl(object):
    """Least squares control given a prescribed trajectory.

    Given Keplerian dynamics and an LVLH-frame control (Xdot = a_t G(X) u)
    the control is calculated in via least squares, where Xdot is meeMl0
    derivatives, a_t is the maximum thrust magnitude, G is Gauss's Variational
    Equations for meeMl0, X is the meeMl0 states, and u is the control with a
    magnitude no greater than 1.

    Attributes:
        X: callable
            Takes an (m, 1) array of times and returns an (m, n) array of
            states.
        Xdot: callable
            Takes an (m, 1) array of times and returns an (m, n) array of
            state derivatives.
        a_t: float, opttional
            Maximum thrust magnitude.
        mu: float, optional
            Standard Gravitational Parameter. Defaults to 1.0, the standard
            value in canonical units.
    """

    def __init__(self, X, Xdot, a_t=1.0, mu=1.0):
        self.X = X
        self.Xdot = Xdot
        self.a_t = a_t
        self.mu = mu

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
        G = GVE(mu=self.mu)(T, self.X(T))

        GTG_inv = np.array([np.linalg.inv(g.T @ g)
                            for g in G])

        return (
            GTG_inv @ G.transpose((0, 2, 1)) @
            self.Xdot(T).reshape((m, 6, 1))
            ).reshape((m, 3))
