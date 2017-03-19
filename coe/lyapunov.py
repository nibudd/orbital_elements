import numpy as np
from orbital_elements.coe.gve import GVE
from orbital_elements.coe.keplerian_dynamics import KeplerianDynamics
from orbital_elements.convert.mod_angles import mod_angles

__author__ = "Nathan I. Budd"
__email__ = "nibudd@gmail.com"
__copyright__ = "Copyright 2017, LASR Lab"
__license__ = "MIT"
__version__ = "0.1"
__status__ = "Production"
__date__ = "19 Mar 2017"


class Lyapunov(object):
    """Lyapunov control in classical orbital elements.

    Attributes:
        Y_func: callable
            The reference trajectory being tracked. Returns an (m, 6) array of
            states given an (m, 1) array of times.
        W: ndarray
            (n, n) weight matrix, where n is the state dimension.
        a_t: float
            Maximum thrust magnitude.
        mu: float, optional
            Standard Gravitational Parameter. Defaults to 1.0, the standard
            value in canonical units.
        E: ndarray
            (m, 6) array of state errors.
        u = ndarray
            (m, 3) array of control accelerations in the LVLH frame.
        V: ndarray
            (m, 1) array of Lyapunov function values.
        dVdt: ndarray
            (m, 1) array of Lyapunov function time derivative values.

    """

    def __init__(self, Y_func, W, a_t, mu=1.0):
        self.Y_func = Y_func
        self.W = W
        self.a_t = a_t
        self.mu = mu
        self.E = np.array([])
        self.u = np.array([])
        self.V = np.array([])
        self.dVdt = np.array([])

    def __call__(self, T, X):
        """Calculate control.

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
        G = GVE()(T, X)

        # state errors
        Y = self.Y_func(T)
        self.E = mod_angles(X - Y, angle_indices=[2, 3, 4, 5])

        # control
        EW = (self.E @ self.W).reshape((m, 1, 6))
        c = EW @ G
        self.u = np.array(
            [-x/np.linalg.norm(x, axis=1) if np.linalg.norm(x, axis=1) > 1
             else -x for x in c])

        # lyapunov function and derivative
        self.V = (
            self.E.reshape((m, 1, 6)) @
            np.tile(self.W, (m, 1, 1)) @
            self.E.reshape((m, 6, 1))
            ).reshape((m, 1))
        kepdyn = KeplerianDynamics()
        a_d = self.a_t * G @ self.u.reshape((m, 3, 1))
        dXdt = kepdyn(T, X) + a_d.reshape((m, 6))
        dYdt = kepdyn(T, Y)
        dEdt = dXdt - dYdt
        self.dVdt = (
            self.E.reshape((m, 1, 6)) @
            np.tile(self.W, (m, 1, 1)) @
            dEdt.reshape((m, 6, 1))
            ).reshape((m, 1))

        self.u.shape = (m, 3)
        return self.a_t * (G @ self.u.reshape((m, 3, 1))).reshape((m, 6))
