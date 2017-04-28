import numpy as np
from orbital_elements.meeMl0.gve import GVE
from orbital_elements.meeMl0.keplerian_dynamics import KeplerianDynamics
from orbital_elements.convert.mod_angles import mod_angles

__author__ = "Nathan I. Budd"
__email__ = "nibudd@gmail.com"
__copyright__ = "Copyright 2017, LASR Lab"
__license__ = "MIT"
__version__ = "0.1"
__status__ = "Production"
__date__ = "21 Apr 2017"


class LyapunovTracking(object):
    """Lyapunov tracking control for MEEs with mean longitude at epoch.

    Whereas the Lyapunov control tracks a final state, this control tracks a
    reference trajectory by determining the additional control effort required
    to make the state follow the reference trajectory.

    Attributes:
        Y_func: callable
            The reference trajectory being tracked. Returns an (m, 6) array of
            states given an (m, 1) array of times.
        U_func: callable
            The control that achieves thee reference trajectory. Returns an
            (m, 3) array of controls given an (m, 1) array of times.
        W: ndarray
            (n, n) weight matrix, where n is the state dimension.
        a_t: float
            Maximum thrust magnitude.
        mu: float, optional
            Standard Gravitational Parameter. Defaults to 1.0, the standard
            value in canonical units.
        E: ndarray
            (m, 6) array of state errors.
        u: ndarray
            (m, 3) array of control accelerations in the LVLH frame.
        du: ndarray
            (m, 3) array of delta control accelerations in the LVLH frame.
        V: ndarray
            (m, 1) array of Lyapunov function values.
        dVdt: ndarray
            (m, 1) array of Lyapunov function time derivative values.
        lower_bound: float, optional
            Specify a lower bound for the control magnitude.
        steering: boolean, optional
            Produces a control vector of constant magnitude (a_t) when True.
            When False, the control vector ranges from 0 to a_t.
        sliding_mag: boolean, optional
            When true, multiplies the control vector by the ratio of the
            current error norm to the inital error norm.

    """

    def __init__(self, Y_func, U_func, W, a_t, mu=1.0, lower_bound=0.0,
                 steering=False, sliding_mag=False):
        self.Y_func = Y_func
        self.U_func = U_func
        self.W = W
        self.a_t = a_t
        self.mu = mu
        self.E = np.array([])
        self.u = np.array([])
        self.du = np.array([])
        self.V = np.array([])
        self.dVdt = np.array([])
        self.lower_bound = lower_bound
        self.steering = steering
        self.sliding_mag = sliding_mag

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
            Xdot: ndarray
                (m, 6) array of state derivatives.
        """
        # state errors
        m = T.shape[0]
        Y = self.Y_func(T)
        self.E = mod_angles(X - Y)

        # delta control
        G_r = GVE(mu=self.mu)(T, Y)
        self.du = - (
            self.a_t * G_r.T @ self.E.reshape((m, 6, 1))
            ).reshape((m, 3))
        self.u = self.U_func(T) + self.du

        # lyapunov function and derivative
        self.V = (
            self.E.reshape((m, 1, 6)) @
            np.tile(self.W, (m, 1, 1)) @
            self.E.reshape((m, 6, 1))
            ).reshape((m, 1))

        self.dVdt = (
            self.E.reshape((m, 1, 6)) @
            np.tile(self.W, (m, 1, 1)) @
            G_r @
            self.du.reshape((m, 3, 1))
            ).reshape((m, 1))

        self.u.shape = (m, 3)
        return self.a_t * (G_r @ self.u.reshape((m, 3, 1))).reshape((m, 6))
