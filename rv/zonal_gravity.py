import numpy as np
from orbital_elements.rv.gve import GVE

__author__ = "Nathan I. Budd"
__email__ = "nibudd@gmail.com"
__copyright__ = "Copyright 2017, LASR Lab"
__license__ = "MIT"
__version__ = "0.1"
__status__ = "Production"
__date__ = "03 Mar 2017"


class ZonalGravity(object):
    """Zonal gravity dynamics for position-velocity elements (in ECI frame).

    Attributes:
        mu: float, optional
            Standard Gravitational Parameter. Defaults to 1.0, the standard
            value in canonical units.
        order: int, optional
            Zonal gravity order. Order of 1 corresponds to two body dynamics.
            Higher orders include preturbations from J2 up to J6. Defaults to
            2, corresponding to the commonly used J2 perturbations.
        r_earth: float, optional
            Equatorial radius of Earth. Defaults to 1.0, Earth's radius in
            canonical units.
    """

    def __init__(self, mu=1.0, order=2, r_earth=1.0):
        self.mu = mu
        self.order = order
        self.r_earth = r_earth

    def lvlh_acceleration(self, T, X):
        """Calculate accelerations due to zonal gravity in LVLH frame.

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
            a_d: ndarray
                (m, 3) array of accelerations in the LVLH frame ordered as
                (a_r, a_t, a_h), where
                a_r = acceleration radial direction
                a_t = acceleration theta direction
                a_h = acceleration out-of-plane direction
        """
        m = X.shape[0]
        r = np.linalg.norm(X[0:, 0:3], ord=2, axis=1).reshape((m, 1))
        xbr = X[:, 0:1] / r
        ybr = X[:, 1:2] / r
        zbr = X[:, 2:3] / r

        J2_to_6 = [1082.63e-6, -2.52e-6, -1.61e-6, -.15e-6, .57e-6]
        J = J2_to_6[0:self.order-1]

        # calculate and accumulate acceleration terms for each J term
        a_d = np.zeros((m, 3))
        try:
            # J2
            a_d += (
                (-3./2. * J[0] * (self.mu/r**2) * (self.r_earth/r)**2) *
                np.concatenate((
                    (1. - 5.*zbr**2) * xbr,
                    (1. - 5.*zbr**2) * ybr,
                    (3. - 5.*zbr**2) * zbr
                ), axis=1)
            )

            # J3
            a_d += (
                (1./2. * J[1] * (self.mu/r**2) * (self.r_earth/r)**3) *
                np.concatenate((
                    5.*(7.*zbr**3 - 3.*zbr) * xbr,
                    5.*(7.*zbr**3 - 3.*zbr) * ybr,
                    3.*(1. - 10.*zbr**2 + 35./3.*zbr**4)
                ), axis=1)
            )

            # J4
            a_d += (
                (5./8. * J[2] * (self.mu/r**2) * (self.r_earth/r)**4) *
                np.concatenate((
                    (3. - 42.*zbr**2 + 63.*zbr**4) * xbr,
                    (3. - 42.*zbr**2 + 63.*zbr**4) * ybr,
                    (15. - 70.*zbr**2 + 63.*zbr**4) * zbr
                ), axis=1)
            )

            # J5
            a_d += (
                (1./8. * J[3] * (self.mu/r**2) * (self.r_earth/r)**5) *
                np.concatenate((
                    3.*(35.*zbr - 210.*zbr**3 + 231.*zbr**5) * xbr,
                    3.*(35.*zbr - 210.*zbr**3 + 231.*zbr**5) * ybr,
                    (-15. + 315.*zbr**2 - 945.*zbr**4 + 693.*zbr**6)
                ), axis=1)
            )

            # J6
            a_d += (
                (-1./16. * J[4] * (self.mu/r**2) * (self.r_earth/r)**6) *
                np.concatenate((
                    (35. - 945.*zbr**2 + 3465.*zbr**4 - 3003.*zbr**6) * xbr,
                    (35. - 945.*zbr**2 + 3465.*zbr**4 - 3003.*zbr**6) * ybr,
                    (245. - 2205.*zbr**2 + 4851.*zbr**4 - 3003.*zbr**6) * zbr
                ), axis=1)
            )
        except IndexError:
            pass

        return a_d

    def __call__(self, T, X):
        """Calculate zonal gravity perturations in position-velocity elements.

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
        G = GVE()(T, X)
        a_d = self.lvlh_acceleration(T, X)
        m = T.shape[0]

        return (G @ a_d.reshape((m, 3, 1))).reshape((m, 6))
