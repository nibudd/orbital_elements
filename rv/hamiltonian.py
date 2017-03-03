"""Hamiltonian for position-velocity elements."""

import numpy as np

__author__ = "Nathan I. Budd"
__email__ = "nibudd@gmail.com"
__copyright__ = "Copyright 2017, LASR Lab"
__license__ = "MIT"
__version__ = "0.1"
__status__ = "Production"
__date__ = "02 Mar 2017"


class Hamiltonian(object):
    """Calculate Hamiltonian change using zonal gravity in RV elements.

    Attributes:
        mu: float
            Standard Gravitational Parameter
        order: int
            Zonal gravity order. Order of 1 corresponds to two body dynamics.
            Higher orders include preturbations from J2 up to J6.
        r_earth: float
            Equatorial radius of Earth.
    """

    def __init__(self, mu=1.0, order=1, r_earth=1.0):
        self.mu = mu
        self.order = order
        self.r_earth = r_earth

    def __call__(self, T, X):
        """Calculate Hamiltonian.

        Args:
            T: ndarray
                (m, 1) array of times.
            X: ndarray
                (m, 6) array of states.
                Columns are ordered as (rx, ry, rz, vx, vy, vz).

        Returns:
            H_rel: ndarray
                (m, 1) array of Hamiltonian over time.
        """
        z = X[:, 2:3]
        r = np.linalg.norm(X[0:, 0:3], ord=2, axis=1).reshape(z.shape)
        v = np.linalg.norm(X[0:, 3:6], ord=2, axis=1).reshape(z.shape)
        sin_phi = z/r

        J2_to_6 = [1082.63e-6, -2.52e-6, -1.61e-6, -.15e-6, .57e-6]
        J = J2_to_6[0:self.order-1]

        # calculate and accumulate potential function terms for each J term
        V = -self.mu/r

        try:
            V -= (-J[0]/2. * self.mu/r * (self.r_earth/r)**2 *
                  (3.*sin_phi**2 - 1.))

            V -= (-J[1]/2. * self.mu/r * (self.r_earth/r)**3 *
                  (5.*sin_phi**3 - 3.*sin_phi))

            V -= (-J[2]/8. * self.mu/r * (self.r_earth/r)**4 *
                  (35.*sin_phi**4 - 30.*sin_phi**2 + 3.))

            V -= (-J[3]/8. * self.mu/r * (self.r_earth/r)**5 *
                  (63.*sin_phi**5 - 70.*sin_phi**3 + 15.*sin_phi))

            V -= (-J[4]/16. * self.mu/r * (self.r_earth/r)**6 *
                  (231.*sin_phi**6 - 315.*sin_phi**4 + 105.*sin_phi**2 - 5.))
        except IndexError:
            pass

        return .5 * v**2 + V
