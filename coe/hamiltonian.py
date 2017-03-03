"""Hamiltonian for claissical orbital elements."""

from orbital_elements.rv.hamiltonian import Hamiltonian as rvHam
import orbital_mechanics.orbit as orb

__author__ = "Nathan I. Budd"
__email__ = "nibudd@gmail.com"
__copyright__ = "Copyright 2017, LASR Lab"
__license__ = "MIT"
__version__ = "0.1"
__status__ = "Production"
__date__ = "02 Mar 2017"


class Hamiltonian(object):
    """Calculate Hamiltonian change using zonal gravity in classical elements.

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
                Columns are ordered as (a, e, i, W, w, f), where
                a = semi-major axis
                e = eccentricity
                i = inclination
                W = right ascension of the ascending node
                w = argument of perigee
                f = true anomaly

        Returns:
            H_rel: ndarray
                (m, 1) array of Hamiltonian over time.
        """
        Hamiltonian = rvHam(mu=self.mu, order=self.order, r_earth=self.r_earth)

        return Hamiltonian(T, orb.coe2rv(X))
