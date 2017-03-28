from orbital_elements.rv.hamiltonian import Hamiltonian as rvHam
import orbital_elements.convert as convert

__author__ = "Nathan I. Budd"
__email__ = "nibudd@gmail.com"
__copyright__ = "Copyright 2017, LASR Lab"
__license__ = "MIT"
__version__ = "0.1"
__status__ = "Production"
__date__ = "27 Mar 2017"


class Hamiltonian(object):
    """Hamiltonian for MEEs with mean longitude at epoch.

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
                (m, 6) array of modified equinoctial elements ordered as
                (p, f, g, h, k, Ml0), where
                p = semi-latus rectum
                f = 1-component of eccentricity vector in perifocal frame
                g = 2-component of eccentricity vector in perifocal frame
                h = 1-component of the ascending node vector in equ. frame
                k = 2-component of the ascending node vector in equ. frame
                Ml0 = mean longitude at epoch

        Returns:
            H_rel: ndarray
                (m, 1) array of Hamiltonian over time.
        """
        Hamiltonian = rvHam(mu=self.mu, order=self.order, r_earth=self.r_earth)

        return Hamiltonian(
            T, convert.rv_mee(
                convert.mee_meeMl0(T, X, mu=self.mu)
                )
            )
