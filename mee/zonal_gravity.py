import orbital_elements.convert as convert
from orbital_elements.mee.gve import GVE
from orbital_elements.rv.zonal_gravity import ZonalGravity as rvZonalGravity


__author__ = "Nathan I. Budd"
__email__ = "nibudd@gmail.com"
__copyright__ = "Copyright 2017, LASR Lab"
__license__ = "MIT"
__version__ = "0.1"
__status__ = "Production"
__date__ = "16 Mar 2017"


class ZonalGravity(object):
    """Zonal gravity dynamics for modified equinoctial elements.

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

    def __call__(self, T, X):
        """Calculate zonal gravity perturations in MEEs.

        Args:
            T: ndarray
                (m, 1) array of times.
            X: ndarray
                (m, 6) array of modified equinoctial elements ordered as
                (p, f, g, h, k, L), where
                p = semi-latus rectum
                f = 1-component of eccentricity vector in perifocal frame
                g = 2-component of eccentricity vector in perifocal frame
                h = 1-component of ascending node vector in equinoctial frame
                k = 2-component of ascending node vector in equinoctial frame
                L = true longitude

        Returns:
            Xdot: ndarray
                (m, 6) array of state derivatives.
        """
        rvzg = rvZonalGravity(mu=self.mu, order=self.order,
                              r_earth=self.r_earth)
        a_d = rvzg.lvlh_acceleration(T, convert.rv_mee(X))
        G = GVE()(T, X)
        m = T.shape[0]

        return (G @ a_d.reshape((m, 3, 1))).reshape((m, 6))
