import orbital_elements.convert as convert
from orbital_elements.mee.gve import GVE
from orbital_elements.rv.zonal_gravity import ZonalGravity as rvZonalGravity


__author__ = "Nathan I. Budd"
__email__ = "nibudd@gmail.com"
__copyright__ = "Copyright 2017, LASR Lab"
__license__ = "MIT"
__version__ = "0.1"
__status__ = "Production"
__date__ = "04 Mar 2017"


class ZonalGravity(rvZonalGravity):
    """Zonal gravity dynamics for modified equinoctial elements."""

    def __init__(self, mu=1.0, order=2, r_earth=1.0):
        super().__init__(mu=mu, order=order, r_earth=r_earth)

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
        super().lvlh_acceleration(T, convert.rv_mee(X))
        G = GVE()(T, X)
        m = T.shape[0]

        return (G @ self.a_lvlh.reshape((m, 3, 1))).reshape((m, 6))
