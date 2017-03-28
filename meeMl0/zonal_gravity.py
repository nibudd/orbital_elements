import orbital_elements.convert as convert
from orbital_elements.meeMl0.gve import GVE
from orbital_elements.rv.zonal_gravity import ZonalGravity as rvZonalGravity


__author__ = "Nathan I. Budd"
__email__ = "nibudd@gmail.com"
__copyright__ = "Copyright 2017, LASR Lab"
__license__ = "MIT"
__version__ = "0.1"
__status__ = "Production"
__date__ = "27 Mar 2017"


class ZonalGravity(rvZonalGravity):
    """Zonal gravity dynamics for MEEs with mean longitude at epoch."""

    def __init__(self, mu=1.0, order=2, r_earth=1.0):
        super().__init__(mu=mu, order=order, r_earth=r_earth)

    def __call__(self, T, X):
        """Calculate zonal gravity perturations for MEEs with Ml0.

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
        super().lvlh_acceleration(T, convert.rv_mee(convert.mee_meeMl0(T, X)))
        G = GVE()(T, X)
        m = T.shape[0]

        return (G @ self.a_lvlh.reshape((m, 3, 1))).reshape((m, 6))
