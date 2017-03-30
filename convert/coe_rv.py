from .mee_rv import mee_rv
from .coe_mee import coe_mee

__author__ = "Nathan I. Budd"
__email__ = "nibudd@gmail.com"
__copyright__ = "Copyright 2017, LASR Lab"
__license__ = "MIT"
__version__ = "0.1"
__status__ = "Production"
__date__ = "06 Mar 2017"


def coe_rv(rv):
    """Convert position-velocity elements to classical orbital elements.

    Args:
        rv: ndarray
            (m, 6) array of COEs with true anomaly ordered as
            (rx, ry, rz, vx, vy, vz),
            where
            rx = position x-component
            ry = position y-component
            rz = position z-component
            vx = velocity x-component
            vy = velocity y-component
            vz = velocity z-component

    Returns:
        coe: ndarray
            (m, 6) array of COEs with true anomaly ordered as
            (a, e, i, W, w, nu), where
            a = semi-major axis
            e = eccentricity
            i = inclination
            W = right ascension of the ascending node
            w = argument of perigee
            nu = true anomaly
    """
    return coe_mee(mee_rv(rv))
