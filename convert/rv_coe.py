from orbital_elements.convert.rv_mee import rv_mee
from orbital_elements.convert.mee_coe import mee_coe

__author__ = "Nathan I. Budd"
__email__ = "nibudd@gmail.com"
__copyright__ = "Copyright 2017, LASR Lab"
__license__ = "MIT"
__version__ = "0.1"
__status__ = "Production"
__date__ = "06 Mar 2017"


def rv_coe(coe):
    """Convert classical orbital elements to position-velocity elements.

    Args:
        coe: ndarray
            (m, 6) array of COEs with true anomaly ordered as
            (a, e, i, W, w, nu), where
            a = semi-major axis
            e = eccentricity
            i = inclination
            W = right ascension of the ascending node
            w = argument of perigee
            nu = true anomaly

    Returns:
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
    """
    return rv_mee(mee_coe(coe))
