import numpy as np

__author__ = "Nathan I. Budd"
__email__ = "nibudd@gmail.com"
__copyright__ = "Copyright 2017, LASR Lab"
__license__ = "MIT"
__version__ = "0.1"
__status__ = "Production"
__date__ = "05 Mar 2017"


def coe_mee(mee):
    """Convert modifieid equinoctial elements to classical orbital elements.

    Args:
        mee: ndarray
            (m, 6) array of position-velocity elements ordered as
            (p, f, g, h, k, L), where
            p = semi-latus rectum
            f = 1-component of eccentricity vector in perifocal frame
            g = 2-component of eccentricity vector in perifocal frame
            h = 1-component of the ascending node vector in equinoctial frame
            k = 2-component of the ascending node vector in equinoctial frame
            L = true longitude

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
    p = mee[0:, 0:1]
    f = mee[0:, 1:2]
    g = mee[0:, 2:3]
    h = mee[0:, 3:4]
    k = mee[0:, 4:5]
    L = mee[0:, 5:6]

    # inclination
    i = np.mod(2. * np.arctan((h**2 + k**2)**.5), 2*np.pi)

    # right ascension of the ascending node
    W = np.mod(np.arctan2(k, h), 2*np.pi)

    # eccentricity
    e = (f**2 + g**2)**.5

    # semi-major axis
    a = p / (1 - e**2)

    # argument of periapsis
    w_bar = np.mod(np.arctan2(g, f), 2*np.pi)
    w = np.mod(w_bar - W, 2*np.pi)

    # true anomaly
    f = np.mod(L - w_bar, 2*np.pi)

    return np.concatenate((a, e, i, W, w, f), axis=1)
