import numpy as np

__author__ = "Nathan I. Budd"
__email__ = "nibudd@gmail.com"
__copyright__ = "Copyright 2017, LASR Lab"
__license__ = "MIT"
__version__ = "0.1"
__status__ = "Production"
__date__ = "05 Mar 2017"


def mee_coe(coe):
    """Convert classical orbital elements to modifieid equinoctial elements.

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
        mee: ndarray
            (m, 6) array of position-velocity elements ordered as
            (p, f, g, h, k, L), where
            p = semi-latus rectum
            f = 1-component of eccentricity vector in perifocal frame
            g = 2-component of eccentricity vector in perifocal frame
            h = 1-component of the ascending node vector in equinoctial frame
            k = 2-component of the ascending node vector in equinoctial frame
            L = true longitude
    """
    a = coe[0:, 0:1]
    e = coe[0:, 1:2]
    i = coe[0:, 2:3]
    W = coe[0:, 3:4]
    w = coe[0:, 4:5]
    nu = coe[0:, 5:6]

    # semi-latus rectum
    p = a * (1 - e**2)

    # 1,2-components of eccentricity vector
    f = e * np.cos(w + W)
    g = e * np.sin(w + W)

    # 1,2-components of ascending node vector
    h = np.tan(i/2.) * np.cos(W)
    k = np.tan(i/2.) * np.sin(W)

    # true longitude
    L = np.mod(W+w+nu, 2*np.pi)

    return np.concatenate((p, f, g, h, k, L), axis=1)
