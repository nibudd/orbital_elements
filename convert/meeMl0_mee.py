import numpy as np
from orbital_elements.convert.meeMl_meefl import meeMl_meefl

__author__ = "Nathan I. Budd"
__email__ = "nibudd@gmail.com"
__copyright__ = "Copyright 2017, LASR Lab"
__license__ = "MIT"
__version__ = "0.1"
__status__ = "Production"
__date__ = "27 Mar 2017"


def meeMl0_mee(T, mee, mu=1.0):
    """Convert MEEs to MEEs with mean longitude at epoch.

    Args:
        T: ndarray
            (m, 1) array of times.
        mee: ndarray
            (m, 6) array of modified equinoctial elements ordered as
            (p, f, g, h, k, L), where
            p = semi-latus rectum
            f = 1-component of eccentricity vector in perifocal frame
            g = 2-component of eccentricity vector in perifocal frame
            h = 1-component of the ascending node vector in equinoctial frame
            k = 2-component of the ascending node vector in equinoctial frame
            L = true longitude
        mu: float, optional
            Standard Gravitational Parameter. Defaults to 1.0, the standard
            value in canonical units.

    Returns:
        meeMl0: ndarray
            (m, 6) array of modified equinoctial elements ordered as
            (p, f, g, h, k, Ml0), where
            p = semi-latus rectum
            f = 1-component of ecentricity vector in perifocal frame
            g = 2-component of eccentricity vector in perifocal frame
            h = 1-component of the ascending node vector in equinoctial frame
            k = 2-component of the ascending node vector in equinoctial frame
            Ml0 = mean longitude at epoch
    """
    meeMl = meeMl_meefl(mee)
    p = meeMl[:, 0:1]
    f = meeMl[:, 1:2]
    g = meeMl[:, 2:3]
    Ml = meeMl[:, 5:6]

    a = p / (1 - f**2 - g**2)
    n = (mu / a**3)**0.5
    Ml0 = Ml - n*T

    return np.concatenate((mee[:, 0:5], Ml0), axis=1)
