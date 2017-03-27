import numpy as np
from orbital_elements.convert.meefl_meeMl import meefl_meeMl

__author__ = "Nathan I. Budd"
__email__ = "nibudd@gmail.com"
__copyright__ = "Copyright 2017, LASR Lab"
__license__ = "MIT"
__version__ = "0.1"
__status__ = "Production"
__date__ = "27 Mar 2017"


def mee_meeMl0(T, meeMl0, mu=1.0):
    """Convert MEEs with mean longitude at epoch to MEEs.

    Args:
        T: ndarray
            (m, 1) array of times.
        meeMl0: ndarray
            (m, 6) array of modified equinoctial elements ordered as
            (p, f, g, h, k, Ml0), where
            p = semi-latus rectum
            f = 1-component of ecentricity vector in perifocal frame
            g = 2-component of eccentricity vector in perifocal frame
            h = 1-component of the ascending node vector in equinoctial frame
            k = 2-component of the ascending node vector in equinoctial frame
            Ml0 = mean longitude at epoch
        mu: float, optional
            Standard Gravitational Parameter. Defaults to 1.0, the standard
            value in canonical units.

    Returns:
        mee: ndarray
            (m, 6) array of modified equinoctial elements ordered as
            (p, f, g, h, k, L), where
            p = semi-latus rectum
            f = 1-component of eccentricity vector in perifocal frame
            g = 2-component of eccentricity vector in perifocal frame
            h = 1-component of the ascending node vector in equinoctial frame
            k = 2-component of the ascending node vector in equinoctial frame
            L = true longitude
    """
    p = meeMl0[:, 0:1]
    f = meeMl0[:, 1:2]
    g = meeMl0[:, 2:3]
    Ml0 = meeMl0[:, 5:6]

    a = p / (1 - f**2 - g**2)
    n = (mu / a**3)**0.5
    Ml = Ml0 + n*T

    return meefl_meeMl(np.concatenate((meeMl0[:, 0:5], Ml), axis=1))
