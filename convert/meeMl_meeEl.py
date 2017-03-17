import numpy as np

__author__ = "Nathan I. Budd"
__email__ = "nibudd@gmail.com"
__copyright__ = "Copyright 2017, LASR Lab"
__license__ = "MIT"
__version__ = "0.1"
__status__ = "Production"
__date__ = "08 Mar 2017"


def meeMl_meeEl(meeEl):
    """Convert MEEs with eccentric longitude to mean longitude.

    Args:
        meeEl: ndarray
            (m, 6) array of modified equinoctial elements ordered as
            (p, f, g, h, k, El), where
            p = semi-latus rectum
            f = 1-component of eccentricity vector in perifocal frame
            g = 2-component of eccentricity vector in perifocal frame
            h = 1-component of the ascending node vector in equinoctial frame
            k = 2-component of the ascending node vector in equinoctial frame
            El = eccentric longitude

    Returns:
        meeMl: ndarray
            (m, 6) array of modified equinoctial elements ordered as
            (p, f, g, h, k, Ml), where
            p = semi-latus rectum
            f = 1-component of eccentricity vector in perifocal frame
            g = 2-component of eccentricity vector in perifocal frame
            h = 1-component of the ascending node vector in equinoctial frame
            k = 2-component of the ascending node vector in equinoctial frame
            Ml = mean longitude
    """
    f = meeEl[:, 1:2]
    g = meeEl[:, 2:3]
    El = meeEl[:, 5:6]

    Ml = np.mod(El - f*np.sin(El) + g*np.cos(El), 2*np.pi)

    return np.concatenate((meeEl[:, 0:5], Ml), axis=1)
