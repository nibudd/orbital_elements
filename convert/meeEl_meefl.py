import numpy as np

__author__ = "Nathan I. Budd"
__email__ = "nibudd@gmail.com"
__copyright__ = "Copyright 2017, LASR Lab"
__license__ = "MIT"
__version__ = "0.1"
__status__ = "Production"
__date__ = "08 Mar 2017"


def meeEl_meefl(meefl):
    """Convert MEEs with true longitude to eccentric longitude.

    Args:
        meefl: ndarray
            (m, 6) array of position-velocity elements ordered as
            (p, f, g, h, k, fl), where
            p = semi-latus rectum
            f = 1-component of eccentricity vector in perifocal frame
            g = 2-component of eccentricity vector in perifocal frame
            h = 1-component of the ascending node vector in equinoctial frame
            k = 2-component of the ascending node vector in equinoctial frame
            fl = true longitude

    Returns:
        meeEl: ndarray
            (m, 6) array of position-velocity elements ordered as
            (p, f, g, h, k, El), where
            p = semi-latus rectum
            f = 1-component of eccentricity vector in perifocal frame
            g = 2-component of eccentricity vector in perifocal frame
            h = 1-component of the ascending node vector in equinoctial frame
            k = 2-component of the ascending node vector in equinoctial frame
            El = eccentric longitude
    """
    f = meefl[:, 1:2]
    g = meefl[:, 2:3]
    fl = meefl[:, 5:6]

    e = (f**2 + g**2)**.5
    B = ((1 + e) / (1 - e))**.5
    tan_wbar_by_2 = ((e - f) / (e + f))**0.5
    tan_fl_by_2 = np.tan(fl/2)
    tan_E_by_2 = 1/B * ((tan_fl_by_2 - tan_wbar_by_2) /
                        (1 + tan_fl_by_2 * tan_wbar_by_2))
    tan_El_by_2 = ((tan_E_by_2 + tan_wbar_by_2) /
                   (1 - tan_E_by_2 * tan_wbar_by_2))
    El = np.mod((2*np.arctan(tan_El_by_2)), 2*np.pi)

    return np.concatenate((meefl[:, 0:5], El), axis=1)
