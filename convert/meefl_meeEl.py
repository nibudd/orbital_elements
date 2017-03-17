import numpy as np

__author__ = "Nathan I. Budd"
__email__ = "nibudd@gmail.com"
__copyright__ = "Copyright 2017, LASR Lab"
__license__ = "MIT"
__version__ = "0.1"
__status__ = "Production"
__date__ = "08 Mar 2017"


def meefl_meeEl(meeEl):
    """Convert MEEs with eccentric longitude to true longitude.

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
        meefl: ndarray
            (m, 6) array of modified equinoctial elements ordered as
            (p, f, g, h, k, fl), where
            p = semi-latus rectum
            f = 1-component of eccentricity vector in perifocal frame
            g = 2-component of eccentricity vector in perifocal frame
            h = 1-component of the ascending node vector in equinoctial frame
            k = 2-component of the ascending node vector in equinoctial frame
            fl = true longitude
    """
    f = meeEl[:, 1:2]
    g = meeEl[:, 2:3]
    El = meeEl[:, 5:6]

    e = (f**2 + g**2)**0.5
    B = ((1 + e) / (1 - e))**0.5
    tan_wbar_by_2 = ((e - f) / (e + f))**0.5
    tan_El_by_2 = np.tan(El/2)
    tan_f_by_2 = B * ((tan_El_by_2 - tan_wbar_by_2) /
                      (1 + tan_El_by_2 * tan_wbar_by_2))
    tan_fl_by_2 = ((tan_f_by_2 + tan_wbar_by_2) /
                   (1 - tan_f_by_2 * tan_wbar_by_2))
    fl = np.mod(2*np.arctan(tan_fl_by_2), 2*np.pi)

    return np.concatenate((meeEl[:, 0:5], fl), axis=1)
