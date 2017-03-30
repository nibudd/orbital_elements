from .meeEl_meefl import meeEl_meefl
from .meeMl_meeEl import meeMl_meeEl

__author__ = "Nathan I. Budd"
__email__ = "nibudd@gmail.com"
__copyright__ = "Copyright 2017, LASR Lab"
__license__ = "MIT"
__version__ = "0.1"
__status__ = "Production"
__date__ = "08 Mar 2017"


def meeMl_meefl(meefl):
    """Convert MEEs with true longitude to mean longitude.

    Args:
        meefl: ndarray
            (m, 6) array of modified equinoctial elements ordered as
            (p, f, g, h, k, fl), where
            p = semi-latus rectum
            f = 1-component of eccentricity vector in perifocal frame
            g = 2-component of eccentricity vector in perifocal frame
            h = 1-component of the ascending node vector in equinoctial frame
            k = 2-component of the ascending node vector in equinoctial frame
            fl = true longitude

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
    return meeMl_meeEl(meeEl_meefl(meefl))
