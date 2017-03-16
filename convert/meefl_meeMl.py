from orbital_elements.convert.meeEl_meeMl import meeEl_meeMl
from orbital_elements.convert.meefl_meeEl import meefl_meeEl

__author__ = "Nathan I. Budd"
__email__ = "nibudd@gmail.com"
__copyright__ = "Copyright 2017, LASR Lab"
__license__ = "MIT"
__version__ = "0.1"
__status__ = "Production"
__date__ = "08 Mar 2017"


def meefl_meeMl(meeMl):
    """Convert MEEs with true longitude to mean longitude.

    Args:
        meeMl: ndarray
            (m, 6) array of position-velocity elements ordered as
            (p, f, g, h, k, Ml), where
            p = semi-latus rectum
            f = 1-component of eccentricity vector in perifocal frame
            g = 2-component of eccentricity vector in perifocal frame
            h = 1-component of the ascending node vector in equinoctial frame
            k = 2-component of the ascending node vector in equinoctial frame
            Ml = mean longitude

    Returns:
        meefl: ndarray
            (m, 6) array of position-velocity elements ordered as
            (p, f, g, h, k, fl), where
            p = semi-latus rectum
            f = 1-component of eccentricity vector in perifocal frame
            g = 2-component of eccentricity vector in perifocal frame
            h = 1-component of the ascending node vector in equinoctial frame
            k = 2-component of the ascending node vector in equinoctial frame
            fl = true longitude
    """
    return meefl_meeEl(meeEl_meeMl(meeMl))
