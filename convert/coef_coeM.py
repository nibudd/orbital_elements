import numpy as np
from .coeE_coeM import coeE_coeM
from .coef_coeE import coef_coeE

__author__ = "Nathan I. Budd"
__email__ = "nibudd@gmail.com"
__copyright__ = "Copyright 2017, LASR Lab"
__license__ = "MIT"
__version__ = "0.1"
__status__ = "Production"
__date__ = "05 Mar 2017"


def coef_coeM(coeM):
    """Convert COEs with mean anomaly to true anomaly.

    Args:
        coeM: ndarray
            (m, 6) array of COEs with eccentric anomaly ordered as
            (a, e, i, W, w, f), where
            a = semi-major axis
            e = eccentricity
            i = inclination
            W = right ascension of the ascending node
            w = argument of perigee
            M = mean anomaly

    Returns:
        coef: ndarray
            (m, 6) array of COEs with true anomaly ordered as
            (a, e, i, W, w, f), where
            a = semi-major axis
            e = eccentricity
            i = inclination
            W = right ascension of the ascending node
            w = argument of perigee
            f = true anomaly
    """
    return coef_coeE(coeE_coeM(coeM))
