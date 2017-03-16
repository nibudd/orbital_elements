import numpy as np

__author__ = "Nathan I. Budd"
__email__ = "nibudd@gmail.com"
__copyright__ = "Copyright 2017, LASR Lab"
__license__ = "MIT"
__version__ = "0.1"
__status__ = "Production"
__date__ = "05 Mar 2017"


def coeE_coef(coef):
    """Convert COEs with true anomaly to eccentric anomaly.

    Args:
        coef: ndarray
            (m, 6) array of COEs with true anomaly ordered as
            (a, e, i, W, w, f), where
            a = semi-major axis
            e = eccentricity
            i = inclination
            W = right ascension of the ascending node
            w = argument of perigee
            f = true anomaly

    Returns:
        coeE: ndarray
            (m, 6) array of COEs with eccentric anomaly ordered as
            (a, e, i, W, w, f), where
            a = semi-major axis
            e = eccentricity
            i = inclination
            W = right ascension of the ascending node
            w = argument of perigee
            E = eccentric anomaly
    """
    e = coef[0:, 1:2]
    f = coef[0:, 5:6]

    tan_E_by_2 = ((1.-e)/(1.+e))**.5 * np.tan(f/2)
    E = np.mod(2 * np.arctan(tan_E_by_2), 2*np.pi)

    return np.concatenate((coef[0:, 0:5], E), axis=1)
