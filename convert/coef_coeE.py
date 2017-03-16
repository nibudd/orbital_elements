import numpy as np

__author__ = "Nathan I. Budd"
__email__ = "nibudd@gmail.com"
__copyright__ = "Copyright 2017, LASR Lab"
__license__ = "MIT"
__version__ = "0.1"
__status__ = "Production"
__date__ = "05 Mar 2017"


def coef_coeE(coeE):
    """Convert COEs with eccentric anomaly to true anomaly.

    Args:
        coeE: ndarray
            (m, 6) array of COEs with eccentric anomaly ordered as
            (a, e, i, W, w, f), where
            a = semi-major axis
            e = eccentricity
            i = inclination
            W = right ascension of the ascending node
            w = argument of perigee
            E = eccentric anomaly

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
    e = coeE[0:, 1:2]
    E = coeE[0:, 5:6]

    tan_f_by_2 = ((1.+e)/(1.-e))**.5 * np.tan(E/2)
    f = np.mod(2 * np.arctan(tan_f_by_2), 2*np.pi)

    return np.concatenate((coeE[0:, 0:5], f), axis=1)
