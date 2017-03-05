import numpy as np

__author__ = "Nathan I. Budd"
__email__ = "nibudd@gmail.com"
__copyright__ = "Copyright 2017, LASR Lab"
__license__ = "MIT"
__version__ = "0.1"
__status__ = "Production"
__date__ = "05 Mar 2017"


def coeE_coeM(coeM, tol=1e-14):
    """Convert COEs with mean anomaly to eccentric anomaly.

    Args:
        coeM: ndarray
            (m, 6) array of COEs with true anomaly ordered as
            (a, e, i, W, w, f), where
            a = semi-major axis
            e = eccentricity
            i = inclination
            W = right ascension of the ascending node
            w = argument of perigee
            M = mean anomaly

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
    e = coeM[0:, 1:2]
    M = coeM[0:, -1:]
    E0 = M + np.sign(np.sin(M))*e
    E1 = E0 + 1.
    max_iterations = 30
    iter_num = 0

    while np.max(np.absolute(E1 - E0)) > tol:
        E0 = E1
        E1 = E0 + (M - E0 + e*np.sin(E0)) / (1. - e*np.cos(E0))
        iter_num += 1
        if iter_num > max_iterations:
            print(
                'max ({}) iterations reached in coeE_coeM. Error: {}'.format(
                    max_iterations, np.max(np.absolute(E1 - E0))))
            break

    return np.concatenate(
        (coeM[0:, 0:-1], np.mod(E1, 2*np.pi)), axis=1)
