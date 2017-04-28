import numpy as np

__author__ = "Nathan I. Budd"
__email__ = "nibudd@gmail.com"
__copyright__ = "Copyright 2017, LASR Lab"
__license__ = "MIT"
__version__ = "0.1"
__status__ = "Production"
__date__ = "08 Mar 2017"


def meeEl_meeMl(meeMl, tol=1e-14):
    """Convert MEEs with mean longitude to eccentric longitude.

    Args:
        meeMl: ndarray
            (m, 6) array of modified equinoctial elements ordered as
            (p, f, g, h, k, Ml), where
            p = semi-latus rectum
            f = 1-component of eccentricity vector in perifocal frame
            g = 2-component of eccentricity vector in perifocal frame
            h = 1-component of the ascending node vector in equinoctial frame
            k = 2-component of the ascending node vector in equinoctial frame
            Ml = mean longitude
        tol: float, optional
            Tolerance for checking convergence of Newton's method.

    Returns:
        meeEl: ndarray
            (m, 6) array of modified equinoctial elements ordered as
            (p, f, g, h, k, El), where
            p = semi-latus rectum
            f = 1-component of eccentricity vector in perifocal frame
            g = 2-component of eccentricity vector in perifocal frame
            h = 1-component of the ascending node vector in equinoctial frame
            k = 2-component of the ascending node vector in equinoctial frame
            El = eccentric longitude
    """
    f = meeMl[:, 1:2]
    g = meeMl[:, 2:3]
    Ml = meeMl[:, 5:6]
    El0 = Ml + np.sign(np.sin(Ml))*(f**2 + g**2)**0.5
    El1 = El0 + 1.
    max_iterations = 15
    iter_num = 0

    while np.max(np.absolute(El1 - El0)) > tol:
        El0 = El1
        El1 = (El0 - (Ml - El0 + f*np.sin(El0) - g*np.cos(El0)) /
               (-1. + f*np.cos(El0) + g*np.sin(El0)))
        iter_num += 1
        if iter_num > max_iterations:
            print(
                'max ({}) iterations reached in meeEl_meeMl. Error: {}'.format(
                    max_iterations, np.max(np.absolute(El1 - El0))))
            break

    return np.concatenate(
        (meeMl[0:, 0:5], np.mod(El1, 2*np.pi)), axis=1)
