import numpy as np

__author__ = "Nathan I. Budd"
__email__ = "nibudd@gmail.com"
__copyright__ = "Copyright 2017, LASR Lab"
__license__ = "MIT"
__version__ = "0.1"
__status__ = "Production"
__date__ = "16 Mar 2017"


class GVE(object):
    """Gauss's Variational Equations for modified equinoctial elements.

    Attributes:
        mu: float, optional
            Standard Gravitational Parameter. Defaults to 1.0, the standard
            value in canonical units.

    References:
        Walker, M.J.H., Ireland, B., & Owens, J. (1985).
        A Set of Modified Equinoctial Orbit Elements.
        Celestial Mechanics, 36, 409–419.

        Walker, M.J.H., Ireland, B., & Owens, J. (1985).
        A Set of Modified Equinoctial Orbit Elements - Errata.
        Celestial Mechanics, 36, 409–419.
    """

    def __init__(self, mu=1.0):
        self.mu = mu

    def __call__(self, T, X):
        """Calculate GVE matrices for modified equinoctial elements.

        Args:
            T: ndarray
                (m, 1) array of times.
            X: ndarray
                (m, 6) array of modified equinoctial elements ordered as
                (p, f, g, h, k, L), where
                p = semi-latus rectum
                f = 1-component of eccentricity vector in perifocal frame
                g = 2-component of eccentricity vector in perifocal frame
                h = 1-component of ascending node vector in equinoctial frame
                k = 2-component of ascending node vector in equinoctial frame
                L = true longitude

        Returns:
            G: ndarray
                (m, 6, 3) array of GVE matrices.
        """
        dims = (T.shape[0], 1, 1)

        p = X[:, 0:1].reshape(dims)
        f = X[:, 1:2].reshape(dims)
        g = X[:, 2:3].reshape(dims)
        h = X[:, 3:4].reshape(dims)
        k = X[:, 4:5].reshape(dims)
        L = X[:, 5:6].reshape(dims)

        sL = np.sin(L)
        cL = np.cos(L)
        s = (1. + h**2 + k**2)
        w = 1. + f*cL + g*sL
        q = (h*sL - k*cL) / w
        zero = np.zeros(dims)

        pdot = np.concatenate((zero, 2*p/w, zero), axis=2)
        fdot = np.concatenate((sL, ((w+1.)*cL + f)/w, -g*q), axis=2)
        gdot = np.concatenate((-cL, ((w+1.)*sL + g)/w, f*q), axis=2)
        hdot = np.concatenate((zero, zero, s**2 * cL/2/w), axis=2)
        kdot = np.concatenate((zero, zero, s**2 * sL/2/w), axis=2)
        Ldot = np.concatenate((zero, zero, q), axis=2)

        return (p / self.mu)**0.5 * np.concatenate(
            (pdot, fdot, gdot, hdot, kdot, Ldot), axis=1)
