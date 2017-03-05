import numpy as np

__author__ = "Nathan I. Budd"
__email__ = "nibudd@gmail.com"
__copyright__ = "Copyright 2017, LASR Lab"
__license__ = "MIT"
__version__ = "0.1"
__status__ = "Production"
__date__ = "05 Mar 2017"


class GVE(object):
    """Gauss's Variational Equations for COEs w/ epoch mean anomaly.

    Attributes:
        mu: float, optional
            Standard Gravitational Parameter. Defaults to 1.0, the standard
            value in canonical units.
    """

    def __init__(self, mu=1.0):
        self.mu = mu

    def __call__(self, T, X):
        """Calculate GVE matrices for classical elements w/ epoch mean anomaly.

        Args:
            T: ndarray
                (m, 1) array of times.
            X: ndarray
                (m, 6) array of states ordered as (a, e, i, W, w, M0), where
                a = semi-major axis
                e = eccentricity
                i = inclination
                W = right ascension of the ascending node
                w = argument of perigee
                M0 = epoch mean anomaly

        Returns:
            G: ndarray
                (m, 6, 3) array of GVE matrices.
        """
        dims = (T.shape[0], 1, 1)

        a = X[:, 0:1].reshape(dims)
        e = X[:, 1:2].reshape(dims)
        i = X[:, 2:3].reshape(dims)
        w = X[:, 4:5].reshape(dims)
        f = X[:, 5:6].reshape(dims)

        sf = np.sin(f)
        cf = np.cos(f)
        st = np.sin(f + w)
        ct = np.cos(f + w)
        si = np.sin(i)
        ci = np.cos(i)
        b = a * (1 - e**2)**0.5
        p = a * (1. - e**2)
        r = p / (1. + e*cf)
        h = (self.mu * p)**.5
        zero = np.zeros(dims)

        adot = np.concatenate((e*sf, p/r, zero), axis=2) * 2*a**2/h
        edot = np.concatenate((p*sf, (p+r)*cf + r*e, zero), axis=2) / h
        idot = np.concatenate((zero, zero, r*ct/h), axis=2)
        Wdot = np.concatenate((zero, zero, r*st/h/si), axis=2)
        wdot = np.concatenate((-p*cf/e, (p+r)*sf/e, -r*st*ci/si), axis=2) / h
        Mdot_minus_n = np.concatenate(
            (p*cf-2*r*e, -(p+r)*sf, zero), axis=2) * b/(a*h*e)
        M0dot = Mdot_minus_n + 3/2*(self.mu/a**5)**0.5 * adot * T.reshape(dims)


        return np.concatenate((adot, edot, idot, Wdot, wdot, M0dot), axis=1)
