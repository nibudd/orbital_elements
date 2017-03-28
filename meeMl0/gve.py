import numpy as np
from .. import convert

__author__ = "Nathan I. Budd"
__email__ = "nibudd@gmail.com"
__copyright__ = "Copyright 2017, LASR Lab"
__license__ = "MIT"
__version__ = "0.1"
__status__ = "Production"
__date__ = "27 Mar 2017"


class GVE(object):
    """Gauss's Variational Equations for MEEs with mean longitude at epoch.

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

        Battin, R.H. (1999)
        An Introduction to the Mathematics and Methods of Astrodynamics.
        AIAA Education Series. 493
    """

    def __init__(self, mu=1.0):
        self.mu = mu

    def __call__(self, T, X):
        """Calculate GVE matrices for MEEs with mean longitude at epoch.

        Args:
            T: ndarray
                (m, 1) array of times.
            X: ndarray
                (m, 6) array of modified equinoctial elements ordered as
                (p, f, g, h, k, Ml0), where
                p = semi-latus rectum
                f = 1-component of eccentricity vector in perifocal frame
                g = 2-component of eccentricity vector in perifocal frame
                h = 1-component of the ascending node vector in equ. frame
                k = 2-component of the ascending node vector in equ. frame
                Ml0 = mean longitude at epoch

        Returns:
            G: ndarray
                (m, 6, 3) array of GVE matrices.
        """
        dims = (T.shape[0], 1, 1)

        X_L = convert.mee_meeMl0(T, X)

        p = X_L[:, 0:1].reshape(dims)
        f = X_L[:, 1:2].reshape(dims)
        g = X_L[:, 2:3].reshape(dims)
        h = X_L[:, 3:4].reshape(dims)
        k = X_L[:, 4:5].reshape(dims)
        L = X_L[:, 5:6].reshape(dims)

        sL = np.sin(L)
        cL = np.cos(L)
        s = 1. + h**2 + k**2
        w = 1. + f*cL + g*sL
        q = (h*sL - k*cL) / w
        zero = np.zeros(dims)

        pdot = np.concatenate((zero, 2*p/w, zero), axis=2)
        fdot = np.concatenate((sL, ((w+1.)*cL + f)/w, -g*q), axis=2)
        gdot = np.concatenate((-cL, ((w+1.)*sL + g)/w, f*q), axis=2)
        hdot = np.concatenate((zero, zero, s*cL/2/w), axis=2)
        kdot = np.concatenate((zero, zero, s*sL/2/w), axis=2)

        b_by_a = (2 - w)**0.5
        a_by_a_plus_b = 1 / (1 + b_by_a)
        # Mldot = np.concatenate(
        #     (
        #         a_by_a_plus_b * w*(w-1) + 2*b_by_a,
        #         a_by_a_plus_b * (w+1) * (g*cL - f*sL),
        #         k*cL - h*sL
        #         ), axis=2) / -w
        X_coe = convert.coe_mee(X_L)
        a = X[:, 0:1].reshape(dims)
        e = X[:, 1:2].reshape(dims)
        i = X[:, 2:3].reshape(dims)
        w = X[:, 4:5].reshape(dims)
        v = X[:, 5:6].reshape(dims)
        st = np.sin(v+w)
        ci = np.cos(i)
        si = np.sin(i)
        cv = np.cos(v)
        sv = np.sin(v)
        r = p / (1. + e*cv)
        H = (self.mu * p)**.5
        b = a * (1 - e**2)**0.5
        Wdot = np.concatenate((zero, zero, r*st/H/si), axis=2)
        wdot = np.concatenate((-p*cv/e, (p+r)*sv/e, -r*st*ci/si), axis=2) / H
        Mdot = np.concatenate((p*cv-2*r*e, -(p+r)*sv, zero), axis=2) * b/a/H/e
        Mldot = Mdot + Wdot + wdot

        ndot = -3/2 * self.mu / p**3 * b_by_a * ((2-w)*pdot +
                                                 2*p*(f*fdot + g*gdot))
        # ndot = np.concatenate(
        #     (f*sL - g*cL, w, zero), axis=2
        #     ) * -3 * (self.mu / p**3) * b_by_a
        Ml0dot = Mldot - ndot*T.reshape(dims)

        return (p / self.mu)**0.5 * np.concatenate(
            (pdot, fdot, gdot, hdot, kdot, Ml0dot), axis=1)
