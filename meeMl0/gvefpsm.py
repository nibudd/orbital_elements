import numpy as np
from orbital_elements.meeMl0.gvessm import GVESSM

__author__ = "Nathan I. Budd"
__email__ = "nibudd@gmail.com"
__copyright__ = "Copyright 2017, LASR Lab"
__license__ = "MIT"
__version__ = "0.1"
__status__ = "Production"
__date__ = "20 Apr 2017"


class GVEFPSM(object):
    """Force parameter sensitivity matrix for MEEs with mean longitude at epoch.

    Attributes:
        h: float
            Step size used in the finite difference method.
        mu: float, optional
            Standard Gravitational Parameter. Defaults to 1.0, the standard
            value in canonical units.
        vectorize: boolean
            When False, the return array is (m, 6, 6). When True, the return
            array is (m, 36, 1), a vectorized version of the (m, 6, 6) matrix
            which can be matrix added to a vectorized Psi.

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

    def __init__(self, h=0, mu=1.0, vectorize=False):
        self.h = h
        self.mu = mu
        self.vectorize = vectorize

    def __call__(self, T, X, a_d):
        """Calculate FPSMs for MEEs with mean longitude at epoch.

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
            a_d: ndarray
                (m, 3) generalized LVLH perturbation vector.

        Returns:
            dGdx: ndarray
                (m, 6, 6) array of STMs, or an (m, 36, 36) block diagonal of
                STMs, depending on the value of vectorize.
        """
        A = GVESSM(h=self.h, mu=self.mu, vectorize=self.vectorize)

        m = T.shape[0]
        Xh = (np.tile(X.reshape((m, 1, 6)), (1, 6, 1)) +
              np.tile(np.identity(6), (m, 1, 1))*self.h)

        G_func = GVE(mu=self.mu)
        G = G_func(T, X)

        a_d = a_d.reshape((m, 3, 1))

        return np.concatenate(
            [(G_func(T, x.reshape((m, 6))) - G)/self.h @ a_d
             for x in Xh.transpose((1, 0, 2))],
            axis=2
        )
