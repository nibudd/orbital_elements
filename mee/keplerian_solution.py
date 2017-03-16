import numpy as np
import orbital_elements.convert as convert


__author__ = "Nathan I. Budd"
__email__ = "nibudd@gmail.com"
__copyright__ = "Copyright 2017, LASR Lab"
__license__ = "MIT"
__version__ = "0.1"
__status__ = "Production"
__date__ = "08 Mar 2017"


class KeplerianSolution(object):
    """Keplerian solution for modified equinoctial elements.

    Attributes:
        X0: ndarray
            (m, 6) array of position-velocity elements ordered as
            (p, f, g, h, k, L), where
            p = semi-latus rectum
            f = 1-component of eccentricity vector in perifocal frame
            g = 2-component of eccentricity vector in perifocal frame
            h = 1-component of the ascending node vector in equinoctial frame
            k = 2-component of the ascending node vector in equinoctial frame
            L = true longitude
        mu: float, optional
            Standard Gravitational Parameter. Defaults to 1.0, the standard
            value in canonical units.
    """

    def __init__(self, X0, mu=1.0):
        self.X0 = X0
        self.mu = mu

    def __call__(self, T):
        """Calculate modified equinocital elements solution.

        Args:
            T: ndarray
                (m, 1) array of times.

        Returns:
            X: ndarray
                (m, 6) array of states.
        """
        T0 = T[0, 0]
        p = self.X0[0, 0]
        f = self.X0[0, 1]
        g = self.X0[0, 2]
        n = (self.mu * ((1 - f**2 - g**2) / p)**3)**.5

        dT = T - T0
        dMl = dT * n
        Ml0 = convert.meeMl_meefl(self.X0)[0, 5]
        Ml = Ml0 + dMl

        meeMl = np.concatenate(
            (np.tile(self.X0[0:, 0:5], T.shape), Ml), axis=1)

        return convert.meefl_meeMl(meeMl)
