import orbital_mechanics.orbit as orb
from orbital_elements.coe.keplerian_solution import KeplerianSolution as KepSol

__author__ = "Nathan I. Budd"
__email__ = "nibudd@gmail.com"
__copyright__ = "Copyright 2017, LASR Lab"
__license__ = "MIT"
__version__ = "0.1"
__status__ = "Production"
__date__ = "02 Mar 2017"


class KeplerianSolution(object):
    """Keplerian solution for position-velocity elements.

    Attributes:
        X0: ndarray
            (1, 6) array of initial states ordered as (rx, ry, rz, vx, vy, vz),
            where
            rx = position x-component
            ry = position y-component
            rz = position z-component
            vx = velocity x-component
            vy = velocity y-component
            vz = velocity z-component
        mu: float, optional
            Standard Gravitational Parameter. Defaults to 1.0, the standard
            value in canonical units.
    """

    def __init__(self, X0, mu=1.0):
        self.X0 = X0
        self.mu = mu

    def __call__(self, T):
        """Calculate position-velocity solution.

        Args:
            T: ndarray
                (m, 1) array of times.

        Returns:
            X: ndarray
                (m, 6) array of states.
        """
        X_coe = KepSol(orb.rv2coe(self.X0), mu=self.mu)(T)
        return orb.coe2rv(X_coe)
