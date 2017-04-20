__author__ = "Nathan I. Budd"
__email__ = "nibudd@gmail.com"
__copyright__ = "Copyright 2017, LASR Lab"
__license__ = "MIT"
__version__ = "0.1"
__status__ = "Production"
__date__ = "14 Apr 2017"


class SplineOrd3():
    """3rd order spline that matches initial and final position and velocity.

    Given an initial and final state, generates a cubic spline that matches the
    provided initial and final states and state velocities across the time
    frame given. Can be called to return spline states at an array of times.

    Attributes:
        X0: ndarray
            (1, n) array of initial states.
        Xf: ndarray
            (1, n) array of final states.
        tf: double
            Time when state reaches Xf (i.e. X(tf) = Xf), given X(0) = X0.
    """

    def __init__(self, X0, Xf, tf):
        """."""
        self.X0 = X0
        self.Xf = Xf
        self.tf = tf

    def X(self, T):
        """Return the splined states.

        Args:
            T : ndarray
                An (m, 1) array of times.

        Returns:
            X : ndarray
                An (m, n) array of states.
        """
        tau = T / self.tf
        Yf = tau**2 * (3 - 2*tau)
        Y0 = 1 - Yf

        return self.X0 * Y0 + self.Xf * Yf

    def Xdot(self, T):
        """Evaluate the full system dynamics.

        Args:
            T : ndarray
                An (m, 1) array of times.

        Returns:
            Xdot : ndarray
                An (m, n) array of state derivatives.
        """
        tau = T / self.tf
        dYf_dtau = 6*tau * (1 - tau)
        dY0_dtau = -dYf_dtau

        return self.X0*dY0_dtau + self.Xf*dYf_dtau
