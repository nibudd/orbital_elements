import numpy as np

__author__ = "Nathan I. Budd"
__email__ = "nibudd@gmail.com"
__copyright__ = "Copyright 2017, LASR Lab"
__license__ = "MIT"
__version__ = "0.1"
__status__ = "Production"
__date__ = "16 Mar 2017"


def secant_method(f, x0, x1, tol=1e-14, Nmax=50):
    """Uses approximation of Newton's method to estimate root of f.

    Args:
        f : callable
            Takes an array with shape x0.shape as input.
        x0 : ndarray
            First guess for f(x) = 0.
        X1 : ndarray
            Second guess for f(x) = 0.
        tol : float
            Error allowed between successive values of f(x).
        Nmax : int
            Total number of allowed iterations.

    Returns:
        x : ndarray
            Approximate root of f
    """

    above_tol = True
    below_Nmax = True
    N = 0

    while above_tol and below_Nmax:
        f0 = f(x0)
        f1 = f(x1)

        x2 = x1 - f1 * (x1 - x0) / (f1 - f0)

        x0 = x1
        x1 = x2

        N += 1
        above_tol = True if np.linalg.norm(f0-f1) > tol else False
        below_Nmax = True if N < Nmax else False

    if below_Nmax is False:
        print('Reached maximum iterations (50)')
    return x2
