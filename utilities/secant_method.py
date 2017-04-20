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
        f: callable
            Takes a float as input and returns a float.
        x0: float
            First guess for f(x) = 0.
        x1: float
            Second guess for f(x) = 0.
        tol: float
            Error allowed between successive values of f(x).
        Nmax: int
            Total number of allowed iterations.

    Returns:
        X: ndarray
            (m, 1) array of all the root guesses found.
        e: ndarray
            (m, 1) array of errors for each root guess found


    """

    above_tol = True
    below_Nmax = True
    X = []
    e = []

    while above_tol and below_Nmax:
        print('Starting iteration {}'.format(len(X)))
        f0 = f(x0)
        f1 = f(x1)

        X += [x1]
        e += [np.abs(f1 - f0)]

        if e[-1] != 0:
            x2 = x1 - f1 * (x1 - x0) / (f1 - f0)
        else:
            x2 = x1

        (x0, x1) = (x1, x2)

        above_tol = True if e[-1] > tol else False
        below_Nmax = True if len(X) < Nmax else False

        print('Iteration {}: (guess, error) = ({}, {})'.format(len(X)-1,
                                                               X[-1],
                                                               e[-1]))
        print('**********************')

    if below_Nmax is False:
        print('Reached maximum iterations (50)')
    return (np.array(X), np.array(e))
