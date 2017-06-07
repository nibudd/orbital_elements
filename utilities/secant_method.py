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
            The objective function, f(x). Takes (n,) ndarrays as input and
            returns (m,) arrays as output.
        x0: float
            (n,) array of first root guess.
        x1: float
            (n,) array of second root guess.
        tol: float, optional
            Allowable error in the zero value.
        Nmax: int, optional
            Maximum number of allowed iterations.

    Returns:
        X: ndarray
            (m, 1) array of all the root estimates found.
        E: ndarray
            (m, 1) array of errors for each root estimate found
        N: int
            Final iteration number.

    """

    above_tol = True
    below_Nmax = True
    X = [x0, x1]
    E = [f(x0), f(x1)]

    while above_tol and below_Nmax:
        print('Starting iteration {}'.format(len(X)))
        f0, f1 = E[-2], E[-1]
        x0, x1 = X[-2], X[-1]

        try:
            X += [x1 - f1 * (x1 - x0) / (f1 - f0)]
        except ZeroDivisionError:
            X += [x1]

        E += [f(X[-1])]

        above_tol = True if np.all(np.abs(E[-1]) > tol) else False
        below_Nmax = True if len(X) < Nmax else False

        print('Iteration {}: (guess, error) = ({}, {})'.format(len(X)-1,
                                                               X[-1],
                                                               E[-1]))
        print('**********************')

    if below_Nmax is False:
        print('Reached maximum iterations (50)')
    return (np.array(X), np.array(E), len(X)-1)
