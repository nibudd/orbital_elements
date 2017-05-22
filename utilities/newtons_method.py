import numpy as np
from scipy.integrate import ode


__author__ = "Nathan I. Budd"
__email__ = "nibudd@gmail.com"
__copyright__ = "Copyright 2017, LASR Lab"
__license__ = "MIT"
__date__ = "19 May 2017"


def newtons_method(f, x_guess, J, tol=1e-14, Nmax=30):
    """Multidimensional Newton's method root solver.

    args:
        f: callable
            The objective function, f(x). Takes (n,) ndarrays as input and
            returns (m,) arrays as output.
        x_guess: ndarray
            (n,) ndarray of initial guesses.
        J: callable
            The Jacobian, J(x), of the objective function. Takes (n,) ndarrays
            as input and returns (m, n) array of partial derivatives of f with
            respect to x.
        tol: float, optional
            Allowable error in the zero value.
        Nmax: int, optional
            Maximum allowed number of iterations.
    returns:
        x: ndarray
            The roots of f. An (n,) ndarray.
    """
    x1 = x_guess
    f1, J1 = f(x1), J(x1)
    iters = 0
    while np.max(np.absolute(f1)) > tol and iters <= Nmax:
        x0, f0, J0 = x1, f1, J1
        try:
            x1 = x0 - np.linalg.pinv(J0) @ f0
        except np.linalg.LinAlgError:
            x1 = x0 - f0 / J0
        f1, J1 = f(x1), J(x1)
        iters += 1

    return x1
