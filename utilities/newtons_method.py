import numpy as np


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
        X: ndarray
            (N, n) array of all root estimates found.
        E: ndarray
            (N, m) array of all function evaluations.
        N: int
            Final iteration number.

    """
    X = [x_guess]
    E = [f(X[0])]
    J1 = J(X[0])
    N = 0
    while np.max(np.absolute(E[-1])) > tol and N <= Nmax:
        x0, f0, J0 = X[-1], E[-1], J1
        try:
            X += [x0 - (np.linalg.pinv(J0) @ f0).flatten()]
        except np.linalg.LinAlgError:
            X += [x0 - f0 / J0]
        E += [f(X[-1])]
        J1 = J(X[-1])
        N += 1
        print("Iteration {}. (root, error) = ({}, {})".format(N, X[-1], E[-1]))

    return (np.array(X), np.array(E), N)
