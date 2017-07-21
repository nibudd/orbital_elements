import numpy as np


__author__ = "Nathan I. Budd"
__email__ = "nibudd@gmail.com"
__copyright__ = "Copyright 2017, LASR Lab"
__license__ = "MIT"
__date__ = "07 June 2017"


def min_root(x0, f, Jac, psi=None, psi_Jac=None, delta_f=None, tol_f=1e-14,
             Nmax=50):
    """Multidimensional equality constraint minimizer.

    Uses `delta_f` to set a target value of the cost function to meet
    while satisfying the constraints `psi`. A root solver is used to find the
    roots which satisfy the constraints and achieve the target cost
    function value. Once met, the target cost function value is lowered
    and the process is repeated. This loop continues until

    args:
        x0: ndarray
            (n,) ndarray of initial guesses.
        f: callable
            The cost function, `f(x)`. Takes (n,) ndarrays as input and
            returns (m,) arrays as output.
        Jac: callable
            The Jacobian `Jac(x)` of the cost function. Takes (n,)
            ndarrays as input and returns (m, n) array of partial derivatives
            of `f` with respect to `x`.
        psi: callable, optional
            The constraint function `psi(x)` which is zero when all constraints
            are satisfied. Takes (n,) ndarray as input and returns (k,) ndarray
            as output.
        psi_Jac: callable, optional
            The Jacobian of the constraint functions. Takes (n,)
            ndarrays as input and returns (k, n) array of partial derivatives
            of `f` with respect to `x`.
        delta_f: ndarray, optional
            Initial sought reduction in the cost function as an (m,)
            ndarray.
        tol_f: float, optional
            Value of `delta_f` below which `f` is considered minimized.
        Nmax: int, optional
            Maximum allowed number of iterations.
    returns:
        X: ndarray
            (N, n) array of all minimizing estimates found.
        E: ndarray
            (N, m) array of all function evaluations.
        N: int
            Final iteration number.
    """
    N = 0
    Nc = 0
    X = np.atleast_2d(x0)
    F = np.atleast_2d(f(x0))
    X_c = X[0]
    f_c = F[0]

    # guess an initial cost reduction if none is provided
    if delta_f is None:
        delta_f = np.abs(f_c) * 0.1

    print("#0-0 (x, f) = ({}, {})".format(X_c, f_c))

    # TODO satisfy constraints, then move on

    while True:
        # MAKE CORRECTION
        f_target = f_c - delta_f
        A_top = Jac(X_c)
        try:
            A_bot = psi_Jac(X_c)
            A = np.concatenate((A_top, A_bot), axis=0)
            b = np.atleast_2d(np.concatenate((f_c - f_target, psi(X_c)))).T
        except TypeError:
            A = A_top
            b = np.atleast_2d(f_c - f_target).T

        X_new = np.atleast_1d(X_c - (np.linalg.pinv(A) @ b).flatten())
        f_new = np.atleast_1d(f(X_new))
        N, Nc = (N+1, Nc+1)

        # SAVE VALUES
        X = np.append(X, np.atleast_2d(X_new), axis=0)
        F = np.append(F, np.atleast_2d(f_new), axis=0)

        # PRINT OUTPUT
        print("#{}-{} (x, f) = ({}, {})".format(N, Nc, X_new, f_new))

        # BREAK CONDITIONS
        if N == Nmax:
            print("Max iterations reached.")
            break

        if delta_f < tol_f / 2:
            print("Minimum found.")
            break

        # BACKTRACK DECISION
        if F[-1] > F[-2]:
            X_c = X[-2]
            f_c = F[-2]
            Nc = 0
            delta_f = delta_f / 20
            print("#{}-{} (x, f) = ({}, {})".format(N, Nc, X_c, f_c))
        else:
            X_c = X[-1]
            f_c = F[-1]

    return X, F, N
