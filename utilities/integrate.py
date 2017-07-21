import numpy as np
from scipy.integrate import ode


__author__ = "Nathan I. Budd"
__email__ = "nibudd@gmail.com"
__copyright__ = "Copyright 2017, LASR Lab"
__license__ = "MIT"
__date__ = "12 May 2017"


def integrate(f, X0, T, *args, atol=1e-13, rtol=1e-13, **kwargs):
    """Easy-to-use wrapper for SciPy's Dormand-Prince integrator.

    args:
        f: callable
            State time derivatives. Takes inputs (t, X). Time, t, is a double,
            and the state, X, is an (n,) ndarray.
        X0: ndarray
            (n,) array of initial state conditions.
        T: ndarray
            (m,) array of times at which the state value should be returned.
        args: list, optional
            Additional arguments for f.
        atol: double, optional
            Absolute tolerance for the integrator.
        rtol: double, optional
            Relative tolerance for the integrator.
        kwargs: dict
            Additional keyword arguments for the integrator.
    returns:
        X: ndarray
            (m, n) array of the state X at each time step in T.
        """
    solver = ode(f).set_integrator("dopri5", atol=atol, rtol=rtol, **kwargs)
    solver.set_initial_value(X0, T[0]).set_f_params(*args)
    m = T.shape[0]
    n = X0.shape[0]
    X = np.empty((m, n))
    X[0] = X0
    for k, t in enumerate(T[1:]):
        X[k+1] = solver.integrate(t)
        if not solver.successful():
            break
    return X
