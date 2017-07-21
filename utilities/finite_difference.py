import numpy as np

from .function_tweak import FunctionTweak


__author__ = "Nathan I. Budd"
__email__ = "nibudd@gmail.com"
__copyright__ = "Copyright 2017, LASR Lab"
__license__ = "MIT"
__date__ = "15 June 2017"


def finite_difference(f, p, h):
    """Create a sensitivity matrix from finite difference.

    Using finite difference, create a sensitivity matrix of the function `f` to
    its parameters.

    args:
        f: callable
            Callable object that takes inputs (t, p), where t is some
            independent variable and p is an ndarray of parameters. Returns
            a (n,) array of values.
        p: ndarray
            (m,) array of parameters about which the Jacobian of f is to be
            linearized.
        h: float
            Amount by which each parameter will be tweaked.
    returns:
        Jac: callable
            Takes as input the independent variable `t`, from `f`, and returns
            an (n, m) sensitivity matrix containing the n*m partial derivatives
            of the function `f` to the `m` parameters of `p`.
    """
    tweaker = FunctionTweak(f, h)
    tweaked_funcs = tweaker(np.atleast_2d(p))
    main_func = tweaked_funcs[0]

    def finite_func(t):
        list_of_diffs = [(tweaked_func(t) - main_func(t))/h
                         for tweaked_func in tweaked_funcs[1:]]
        return np.column_stack(list_of_diffs)

    return finite_func
