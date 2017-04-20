import numpy as np

__author__ = "Nathan I. Budd"
__email__ = "nibudd@gmail.com"
__copyright__ = "Copyright 2017, LASR Lab"
__license__ = "MIT"
__version__ = "0.1"
__status__ = "Production"
__date__ = "20 Apr 2017"


class FunctionTweak():
    """Generate a set of callables with tweaked parameters.

    Given a callable that relies on a set of parameters and generates a
    trajectory as a function of time, this will produce a set of callables with
    each parameter tweaked, as well as a reference callable with no tweaks.

    Attributes:
        f: callable
            Callable object that takes inputs (T, p), where T is some
            independent variable and p is an ndarray of parameters, and returns
            a trajectory.
        tweak: float
            Amount by which each parameter will be tweaked.
    """

    def __init__(self, f, tweak):
        self.f = f
        self.tweak = tweak

    def __call__(self, p):
        """Generate a set of tweaked callables.

        Args:
            p: ndarray
                (1, n) array of parameters.

        Returns:
            F : list
                List of callables. The first is the untweaked callable. The
                rest are the tweaked callables, tweaked in the order that they
                are listed in p, and used in self.f.
        """
        def make_tweaked_f(p):
            def tweaked_f(T):
                return self.f(T, p)

            return tweaked_f

        n = p.shape[1]

        tweak_matrix = np.concatenate(
            (np.zeros((1, n)),
             np.identity(n)), axis=0
        ) * self.tweak

        tweaked_p = np.tile(p, (n+1, 1)) + tweak_matrix
        return [make_tweaked_f(p_i.reshape((1, n))) for p_i in tweaked_p]
