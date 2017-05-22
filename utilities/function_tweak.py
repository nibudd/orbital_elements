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
        h: float
            Amount by which each parameter will be tweaked.
    """

    def __init__(self, f, h):
        self.f = f
        self.h = h

    def __call__(self, p, tweak_indicies=None):
        """Generate a set of tweaked callables.

        Args:
            p: ndarray
                (1, n) array of parameters.
            tweak_indicies: list, optional
                List of indicies, indicating which parameters should be tweaked
                in this iteration. If not provided, all parameters will be
                tweaked.

        Returns:
            F : list
                List of callables. The first is the untweaked callable. The
                rest are the tweaked callables, tweaked in the order that they
                are listed in p, and used in self.f. The callables take an
                (m, 1) time array and an unused X parameter as input, which
                allows it to be used by integrators. The callable returns a
                trajectory just as the original f callable does.
        """
        def make_tweaked_f(p):
            def tweaked_f(T, X=[]):
                return self.f(T, p)

            return tweaked_f

        # CODE FROM AFTER tweak_indicies WAS ADDED.
        # F = [make_tweaked_f(p)]
        # if tweak_indicies is None:
        #     tweak_indicies = range(p.shape[1])
        #     for idx in tweak_indicies:
        #         p_tweaked = np.array(p)
        #         p_tweaked[0, idx] = p_tweaked[0, idx] + self.h
        #         F += [make_tweaked_f(p_tweaked)]

        # return F

        n = p.shape[1]

        tweak_matrix = np.concatenate(
            (np.zeros((1, n)),
             np.identity(n)), axis=0
            ) * self.h

        tweaked_p = np.tile(p, (n+1, 1)) + tweak_matrix
        return [make_tweaked_f(p_i.reshape((1, n))) for p_i in tweaked_p]
