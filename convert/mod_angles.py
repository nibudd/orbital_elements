import numpy as np

__author__ = "Nathan I. Budd"
__email__ = "nibudd@gmail.com"
__copyright__ = "Copyright 2017, LASR Lab"
__license__ = "MIT"
__version__ = "0.1"
__status__ = "Production"
__date__ = "05 Mar 2017"


def mod_angles(X, angle_indices=[5]):
    """Mod by 2pi any indicated columns.

    Args:
        X: ndarray
            (m, n) array of n states, some of which are angles
        angle_indices: list, optional
            Integer list of indices of columns to be modded by 2pi.

    Returns:
        Xmod: ndarray
            (m, n) array of n states with the angle states modded by 2pi
    """
    for k in angle_indices:
        X[:, k:k+1] = np.mod(X[:, k:k+1], 2*np.pi)
        X[:, k:k+1] = np.array(
            [x-2*np.pi if x > np.pi else x for x in X[:, k:k+1]]
        )
        X[:, k:k+1] = np.array(
            [x+2*np.pi if x < -np.pi else x for x in X[:, k:k+1]]
        )

    return X
