import numpy as np

__author__ = "Nathan I. Budd"
__email__ = "nibudd@gmail.com"
__copyright__ = "Copyright 2017, LASR Lab"
__license__ = "MIT"
__version__ = "0.1"
__status__ = "Production"
__date__ = "04 Mar 2017"


def euler_angles(axes, angles):
    """Create a 3D array of direction cosine matrices (DCMs).


    Args:
        axes: list
            List containing any number of axis numbers (1, 2, 3), corresponding
            to the 1 (x), 2 (y), and 3 (z) axes.
        angles: ndarray
            An (m, n) array where m is the number of DCMs to be produced and n
            is the number of principle rotations making up each DCM (i.e. the
            length of the axes list).

    Returns:
        DCM: ndarray
            (m, 3, 3) array of DCMs. Every element along the 0-axis is a (3, 3)
            DCM. For example, suppose C = euler_angles([1, 2, 3], Phi), where
            Phi is (m, 3). Then the (3, 3) DCM C[j] will be the result of a
            1-2-3 Euler angle rotation through the three angles in the j-th row
            of Phi (Phi[j]).
    """
    def principal_rotation(axis, angle):
        """Create a 3D array of a principal axis DCM.

        Args:
            axis: int
                Axis about which these DCMs rotate.
            angle: ndarray
                (m, 1, 1) array of angles, where m is the number of DCMs to be
                created.

        Returns:
            C: ndarray
                (m, 3, 3) array of DCMs.
        """
        dims = (angle.shape[0], 1, 1)
        zero = np.zeros(dims)
        one = np.ones(dims)
        cosx = np.cos(angle).reshape(dims)
        sinx = np.sin(angle).reshape(dims)

        if axis == 1:
            return np.concatenate((
                np.concatenate((one, zero, zero), axis=2),
                np.concatenate((zero, cosx, sinx), axis=2),
                np.concatenate((zero, -sinx, cosx), axis=2)
                 ), axis=1)

        elif axis == 2:
            return np.concatenate((
                np.concatenate((cosx, zero, -sinx), axis=2),
                np.concatenate((zero, one, zero), axis=2),
                np.concatenate((sinx, zero, cosx), axis=2)
                 ), axis=1)

        elif axis == 3:
            return np.concatenate((
                np.concatenate((cosx, sinx, zero), axis=2),
                np.concatenate((-sinx, cosx, zero), axis=2),
                np.concatenate((zero, zero, one), axis=2)
                 ), axis=1)

    DCM = np.tile(np.eye(3), (angles.shape[0], 1, 1))

    for j, axis in enumerate(axes):
        DCM = principal_rotation(axis, angles[:, j:j+1]) @ DCM

    return DCM
