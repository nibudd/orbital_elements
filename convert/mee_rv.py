import numpy as np

__author__ = "Nathan I. Budd"
__email__ = "nibudd@gmail.com"
__copyright__ = "Copyright 2017, LASR Lab"
__license__ = "MIT"
__version__ = "0.1"
__status__ = "Production"
__date__ = "05 Mar 2017"


def mee_rv(rv, mu=1.):
    """Convert position-velocity elements to modifieid equinoctial elements.

    Args:
        rv: ndarray
            (m, 6) array of position-velocity elements ordered as
            (rx, ry, rz, vx, vy, vz),
            where
            rx = position x-component
            ry = position y-component
            rz = position z-component
            vx = velocity x-component
            vy = velocity y-component
            vz = velocity z-component
        mu: float, optional
            Standard Gravitational Parameter. Defaults to 1.0, the standard
            value in canonical units.

    Returns:
        mee: ndarray
            (m, 6) array of modified equinoctial elements ordered as
            (p, f, g, h, k, L), where
            p = semi-latus rectum
            f = 1-component of eccentricity vector in perifocal frame
            g = 2-component of eccentricity vector in perifocal frame
            h = 1-component of the ascending node vector in the perifocal frame
            k = 2-component of the ascending node vector in the perifocal frame
            L = true longitude
    """
    r = rv[0:, 0:3]
    v = rv[0:, 3:6]
    m = rv.shape[0]

    r_norm = np.linalg.norm(r, ord=2, axis=1, keepdims=True)
    v_norm = np.linalg.norm(v, ord=2, axis=1, keepdims=True)

    # angular momentum
    ang_mom = np.cross(r, v)
    ang_mom_norm = np.linalg.norm(ang_mom, ord=2, axis=1, keepdims=True)
    ang_mom_hat = ang_mom / ang_mom_norm

    # semilatus rectum
    p = ang_mom_norm**2 / mu

    # equinocital 1,2-components of ascending node vector
    h = -ang_mom_hat[0:, 1:2] / (1. + ang_mom_hat[0:, 2:3])
    k = ang_mom_hat[0:, 0:1] / (1. + ang_mom_hat[0:, 2:3])

    # construct direction cosine matrix from equinoctial to ECI frame
    h1 = h.reshape((m, 1, 1))
    k1 = k.reshape((m, 1, 1))
    h2 = h1**2
    k2 = k1**2

    DCM = np.concatenate((
        np.concatenate((1+h2-k2, 2*h1*k1, 2*k1), axis=2),
        np.concatenate((2*h1*k1, 1-h2+k2, -2*h1), axis=2),
        np.concatenate((-2*k1, 2*h1, 1-h2-k2), axis=2),
    ), axis=1) / (1. + h2 + k2)

    # equinoctial 1,2-unit vectors in ECI frame
    one = np.ones((m, 1, 1))
    zero = np.zeros((m, 1, 1))
    f_equ = np.concatenate((one, zero, zero), axis=1)
    g_equ = np.concatenate((zero, one, zero), axis=1)
    f_eci = (DCM @ f_equ)
    g_eci = (DCM @ g_equ)

    # eccentricity vector
    e = np.cross(v, ang_mom)/mu - r/r_norm

    # equinoctial 1,2-components of eccentricity vector
    f = (e.reshape((m, 1, 3)) @ f_eci).reshape((m, 1))
    g = (e.reshape((m, 1, 3)) @ g_eci).reshape((m, 1))

    # true longitude
    cL = (r.reshape((m, 1, 3)) @ f_eci).reshape((m, 1))
    sL = (r.reshape((m, 1, 3)) @ g_eci).reshape((m, 1))
    L = np.mod(np.arctan2(sL, cL), 2*np.pi)

    return np.concatenate((p, f, g, h, k, L), axis=1)
