import numpy as np

__author__ = "Nathan I. Budd"
__email__ = "nibudd@gmail.com"
__copyright__ = "Copyright 2017, LASR Lab"
__license__ = "MIT"
__version__ = "0.1"
__status__ = "Production"
__date__ = "05 Mar 2017"


def rv_mee(mee, mu=1.):
    """Convert modifieid equinoctial elements to position-velocity elements.

    Args:
        mee: ndarray
            (m, 6) array of position-velocity elements ordered as
            (p, f, g, h, k, L), where
            p = semi-latus rectum
            f = 1-component of eccentricity vector in perifocal frame
            g = 2-component of eccentricity vector in perifocal frame
            h = 1-component of the ascending node vector in the perifocal frame
            k = 2-component of the ascending node vector in the perifocal frame
            L = true longitude
        mu: float, optional
            Standard Gravitational Parameter. Defaults to 1.0, the standard
            value in canonical units.

    Returns:
        rv: ndarray
            (m, 6) array of COEs with true anomaly ordered as
            (rx, ry, rz, vx, vy, vz),
            where
            rx = position x-component
            ry = position y-component
            rz = position z-component
            vx = velocity x-component
            vy = velocity y-component
            vz = velocity z-component
    """
    m = mee.shape[0]

    p = mee[0:, 0:1]
    f = mee[0:, 1:2]
    g = mee[0:, 2:3]
    h = mee[0:, 3:4].reshape((m, 1, 1))
    k = mee[0:, 4:5].reshape((m, 1, 1))
    L = mee[0:, 5:6]

    cL = np.cos(L)
    sL = np.sin(L)

    r = p / (1. + f*cL + g*sL)

    # r in equinoctial frame
    zero = np.zeros(p.shape)
    r_equ = r * np.concatenate((cL, sL, zero), 1)

    # v in equinoctial frame
    r_dot = (mu/p)**0.5 * (f*sL - g*cL)
    rL_dot = (mu/p)**0.5 * (1. + f*cL + g*sL)
    v_equ = np.concatenate((r_dot*cL - rL_dot*sL,
                            r_dot*sL + rL_dot*cL,
                            zero), 1)

    # construct direction cosine matrix from equinoctial to ECI frame
    h2 = h**2
    k2 = k**2

    DCM = np.concatenate((
        np.concatenate((1+h2-k2, 2*h*k, 2*k), axis=2),
        np.concatenate((2*h*k, 1-h2+k2, -2*h), axis=2),
        np.concatenate((-2*k, 2*h, 1-h2-k2), axis=2),
    ), axis=1) / (1. + h2 + k2)

    # represent r and v in ECI frame
    r_eci = (DCM @ r_equ.reshape((m, 3, 1))).reshape((m, 3))
    v_eci = (DCM @ v_equ.reshape((m, 3, 1))).reshape((m, 3))

    return np.concatenate((r_eci, v_eci), axis=1)
