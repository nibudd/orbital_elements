"""
UnitTest classes for testing orbital_elements classes
"""
import unittest
import numpy as np
import orbital_mechanics.orbit as orb
import orbital_elements.rv as rv
import orbital_elements.coe as coe

__author__ = "Nathan I. Budd"
__email__ = "nibudd@gmail.com"
__copyright__ = "Copyright 2017, LASR Lab"
__license__ = "MIT"
__version__ = "0.1"
__status__ = "Production"
__date__ = "02 Mar 2017"


DU = 1/6378.137  # DU/km (Earth Radius)
TU = 1/806.811  # TU/s
m = 1000
tol = 1e-14

coe_0 = np.array([[8000*DU, 0.1, 10*np.pi/180, 0., 0., 0.]])
rv_0 = orb.coe2rv(coe_0)


class TestRV(unittest.TestCase):

    def test_hamiltonian_constant_trajectory(self):
        X0 = rv_0

        X = np.tile(X0, (m, 1))
        T = np.zeros((m, 1))

        order = 6
        H = rv.Hamiltonian(order=order)(T, X)

        self.assertTrue(all(H < tol))

    def test_hamiltonian_keplerian_trajectory(self):
        X0 = rv_0

        T = np.linspace(0, 10, num=m).reshape((m, 1))
        X = rv.KeplerianPath(X0)(T)

        order = 1
        H = rv.Hamiltonian(order=order)(T, X)

        self.assertTrue(all(H < tol))


class TestCOE(unittest.TestCase):

    def test_hamiltonian_constant_trajectory(self):
        X0 = coe_0

        X = np.tile(X0, (m, 1))
        T = np.zeros((m, 1))

        order = 6
        H_rel = coe.Hamiltonian(order=order)(T, X)

        self.assertTrue(all(H_rel < tol))

    def test_hamiltonian_keplerian_trajectory(self):
        X0 = coe_0

        T = np.linspace(0, 10, num=m).reshape((m, 1))
        X = coe.KeplerianPath(X0)(T)

        order = 1
        H = coe.Hamiltonian(order=order)(T, X)

        self.assertTrue(all(H < tol))
