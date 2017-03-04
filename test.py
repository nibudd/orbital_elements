"""
UnitTest classes for testing orbital_elements classes
"""
import unittest
import numpy as np
import mcpyi
import orbital_mechanics.orbit as orb
import orbital_elements.rv as rv
import orbital_elements.coe as coe
import orbital_mechanics.dynamics as orbdyn

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

mu = 1.0
a_0 = 8000. * DU
e_0 = 0.1
i_0 = 10.*np.pi/180.
W_0 = 0.*np.pi/180.
w_0 = 0.*np.pi/180.
M0_0 = 0.*np.pi/180.
period = 2.*np.pi*(a_0**3 / mu)**.5
coe_0 = np.array([[a_0, e_0, i_0, W_0, w_0, M0_0]])
rv_0 = orb.coe2rv(coe_0)


class TestRV(unittest.TestCase):

    def test_hamiltonian_constant_solution(self):
        X0 = rv_0

        X = np.tile(X0, (m, 1))
        T = np.zeros((m, 1))

        order = 6
        H = rv.Hamiltonian(order=order)(T, X)

        self.assertTrue(all(H < tol))

    def test_hamiltonian_keplerian_solution(self):
        X0 = rv_0

        T = np.linspace(0, 10, num=m).reshape((m, 1))
        X = rv.KeplerianSolution(X0)(T)

        order = 1
        H = rv.Hamiltonian(order=order)(T, X)

        self.assertTrue(all(H < tol))

    def test_hamiltonian_dynamics_solution(self):
        X0 = rv_0
        kep_dyn = rv.KeplerianDynamics(X0)

        orbits = 13
        segs_per_orbit = 3
        order_mcpi = 60
        segments = orbits * segs_per_orbit
        domains = [k*period/segs_per_orbit for k in range(segments+1)]
        seg_number = len(domains) - 1
        N = (order_mcpi,) * seg_number
        mcpi = mcpyi.MCPI(kep_dyn, domains, N, 'warm', X0, tol)

        T = np.linspace(0, 10, num=m).reshape((m, 1))
        X = mcpi.solve_serial()(T)

        order_H = 1
        H = rv.Hamiltonian(order=order_H)(T, X)

        self.assertTrue(all(H < tol))

    def test_zonal_perturbed_solution(self):
        X0 = rv_0
        order_H = 6
        kep_dyn = rv.KeplerianDynamics(X0)
        zon_grav = rv.ZonalGravity(order=order_H)
        system = orbdyn.SystemDynamics(kep_dyn, perturbations=zon_grav)

        orbits = 13
        segs_per_orbit = 3
        order_mcpi = 60
        segments = orbits * segs_per_orbit
        domains = [k*period/segs_per_orbit for k in range(segments+1)]
        seg_number = len(domains) - 1
        N = (order_mcpi,) * seg_number
        mcpi = mcpyi.MCPI(system, domains, N, 'warm', X0, tol)

        T = np.linspace(0, 10, num=m).reshape((m, 1))
        X = mcpi.solve_serial()(T)

        H = rv.Hamiltonian(order=order_H)(T, X)

        self.assertTrue(all(H < tol))


class TestCOE(unittest.TestCase):

    def test_hamiltonian_constant_solution(self):
        X0 = coe_0

        X = np.tile(X0, (m, 1))
        T = np.zeros((m, 1))

        order = 6
        H_rel = coe.Hamiltonian(order=order)(T, X)

        self.assertTrue(all(H_rel < tol))

    def test_hamiltonian_keplerian_solution(self):
        X0 = coe_0

        T = np.linspace(0, 10, num=m).reshape((m, 1))
        X = coe.KeplerianSolution(X0)(T)

        order = 1
        H = coe.Hamiltonian(order=order)(T, X)

        self.assertTrue(all(H < tol))

    def test_compare_keplerian_to_rv(self):
        X0_coe = coe_0
        X0_rv = rv_0

        T = np.linspace(0, 10, num=m).reshape((m, 1))
        X_coe = coe.KeplerianSolution(X0_coe)(T)
        X_rv = rv.KeplerianSolution(X0_rv)(T)

        X_diff_norm = np.linalg.norm(orb.coe2rv(X_coe) - X_rv, ord=2, axis=1)

        self.assertTrue(all(X_diff_norm < tol))
