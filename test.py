"""UnitTest classes for testing orbital_elements classes."""
import unittest
import numpy as np
import mcpyi
import orbital_mechanics.orbit as orb
import orbital_mechanics.dynamics as orbdyn
from orbital_elements.pose.euler_angles import euler_angles
import orbital_elements.rv as rv
import orbital_elements.coe as coe
import orbital_elements.coe_m0 as coeM0

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
coeM0_0 = orb.coe2coeM0(coe_0, 0)
rv_0 = orb.coe2rv(coe_0)


class TestRV(unittest.TestCase):

    def test_hamiltonian_constant_solution(self):
        X0 = rv_0

        X = np.tile(X0, (m, 1))
        T = np.zeros((m, 1))

        order = 6
        H = rv.Hamiltonian(order=order)(T, X)
        H_rel = np.abs(H - H[0, 0])

        self.assertTrue(all(H_rel < tol))

    def test_hamiltonian_keplerian_solution(self):
        X0 = rv_0

        T = np.linspace(0, 10, num=m).reshape((m, 1))
        X = rv.KeplerianSolution(X0)(T)

        order = 1
        H = rv.Hamiltonian(order=order)(T, X)
        H_rel = np.abs(H - H[0, 0])

        self.assertTrue(all(H_rel < tol))

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
        H_rel = np.abs(H - H[0, 0])

        self.assertTrue(all(H_rel < tol))

    def test_zonal_solution(self):
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
        H_rel = np.abs(H - H[0, 0])

        self.assertTrue(all(H_rel < tol))

    def test_zonal_round_trip(self):
        X0_forward = rv_0
        order_H = 6
        kep_dyn_forward = rv.KeplerianDynamics(X0_forward)
        zon_grav = rv.ZonalGravity(order=order_H)
        sysfor = orbdyn.SystemDynamics(kep_dyn_forward, perturbations=zon_grav)

        orbits = 13
        segs_per_orbit = 3
        order_mcpi = 60
        segments = orbits * segs_per_orbit
        domains_for = [k*period/segs_per_orbit for k in range(segments+1)]
        seg_number = len(domains_for) - 1
        N = (order_mcpi,) * seg_number
        mcpi_forward = mcpyi.MCPI(sysfor, domains_for, N, 'warm', X0_forward,
                                  tol)

        T_for = np.linspace(0, 10, num=m).reshape((m, 1))
        X_for = mcpi_forward.solve_serial()(T_for)

        X0_bckward = X_for[-1:]
        kep_dyn_bckward = rv.KeplerianDynamics(X0_bckward)
        sysbck = orbdyn.SystemDynamics(kep_dyn_bckward, perturbations=zon_grav)
        domains_bck = [-x for x in domains_for]
        mcpi_bckward = mcpyi.MCPI(sysbck, domains_bck, N, 'warm', X0_bckward,
                                  tol)
        T_bck = np.linspace(0, -10, num=m).reshape((m, 1))
        X_bck = mcpi_bckward.solve_serial()(T_bck)

        self.assertTrue(np.linalg.norm(X0_forward-X_bck[-1]) < 1e-12)


class TestCOE(unittest.TestCase):

    def test_hamiltonian_constant_solution(self):
        X0 = coe_0

        X = np.tile(X0, (m, 1))
        T = np.zeros((m, 1))

        order = 6
        H = coe.Hamiltonian(order=order)(T, X)
        H_rel = np.abs(H - H[0, 0])

        self.assertTrue(all(H_rel < tol))

    def test_hamiltonian_keplerian_solution(self):
        X0 = coe_0

        T = np.linspace(0, 10, num=m).reshape((m, 1))
        X = coe.KeplerianSolution(X0)(T)

        order = 1
        H = coe.Hamiltonian(order=order)(T, X)
        H_rel = np.abs(H - H[0, 0])

        self.assertTrue(all(H_rel < tol))

    def test_compare_keplerian_to_rv(self):
        X0_coe = coe_0
        X0_rv = rv_0

        T = np.linspace(0, 10, num=m).reshape((m, 1))
        X_coe = coe.KeplerianSolution(X0_coe)(T)
        X_rv = rv.KeplerianSolution(X0_rv)(T)

        X_diff_norm = np.linalg.norm(orb.coe2rv(X_coe) - X_rv, ord=2, axis=1)

        self.assertTrue(all(X_diff_norm < tol))

    def test_hamiltonian_dynamics_solution(self):
        X0 = coe_0
        kep_dyn = coe.KeplerianDynamics(X0)

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
        H = coe.Hamiltonian(order=order_H)(T, X)
        H_rel = np.abs(H - H[0, 0])

        self.assertTrue(all(H_rel < tol))

    def test_compare_dynamics_to_rv(self):
        X0_coe = coe_0
        X0_rv = rv_0
        coe_dyn = coe.KeplerianDynamics(X0_coe)
        rv_dyn = rv.KeplerianDynamics(X0_rv)

        orbits = 13
        segs_per_orbit = 3
        order_mcpi = 60
        segments = orbits * segs_per_orbit
        domains = [k*period/segs_per_orbit for k in range(segments+1)]
        seg_number = len(domains) - 1
        N = (order_mcpi,) * seg_number
        mcpi_coe = mcpyi.MCPI(coe_dyn, domains, N, 'warm', X0_coe, tol)
        mcpi_rv = mcpyi.MCPI(rv_dyn, domains, N, 'warm', X0_rv, tol)

        T = np.linspace(0, 10, num=m).reshape((m, 1))
        X_coe = mcpi_coe.solve_serial()(T)
        X_rv = mcpi_rv.solve_serial()(T)

        X_diff_norm = np.linalg.norm(orb.coe2rv(X_coe) - X_rv, ord=2, axis=1)
        diff_tol = 1e-13

        self.assertTrue(all(X_diff_norm < diff_tol))

    def test_zonal_solution(self):
        X0 = coe_0
        order_H = 2
        kep_dyn = coe.KeplerianDynamics(X0)
        zon_grav = coe.ZonalGravity(order=order_H)
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

        H = coe.Hamiltonian(order=order_H)(T, X)
        H_rel = np.abs(H - H[0, 0])
        print(max(H_rel))
        self.assertTrue(all(H_rel < tol))

    def test_compare_zonal_to_rv(self):
        X0_coe = coe_0
        X0_rv = rv_0
        order_H = 2
        kep_dyn_coe = coe.KeplerianDynamics(X0_coe)
        zon_grav_coe = coe.ZonalGravity(order=order_H)
        syscoe = orbdyn.SystemDynamics(kep_dyn_coe, perturbations=zon_grav_coe)
        kep_dyn_rv = rv.KeplerianDynamics(X0_rv)
        zon_grav_rv = rv.ZonalGravity(order=order_H)
        sysrv = orbdyn.SystemDynamics(kep_dyn_rv, perturbations=zon_grav_rv)

        orbits = 13
        segs_per_orbit = 3
        order_mcpi = 60
        segments = orbits * segs_per_orbit
        domains = [k*period/segs_per_orbit for k in range(segments+1)]
        seg_number = len(domains) - 1
        N = (order_mcpi,) * seg_number
        mcpi_coe = mcpyi.MCPI(syscoe, domains, N, 'warm', X0_coe, tol)
        mcpi_rv = mcpyi.MCPI(sysrv, domains, N, 'warm', X0_rv, tol)

        T = np.linspace(0, 10, num=m).reshape((m, 1))
        X_coe = mcpi_coe.solve_serial()(T)
        X_rv = mcpi_rv.solve_serial()(T)

        X_diff_norm = np.linalg.norm(orb.coe2rv(X_coe) - X_rv, ord=2, axis=1)
        diff_tol = 1e-13

        self.assertTrue(all(X_diff_norm < diff_tol))


class TestCOEM0(unittest.TestCase):

    def test_hamiltonian_constant_solution(self):
        X0 = coeM0_0

        X = np.tile(X0, (m, 1))
        T = np.zeros((m, 1))

        order = 6
        H = coeM0.Hamiltonian(order=order)(T, X)
        H_rel = np.abs(H - H[0, 0])

        self.assertTrue(all(H_rel < tol))

    def test_hamiltonian_keplerian_solution(self):
        X0 = coeM0_0

        T = np.linspace(0, 10, num=m).reshape((m, 1))
        X = coeM0.KeplerianSolution(X0)(T)

        order = 1
        H = coeM0.Hamiltonian(order=order)(T, X)
        H_rel = np.abs(H - H[0, 0])

        self.assertTrue(all(H_rel < tol))

    def test_compare_keplerian_to_coe(self):
        X0_coeM0 = coeM0_0
        X0_coe = coe_0

        T = np.linspace(0, 10, num=m).reshape((m, 1))
        X_coeM0 = coeM0.KeplerianSolution(X0_coeM0)(T)
        X_coe = coe.KeplerianSolution(X0_coe)(T)

        np.testing.assert_allclose(X_coe, orb.coeM02coe(X_coeM0, T), rtol=0,
                                   atol=1e-14)

    def test_hamiltonian_dynamics_solution(self):
        X0 = coeM0_0
        kep_dyn = coeM0.KeplerianDynamics(X0)

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
        H = coeM0.Hamiltonian(order=order_H)(T, X)
        H_rel = np.abs(H - H[0, 0])

        self.assertTrue(all(H_rel < tol))

    def test_compare_dynamics_to_rv(self):
        X0_coeM0 = coeM0_0
        X0_rv = rv_0
        coeM0_dyn = coeM0.KeplerianDynamics(X0_coeM0)
        rv_dyn = rv.KeplerianDynamics(X0_rv)

        orbits = 13
        segs_per_orbit = 3
        order_mcpi = 60
        segments = orbits * segs_per_orbit
        domains = [k*period/segs_per_orbit for k in range(segments+1)]
        seg_number = len(domains) - 1
        N = (order_mcpi,) * seg_number
        mcpi_coeM0 = mcpyi.MCPI(coeM0_dyn, domains, N, 'warm', X0_coeM0, tol)
        mcpi_rv = mcpyi.MCPI(rv_dyn, domains, N, 'warm', X0_rv, tol)

        T = np.linspace(0, 10, num=m).reshape((m, 1))
        X_coeM0 = mcpi_coeM0.solve_serial()(T)
        X_rv = mcpi_rv.solve_serial()(T)

        X_diff_norm = np.linalg.norm(orb.coeM02rv(X_coeM0) - X_rv, ord=2, axis=1)
        diff_tol = 1e-13

        self.assertTrue(all(X_diff_norm < diff_tol))

    def test_zonal_solution(self):
        X0 = coeM0_0
        order_H = 2
        kep_dyn = coeM0.KeplerianDynamics(X0)
        zon_grav = coeM0.ZonalGravity(order=order_H)
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

        H = coeM0.Hamiltonian(order=order_H)(T, X)
        H_rel = np.abs(H - H[0, 0])
        print(max(H_rel))
        self.assertTrue(all(H_rel < tol))

    def test_compare_zonal_to_rv(self):
        X0_coeM0 = coeM0_0
        X0_rv = rv_0
        order_H = 2
        kep_dyn_coeM0 = coeM0.KeplerianDynamics(X0_coeM0)
        zon_grav_coeM0 = coeM0.ZonalGravity(order=order_H)
        syscoeM0 = orbdyn.SystemDynamics(kep_dyn_coeM0, perturbations=zon_grav_coeM0)
        kep_dyn_rv = rv.KeplerianDynamics(X0_rv)
        zon_grav_rv = rv.ZonalGravity(order=order_H)
        sysrv = orbdyn.SystemDynamics(kep_dyn_rv, perturbations=zon_grav_rv)

        orbits = 13
        segs_per_orbit = 3
        order_mcpi = 60
        segments = orbits * segs_per_orbit
        domains = [k*period/segs_per_orbit for k in range(segments+1)]
        seg_number = len(domains) - 1
        N = (order_mcpi,) * seg_number
        mcpi_coeM0 = mcpyi.MCPI(syscoeM0, domains, N, 'warm', X0_coeM0, tol)
        mcpi_rv = mcpyi.MCPI(sysrv, domains, N, 'warm', X0_rv, tol)

        T = np.linspace(0, 10, num=m).reshape((m, 1))
        X_coeM0 = mcpi_coeM0.solve_serial()(T)
        X_rv = mcpi_rv.solve_serial()(T)

        X_diff_norm = np.linalg.norm(orb.coeM02rv(X_coe) - X_rv, ord=2, axis=1)
        diff_tol = 1e-13

        self.assertTrue(all(X_diff_norm < diff_tol))


class TestPose(unittest.TestCase):

    def test_single_dcm(self):
        axes = [1]
        angles = np.array([[1]])
        dcm = euler_angles(axes, angles)

        self.assertTrue(dcm.shape == (1, 3, 3))

    def test_313_rotation(self):
        axes = [3, 1, 3]
        angles = np.array([[1, 1.5, 2]])
        dcm = euler_angles(axes, angles)

        self.assertTrue(dcm.shape == (1, 3, 3))
