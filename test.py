"""UnitTest classes for testing orbital_elements classes."""
import unittest
import numpy as np
import mcpyi
import orbital_mechanics.orbit as orb
import orbital_mechanics.dynamics as orbdyn
import orbital_elements.pose as pose
import orbital_elements.convert as convert
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

# INITIAL CONDITIONS
mu = 1.0
a_0 = 8000. * DU
e_0 = 0.1
i_0 = 10.*np.pi/180.
W_0 = 0.*np.pi/180.
w_0 = 0.*np.pi/180.
f_0 = 0.*np.pi/180.
period = 2.*np.pi*(a_0**3 / mu)**.5

coe_0 = np.array([[a_0, e_0, i_0, W_0, w_0, f_0]])
rv_0 = orb.coe2rv(coe_0)

T = np.linspace(0, 10, num=m).reshape((m, 1))
coe_f = coe.KeplerianSolution(coe_0)(T)


class TestRV(unittest.TestCase):

    def test_hamiltonian_constant_solution(self):
        X0 = rv_0

        X = np.tile(X0, (m, 1))
        T = np.zeros((m, 1))

        order = 6
        H = rv.Hamiltonian(order=order)(T, X)

        np.testing.assert_allclose(H[0, 0], H, rtol=tol)

    def test_hamiltonian_keplerian_solution(self):
        X0 = rv_0

        X = rv.KeplerianSolution(X0)(T)

        order = 1
        H = rv.Hamiltonian(order=order)(T, X)

        np.testing.assert_allclose(H[0, 0], H, rtol=tol)

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

        X = mcpi.solve_serial()(T)

        order_H = 1
        H = rv.Hamiltonian(order=order_H)(T, X)

        np.testing.assert_allclose(H[0, 0], H, rtol=tol)

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

        X = mcpi.solve_serial()(T)

        H = rv.Hamiltonian(order=order_H)(T, X)

        np.testing.assert_allclose(H[0, 0], H, rtol=tol)

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

        T_for = T
        X_for = mcpi_forward.solve_serial()(T_for)

        X0_bckward = X_for[-1:]
        kep_dyn_bckward = rv.KeplerianDynamics(X0_bckward)
        sysbck = orbdyn.SystemDynamics(kep_dyn_bckward, perturbations=zon_grav)
        domains_bck = [-x for x in domains_for]
        mcpi_bckward = mcpyi.MCPI(sysbck, domains_bck, N, 'warm', X0_bckward,
                                  tol)
        T_bck = np.linspace(0, -10, num=m).reshape((m, 1))
        X_bck = mcpi_bckward.solve_serial()(T_bck)

        np.testing.assert_allclose(X0_forward, X_bck[-1:], rtol=0, atol=1e-13)


class TestCOE(unittest.TestCase):

    def test_hamiltonian_constant_solution(self):
        X0 = coe_0

        X = np.tile(X0, (m, 1))
        T = np.zeros((m, 1))

        order = 6
        H = coe.Hamiltonian(order=order)(T, X)

        np.testing.assert_allclose(H[0, 0], H, rtol=tol)

    def test_hamiltonian_keplerian_solution(self):
        X0 = coe_0

        X = coe.KeplerianSolution(X0)(T)

        order = 1
        H = coe.Hamiltonian(order=order)(T, X)

        np.testing.assert_allclose(H[0, 0], H, rtol=tol)

    def test_compare_keplerian_to_rv(self):
        X0_coe = coe_0
        X0_rv = rv_0

        X_coe = coe.KeplerianSolution(X0_coe)(T)
        X_rv = rv.KeplerianSolution(X0_rv)(T)

        np.testing.assert_allclose(X_rv, orb.coe2rv(X_coe), rtol=0, atol=tol)

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

        X = mcpi.solve_serial()(T)

        order_H = 1
        H = coe.Hamiltonian(order=order_H)(T, X)

        np.testing.assert_allclose(H[0, 0], H, rtol=tol)

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

        X_coe = mcpi_coe.solve_serial()(T)
        X_rv = mcpi_rv.solve_serial()(T)

        np.testing.assert_allclose(X_rv, orb.coe2rv(X_coe), rtol=0, atol=1e-13)

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

        X = mcpi.solve_serial()(T)

        H = coe.Hamiltonian(order=order_H)(T, X)

        np.testing.assert_allclose(H[0, 0], H, rtol=tol)

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

        X_coe = mcpi_coe.solve_serial()(T)
        X_rv = mcpi_rv.solve_serial()(T)

        np.testing.assert_allclose(X_rv, orb.coe2rv(X_coe), rtol=0, atol=1e-13)


class TestPose(unittest.TestCase):

    def test_single_dcm(self):
        axes = [1]
        angles = np.array([[1]])
        dcm = pose.euler_angles(axes, angles)

        self.assertEqual(dcm.shape, (1, 3, 3))

    def test_313_rotation(self):
        axes = [3, 1, 3]
        angles = np.array([[1, 1.5, 2]])
        dcm = pose.euler_angles(axes, angles)

        self.assertEqual(dcm.shape, (1, 3, 3))


class TestConvert(unittest.TestCase):

    def test_coef_coeE_coef(self):
        coef = convert.mod_angles(coe_f)
        coeE = convert.coeE_coef(coef)
        coef2 = convert.mod_angles(convert.coef_coeE(coeE))
        diff = convert.mod_angles(np.abs(coef-coef2), angle_indices=[0])
        indices_2pi = np.where(2*np.pi-tol < diff)
        diff[indices_2pi] -= 2*np.pi

        np.testing.assert_allclose(diff, 0., rtol=0, atol=tol)

    def test_coeE_coef_coeE(self):
        coeE = convert.mod_angles(convert.coeE_coef(coe_f))
        coef = convert.coef_coeE(coeE)
        coeE2 = convert.mod_angles(convert.coeE_coef(coef))
        diff = convert.mod_angles(np.abs(coeE-coeE2), angle_indices=[0])
        indices_2pi = np.where(2*np.pi-tol < diff)
        diff[indices_2pi] -= 2*np.pi

        np.testing.assert_allclose(diff, 0., rtol=0, atol=tol)

    def test_coeE_coeM_coeE(self):
        coeE = convert.mod_angles(convert.coeE_coef(coe_f))
        coeM = convert.coeM_coeE(coeE)
        coeE2 = convert.mod_angles(convert.coeE_coeM(coeM))
        diff = convert.mod_angles(np.abs(coeE-coeE2), angle_indices=[0])
        indices_2pi = np.where(2*np.pi-tol < diff)
        diff[indices_2pi] -= 2*np.pi

        np.testing.assert_allclose(diff, 0., rtol=0, atol=tol)

    def test_coeM_coeE_coeM(self):
        coeE = convert.coeE_coef(coe_f)
        coeM = convert.mod_angles(convert.coeM_coeE(coeE))
        coeE = convert.coeE_coeM(coeM)
        coeM2 = convert.mod_angles(convert.coeM_coeE(coeE))
        diff = convert.mod_angles(np.abs(coeM-coeM2), angle_indices=[0])
        indices_2pi = np.where(2*np.pi-tol < diff)
        diff[indices_2pi] -= 2*np.pi

        np.testing.assert_allclose(diff, 0., rtol=0, atol=tol)

    def test_coef_coeM_coef(self):
        coef = convert.mod_angles(coe_f)
        coeM = convert.coeM_coef(coef)
        coef2 = convert.mod_angles(convert.coef_coeM(coeM))
        diff = convert.mod_angles(np.abs(coef-coef2), angle_indices=[0])
        indices_2pi = np.where(2*np.pi-tol < diff)
        diff[indices_2pi] -= 2*np.pi

        np.testing.assert_allclose(diff, 0., rtol=0, atol=tol)

    def test_coeM_coef_coeM(self):
        coeM = convert.mod_angles(convert.coeM_coef(coe_f))
        coef = convert.coef_coeM(coeM)
        coeM2 = convert.mod_angles(convert.coeM_coef(coef))
        diff = convert.mod_angles(np.abs(coeM-coeM2), angle_indices=[0])
        indices_2pi = np.where(2*np.pi-tol < diff)
        diff[indices_2pi] -= 2*np.pi

        np.testing.assert_allclose(diff, 0., rtol=0, atol=tol)

    def test_coe_mee_coe(self):
        coe = convert.mod_angles(coe_f)
        mee = convert.mee_coe(coe)
        coe2 = convert.mod_angles(convert.coe_mee(mee))
        diff = convert.mod_angles(np.abs(coe-coe2), angle_indices=[2, 3, 4, 5])
        indices_2pi = np.where(2*np.pi-tol < diff)
        diff[indices_2pi] -= 2*np.pi

        np.testing.assert_allclose(diff, 0., rtol=0, atol=tol)

    def test_mee_coe_mee(self):
        mee = convert.mod_angles(convert.mee_coe(coe_f))
        coe = convert.coe_mee(mee)
        mee2 = convert.mod_angles(convert.mee_coe(coe))
        diff = convert.mod_angles(np.abs(mee-mee2), angle_indices=[5])
        indices_2pi = np.where(2*np.pi-tol < diff)
        diff[indices_2pi] -= 2*np.pi

        np.testing.assert_allclose(diff, 0., rtol=0, atol=tol)

    def test_mee_rv_mee(self):
        mee = convert.mod_angles(convert.mee_coe(coe_f))
        rv = convert.rv_mee(mee)
        mee2 = convert.mod_angles(convert.mee_rv(rv))

        np.testing.assert_allclose(mee, mee2, rtol=0, atol=tol)

    def test_rv_mee_rv(self):
        mee = convert.mod_angles(convert.mee_coe(coe_f))
        rv = convert.rv_mee(mee)
        mee = convert.mee_rv(rv)
        rv2 = convert.rv_mee(mee)

        np.testing.assert_allclose(rv, rv2, rtol=0, atol=tol)

    def test_coe_rv_coe(self):
        coe = coe_f
        rv = convert.rv_coe(coe)
        coe2 = convert.coe_rv(rv)
        diff = convert.mod_angles(np.abs(coe-coe2), angle_indices=[2, 3, 4, 5])
        indices_2pi = np.where(2*np.pi-tol < diff)
        diff[indices_2pi] -= 2*np.pi

        np.testing.assert_allclose(diff, 0., rtol=0, atol=tol)

    def test_rv_coe_rv(self):
        rv = convert.rv_coe(coe_f)
        coe = convert.coe_rv(rv)
        rv2 = convert.rv_coe(coe)

        np.testing.assert_allclose(rv, rv2, rtol=0, atol=tol)
