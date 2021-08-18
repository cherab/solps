import unittest

import numpy as np
from numpy.random import default_rng

from cherab.solps import SOLPSMesh, SOLPSSimulation


class TestSOLPSSimulation(unittest.TestCase):
    """
    Test SOLPSSimulation.
    """

    def setUp(self):
        # Creating a test SOLPSMesh
        # closed magnetic surfaces only with circular cross section
        r0 = 2.  # major radius
        beta = np.linspace(0, 2 * np.pi, 19)  # poloidal angle
        rho = np.linspace(0.5, 1., 6)  # magnetic surface radius
        # cell vertices
        r2d = r0 + rho[:, None] * np.cos(beta)[None, :]
        z2d = rho[:, None] * np.sin(beta)[None, :]
        # exact cell volume
        vol = (np.pi * r0 * (rho[1:]**2 - rho[:-1]**2)[:, None] * (beta[1:] - beta[:-1])[None, :] +
               2. / 3. * np.pi * (rho[1:]**3 - rho[:-1]**3)[:, None] * (np.sin(beta[1:]) - np.sin(beta[:-1]))[None, :])
        ny = rho.size - 1
        nx = beta.size - 1
        r = np.zeros((4, ny, nx))
        z = np.zeros((4, ny, nx))
        r[0] = r2d[:-1, :-1]
        z[0] = z2d[:-1, :-1]
        r[1] = r2d[:-1, 1:]  # poloidal + 1
        z[1] = z2d[:-1, 1:]  # poloidal + 1
        r[2] = r2d[1:, :-1]  # radial + 1
        z[2] = z2d[1:, :-1]  # radial + 1
        r[3] = r2d[1:, 1:]  # poloidal + 1, radial + 1
        z[3] = z2d[1:, 1:]  # poloidal + 1, radial + 1
        neighbix = -1 * np.ones((4, ny, nx), dtype=int)
        neighbix[0, 1:, :] = np.arange(nx, dtype=int)[None, :]  # left
        neighbix[1, :, 1:] = np.arange(nx - 1, dtype=int)[None, :]  # bottom
        neighbix[2, :-1, :] = np.arange(nx, dtype=int)[None, :]  # right
        neighbix[3, :, :-1] = np.arange(1, nx, dtype=int)[None, :]  # top
        neighbix[1, :, 0] = nx - 1  # closing the surfaces
        neighbix[3, :, -1] = 0  # closing the surfaces
        neighbiy = -1 * np.ones((4, ny, nx), dtype=int)
        neighbiy[0, 1:, :] = np.arange(ny - 1, dtype=int)[:, None]  # left
        neighbiy[1, :, :] = np.arange(ny, dtype=int)[:, None]  # bottom
        neighbiy[2, :-1, :] = np.arange(1, ny, dtype=int)[:, None]  # right
        neighbiy[3, :, :] = np.arange(ny, dtype=int)[:, None]  # top
        mesh = SOLPSMesh(r, z, vol, neighbix, neighbiy)

        # Creating a test SOLPSSimulation
        species_list = [('hydrogen', 0), ('hydrogen', 1)]
        self.sim = SOLPSSimulation(mesh, species_list)
        rng = default_rng(seed=42)
        b0 = 1.
        b_field = np.zeros((3, ny, nx))
        b_field[2] = b0 * r0 / (r0 + 0.5 * (rho[:-1] + rho[1:])[:, None] * np.cos(0.5 * (beta[:-1] + beta[1:]))[None, :])
        b_field[0] = 0.01 * (rng.random((ny, nx)) - 0.5)
        b_field[1] = 0.01 * (rng.random((ny, nx)) - 0.5)
        self.sim.b_field = b_field
        self.sim.electron_temperature = 1.e3 * (1 - (rho[:-1, None] - rho[0]) / (rho[-1] - rho[0])) + 1.e2 * (rng.random((ny, nx)) - 0.5)
        self.sim.electron_density = 1.e19 * (1 - (rho[:-1, None] - rho[0]) / (rho[-1] - rho[0])) + 1.e18 * (rng.random((ny, nx)) - 0.5)
        electron_velocities = 1.e4 * rng.random((3, ny, nx))
        electron_velocities[2] += 1.e5
        self.sim.electron_velocities = electron_velocities
        self.sim.ion_temperature = self.sim.electron_temperature
        self.sim.neutral_temperature = 0.1 * self.sim.ion_temperature[None, :, :]
        self.sim.species_density = np.zeros((2, ny, nx))
        self.sim.species_density[0] = 0.1 * self.sim.electron_density[::-1, :]
        self.sim.species_density[1] = self.sim.electron_density
        velocities = 1.e4 * rng.random((2, 3, ny, nx))
        velocities[1, 2] -= 1.e5
        self.sim.velocities = velocities

    def test_vector_transform(self):
        """ Tests vector transforms (poloidal <--> cylindrical)"""

        test_list = []
        # save a copy of sim.b_field
        b_field = self.sim.b_field
        # this creates a new array at sim.b_field
        self.sim.b_field_cylindrical = self.sim.b_field_cylindrical
        # check if the new array is identical to the original one (poloidal --> cylindrical --> poloidal)
        test_list.append(np.allclose(b_field, self.sim.b_field))

        electron_velocities = self.sim.electron_velocities
        self.sim.electron_velocities_cylindrical = self.sim.electron_velocities_cylindrical
        test_list.append(np.allclose(electron_velocities, self.sim.electron_velocities))

        velocities = self.sim.velocities
        self.sim.velocities_cylindrical = self.sim.velocities_cylindrical
        test_list.append(np.allclose(velocities, self.sim.velocities))

        return self.assertTrue(np.all(test_list))


if __name__ == '__main__':
    unittest.main()
