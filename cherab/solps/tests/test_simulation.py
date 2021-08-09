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
        rv = np.linspace(1., 2., 11)
        zv = np.linspace(-1., 1., 21)
        r2d, z2d = np.meshgrid(rv, zv, indexing='ij')
        vol = (np.pi * (rv[1:] + rv[:-1]) * np.diff(rv))[:, None] * np.diff(zv)[None, :]
        ny = rv.size - 1
        nx = zv.size - 1
        r = np.zeros((4, ny, nx))
        z = np.zeros((4, ny, nx))
        r[0] = r2d[:-1, :-1]
        z[0] = z2d[:-1, :-1]
        r[1] = r2d[:-1, 1:]
        z[1] = z2d[:-1, 1:]
        r[2] = r2d[1:, :-1]
        z[2] = z2d[1:, :-1]
        r[3] = r2d[1:, 1:]
        z[3] = z2d[1:, 1:]
        neighbix = -1 * np.ones((4, ny, nx), dtype=int)
        neighbix[0, :, 1:] = np.arange(nx - 1, dtype=int)[None, :]
        neighbix[1, 1:, :] = np.arange(nx, dtype=int)[None, :]
        neighbix[1, :, :-1] = np.arange(1, nx, dtype=int)[None, :]
        neighbix[1, :-1, :] = np.arange(nx, dtype=int)[None, :]
        neighbiy = -1 * np.ones((4, ny, nx), dtype=int)
        neighbiy[0, :, 1:] = np.arange(ny, dtype=int)[:, None]
        neighbiy[1, 1:, :] = np.arange(ny - 1, dtype=int)[:, None]
        neighbiy[1, :, :-1] = np.arange(ny, dtype=int)[:, None]
        neighbiy[1, :-1, :] = np.arange(1, ny, dtype=int)[:, None]
        mesh = SOLPSMesh(r, z, vol, neighbix, neighbiy)

        # Creating a test SOLPSSimulation
        species_list = [('hydrogen', 0), ('hydrogen', 1)]
        self.sim = SOLPSSimulation(mesh, species_list)
        rng = default_rng()
        self.sim.electron_temperature = 1.e3 + 1.e2 * rng.random((ny, nx))
        self.sim.electron_density = 1.e19 + 1.e18 * rng.random((ny, nx))
        electron_velocities = 1.e4 * rng.random((3, ny, nx))
        electron_velocities[2] += 1.e5
        self.sim.electron_velocities = electron_velocities
        self.sim.ion_temperature = self.sim.electron_temperature
        self.sim.neutral_temperature = 0.1 * self.sim.ion_temperature[None, :, :]
        self.sim.species_density = np.zeros((2, ny, nx))
        self.sim.species_density[0] = 0.1 * self.sim.electron_density
        self.sim.species_density[1] = self.sim.electron_density
        velocities = 1.e4 + rng.random((2, 3, ny, nx))
        velocities[1, 2] -= 1.e5
        self.sim.velocities = velocities
        b_field = np.zeros((3, ny, nx))
        b_field[2] = 1. + 0.1 * rng.random((ny, nx))
        self.sim.b_field = b_field

    def test_vector_transform(self):
        """ Tests vector transforms (poloidal <--> cylindrical)"""

        test_list = []
        # save a copy of sim.b_field
        b_field = self.sim.b_field
        # this creates a new array at sim.b_field
        self.sim.b_field_cylindrical = self.sim.b_field_cylindrical
        # check if the new array is identical to the original one (poloidal --> cylindrical --> poloidal)
        test_list.append(np.array_equiv(b_field, self.sim.b_field))

        electron_velocities = self.sim.electron_velocities
        self.sim.electron_velocities_cylindrical = self.sim.electron_velocities_cylindrical
        test_list.append(np.array_equiv(electron_velocities, self.sim.electron_velocities))

        velocities = self.sim.velocities
        self.sim.velocities_cylindrical = self.sim.velocities_cylindrical
        test_list.append(np.array_equiv(velocities, self.sim.velocities))

        return self.assertTrue(np.all(test_list))


if __name__ == '__main__':
    unittest.main()
