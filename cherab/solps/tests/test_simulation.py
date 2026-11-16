import unittest

import pickle
import numpy as np
from numpy.random import default_rng

from cherab.solps import SOLPSSimulation
from cherab.solps.tests.test_mesh import create_test_mesh


def create_test_simulation():
    # Creating a test SOLPSSimulation
    mesh, rho, beta, r0 = create_test_mesh()
    nx = mesh.nx
    ny = mesh.ny

    species_list = [('hydrogen', 0), ('hydrogen', 1)]
    sim = SOLPSSimulation(mesh, species_list)
    rng = default_rng(seed=42)
    b0 = 1.
    b_field = np.zeros((3, ny, nx))
    b_field[2] = b0 * r0 / (r0 + 0.5 * (rho[:-1] + rho[1:])[:, None] * np.cos(0.5 * (beta[:-1] + beta[1:]))[None, :])
    b_field[0] = 0.01 * (rng.random((ny, nx)) - 0.5)
    b_field[1] = 0.01 * (rng.random((ny, nx)) - 0.5)
    sim.b_field = b_field
    sim.electron_temperature = 1.e3 * (1 - (rho[:-1, None] - rho[0]) / (rho[-1] - rho[0])) + 1.e2 * (rng.random((ny, nx)) - 0.5)
    sim.electron_density = 1.e19 * (1 - (rho[:-1, None] - rho[0]) / (rho[-1] - rho[0])) + 1.e18 * (rng.random((ny, nx)) - 0.5)
    electron_velocities = 1.e4 * rng.random((3, ny, nx))
    electron_velocities[2] += 1.e5
    sim.electron_velocities = electron_velocities
    sim.ion_temperature = sim.electron_temperature
    sim.neutral_temperature = 0.1 * sim.ion_temperature[None, :, :]
    sim.species_density = np.zeros((2, ny, nx))
    sim.species_density[0] = 0.1 * sim.electron_density[::-1, :]
    sim.species_density[1] = sim.electron_density
    velocities = 1.e4 * rng.random((2, 3, ny, nx))
    velocities[1, 2] -= 1.e5
    sim.velocities = velocities

    return sim


class TestSOLPSSimulation(unittest.TestCase):
    """
    Test SOLPSSimulation.
    """

    sim = create_test_simulation()

    def test_array_like_inputs(self):
        """NumPy 2 must allow lists and arrays requiring dtype conversion."""
        sim = create_test_simulation()
        fields = (
            'electron_temperature', 'ion_temperature', 'neutral_temperature',
            'electron_density', 'electron_velocities', 'electron_velocities_cylindrical',
            'species_density', 'velocities', 'velocities_cylindrical',
            'b_field', 'b_field_cylindrical', 'total_radiation',
            'halpha_mol_radiation', 'halpha_total_radiation',
        )
        for field in fields:
            original = getattr(sim, field)
            shape = original.shape if original is not None else (sim.mesh.ny, sim.mesh.nx)
            values = np.ones(shape, dtype=np.int32)
            for value in (values, values.tolist()):
                with self.subTest(field=field, input_type=type(value).__name__):
                    setattr(sim, field, value)
                    result = getattr(sim, field)
                    self.assertEqual(result.dtype, np.float64)
                    np.testing.assert_array_equal(result, values)

    def test_interpolator_shares_data(self):
        """Editing a supplied float64 array must update its interpolator."""
        sim = create_test_simulation()
        values = np.ones((sim.mesh.ny, sim.mesh.nx), dtype=np.float64)
        sim.electron_density = values
        self.assertIs(sim.electron_density, values)
        values[0, 0] = 42.
        self.assertEqual(sim.electron_density_f2d(sim.mesh.cr[0, 0], sim.mesh.cz[0, 0]), 42.)

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

    def test_pickle(self):
        sim1 = self.sim
        sim2 = pickle.loads(pickle.dumps(sim1))

        test_list = []
        test_list.append(sim1.species_list == sim2.species_list)
        test_list.append(sim1.neutral_list == sim2.neutral_list)

        test_list.append(np.array_equiv(sim1.b_field_cylindrical, sim2.b_field_cylindrical))
        test_list.append(np.array_equiv(sim1.electron_temperature, sim2.electron_temperature))
        test_list.append(np.array_equiv(sim1.ion_temperature, sim2.ion_temperature))
        test_list.append(np.array_equiv(sim1.neutral_temperature, sim2.neutral_temperature))
        test_list.append(np.array_equiv(sim1.electron_density, sim2.electron_density))
        test_list.append(np.array_equiv(sim1.electron_velocities_cylindrical, sim2.electron_velocities_cylindrical))
        test_list.append(np.array_equiv(sim1.species_density, sim2.species_density))
        test_list.append(np.array_equiv(sim1.velocities_cylindrical, sim2.velocities_cylindrical))

        test_list.append(np.allclose(sim1.b_field, sim2.b_field))
        test_list.append(np.allclose(sim1.electron_velocities, sim2.electron_velocities))
        test_list.append(np.allclose(sim1.velocities, sim2.velocities))

        return self.assertTrue(np.all(test_list))


if __name__ == '__main__':
    unittest.main()
