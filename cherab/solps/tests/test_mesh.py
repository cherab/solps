import unittest

import pickle
import numpy as np

from cherab.solps import SOLPSMesh


def create_test_mesh():
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

    return mesh, rho, beta, r0


class TestSOLPSMesh(unittest.TestCase):
    """
    Test SOLPSMesh.
    """

    mesh, _, _, _ = create_test_mesh()

    def test_pickle(self):
        """ Tests SOLPSMesh pickling"""

        mesh1 = self.mesh
        mesh2 = pickle.loads(pickle.dumps(mesh1))
        test_list = []
        # test initial parameters
        test_list.append(np.array_equiv(mesh1.r, mesh2.r))
        test_list.append(np.array_equiv(mesh1.z, mesh2.z))
        test_list.append(np.array_equiv(mesh1.vol, mesh2.vol))
        test_list.append(np.array_equiv(mesh1.neighbix, mesh2.neighbix))
        test_list.append(np.array_equiv(mesh1.neighbiy, mesh2.neighbiy))
        # test derivative parameters
        test_list.append(np.array_equiv(mesh1.radial_basis_vector, mesh2.radial_basis_vector))
        test_list.append(np.array_equiv(mesh1.poloidal_basis_vector, mesh2.poloidal_basis_vector))
        test_list.append(np.array_equiv(mesh1.vertex_coordinates, mesh2.vertex_coordinates))
        test_list.append(np.array_equiv(mesh1.quadrangles, mesh2.quadrangles))

        return self.assertTrue(np.all(test_list))


if __name__ == '__main__':
    unittest.main()
