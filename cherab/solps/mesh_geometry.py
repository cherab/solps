
# Copyright 2016-2018 Euratom
# Copyright 2016-2018 United Kingdom Atomic Energy Authority
# Copyright 2016-2018 Centro de Investigaciones Energéticas, Medioambientales y Tecnológicas
#
# Licensed under the EUPL, Version 1.1 or – as soon they will be approved by the
# European Commission - subsequent versions of the EUPL (the "Licence");
# You may not use this work except in compliance with the Licence.
# You may obtain a copy of the Licence at:
#
# https://joinup.ec.europa.eu/software/page/eupl5
#
# Unless required by applicable law or agreed to in writing, software distributed
# under the Licence is distributed on an "AS IS" basis, WITHOUT WARRANTIES OR
# CONDITIONS OF ANY KIND, either express or implied.
#
# See the Licence for the specific language governing permissions and limitations
# under the Licence.

# External imports
# from collections import namedtuple

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
from matplotlib.collections import PatchCollection
from raysect.core.math.function.float import Discrete2DMesh

# INFINITY = 1E99


class SOLPSMesh:
    """
    SOLPSMesh geometry object.

    The SOLPS mesh is rectangular. Each mesh cell is denoted by four vertices with one centre point. Vertices
    may be shared with neighbouring cells. The centre points should be unique.

    Raysect's mesh interpolator uses a different mesh scheme. Mesh cells are triangles and data values are stored at the
    triangle vertices. Therefore, each SOLPS rectangular cell is split into two triangular cells. The data points are
    later interpolated onto the vertex points.

    :param ndarray r: Array of cell vertex r coordinates, must be 3 dimensional. Example shape is (98 x 32 x 4).
    :param ndarray z: Array of cell vertex z coordinates, must be 3 dimensional. Example shape is (98 x 32 x 4).
    :param ndarray vol: Array of cell volumes. Example shape is (98 x 32).
    """

    def __init__(self, r, z, vol):

        if r.shape != z.shape:
            raise ValueError('Shape of r array: %s mismatch the shape of z array: %s.' % (r.shape, z.shape))

        if vol.shape != r.shape[:-1]:
            raise ValueError('Shape of vol array: %s mismatch the grid dimentions: %s.' % (vol.shape, r.shape[:-1]))

        self._cr = r.sum(2) / 4.
        self._cz = z.sum(2) / 4.

        self._nx = r.shape[0]
        self._ny = r.shape[1]

        self._r = r
        self._z = z

        self._vol = vol

        self.vessel = None

        # Calculating parallel basis vector
        self._parallel_basis_vector = np.zeros((self._nx, self._ny, 2))
        vec_r = r[:, :, 1] - r[:, :, 0] + r[:, :, 3] - r[:, :, 2]
        vec_z = z[:, :, 1] - z[:, :, 0] + z[:, :, 3] - z[:, :, 2]
        vec_magn = np.sqrt(vec_r**2 + vec_z**2)
        self._parallel_basis_vector[:, :, 0] = vec_r / vec_magn
        self._parallel_basis_vector[:, :, 1] = vec_z / vec_magn

        # Calculating radial basis vector
        self._radial_basis_vector = np.zeros((self._nx, self._ny, 2))
        vec_r = r[:, :, 2] - r[:, :, 0] + r[:, :, 3] - r[:, :, 1]
        vec_z = z[:, :, 2] - z[:, :, 0] + z[:, :, 3] - z[:, :, 1]
        vec_magn = np.sqrt(vec_r**2 + vec_z**2)
        self._radial_basis_vector[:, :, 0] = vec_r / vec_magn
        self._radial_basis_vector[:, :, 1] = vec_z / vec_magn

        # Test for basis vector calculation
        # plt.quiver(self._cr[:, 0], self._cz[:, 0], self._radial_basis_vector[:, 0, 0], self._radial_basis_vector[:, 0, 1], color='k')
        # plt.quiver(self._cr[:, 0], self._cz[:, 0], self._parallel_basis_vector[:, 0, 0], self._parallel_basis_vector[:, 0, 1], color='r')
        # plt.quiver(self._cr[:, -1], self._cz[:, -1], self._radial_basis_vector[:, -1, 0], self._radial_basis_vector[:, -1, 1], color='k')
        # plt.quiver(self._cr[:, -1], self._cz[:, -1], self._parallel_basis_vector[:, -1, 0], self._parallel_basis_vector[:, -1, 1], color='r')
        # plt.gca().set_aspect('equal')
        # plt.show()

        # Finding unique vertices
        vertices = np.array([r.flatten(), z.flatten()]).T
        self._vertex_coords, unique_vertices = np.unique(vertices, axis=0, return_inverse=True)
        self._num_vertices = self._vertex_coords.shape[0]

        # Work out the extent of the mesh.
        self._mesh_extent = {"minr": r.min(), "maxr": r.max(), "minz": z.min(), "maxz": z.max()}

        # Number of triangles must be equal to number of rectangle centre points times 2.
        self._num_tris = self._nx * self._ny * 2
        self._triangles = np.zeros((self._num_tris, 3), dtype=np.int32)
        self._triangle_to_grid_map = np.zeros((self._num_tris, 2), dtype=np.int32)

        # Pull out the index number for each unique vertex in this rectangular cell.
        # Unusual vertex indexing is based on SOLPS output, see Matlab code extract from David Moulton.
        self._triangles[0::2, 0] = unique_vertices[0::4]
        self._triangles[0::2, 1] = unique_vertices[2::4]
        self._triangles[0::2, 2] = unique_vertices[3::4]
        # Split the quad cell into two triangular cells.
        self._triangles[1::2, 0] = unique_vertices[3::4]
        self._triangles[1::2, 1] = unique_vertices[1::4]
        self._triangles[1::2, 2] = unique_vertices[0::4]

        # Each triangle cell is mapped to the tuple ID (ix, iy) of its parent mesh cell.
        xm, ym = np.meshgrid(np.arange(self._nx, dtype=np.int32), np.arange(self._ny, dtype=np.int32), indexing='ij')
        self._triangle_to_grid_map[::2, 0] = xm.flatten()
        self._triangle_to_grid_map[::2, 1] = ym.flatten()
        self._triangle_to_grid_map[1::2, :] = self._triangle_to_grid_map[::2, :]

        tri_indices = np.arange(self._num_tris, dtype=np.int32)
        self._tri_index_loopup = Discrete2DMesh(self._vertex_coords, self._triangles, tri_indices)

    @property
    def nx(self):
        """Number of grid cells in the x direction."""
        return self._nx

    @property
    def ny(self):
        """Number of grid cells in the y direction."""
        return self._ny

    @property
    def cr(self):
        """R-coordinate of the cell centres."""
        return self._cr

    @property
    def cz(self):
        """Z-coordinate of the cell centres."""
        return self._cz

    @property
    def r(self):
        """R-coordinates of the cell vertices."""
        return self._r

    @property
    def z(self):
        """Z-coordinate of the cell vertices."""
        return self._z

    @property
    def vol(self):
        """Volume/area of each grid cell."""
        return self._vol

    @property
    def vertex_coordinates(self):
        """RZ-coordinates of unique vertices."""
        return self._vertex_coords

    @property
    def num_vertices(self):
        """Total number of unique vertices."""
        return self._num_vertices

    @property
    def mesh_extent(self):
        """Extent of the mesh. A dictionary with minr, maxr, minz and maxz keys."""
        return self._mesh_extent

    @property
    def num_triangles(self):
        """Total number of triangles (the number of cells doubled)."""
        return self._num_tris

    @property
    def triangles(self):
        """Array of triangle vertex indices with (num_thiangles, 3) shape."""
        return self._triangles

    @property
    def parallel_basis_vector(self):
        """
        Array of 2D parallel basis vectors for grid cells.

        For each cell there is a parallel and radial basis vector.

        Any vector on the poloidal grid can be converted to cartesian with the following transformation.
        bx = (p_x  r_x) ( b_p )
        by   (p_y  r_y) ( b_r )

        :return: ndarray with shape (nx, ny, 2).
        """
        return self._parallel_basis_vector

    @property
    def radial_basis_vector(self):
        """
        Array of 2D radial basis vectors for grid cells.

        For each cell there is a parallel and radial basis vector.

        Any vector on the poloidal grid can be converted to cartesian with the following transformation.
        bx = (p_x  r_x) ( b_p )
        by   (p_y  r_y) ( b_r )

        :return: ndarray with shape (nx, ny, 2).
        """
        return self._radial_basis_vector

    @property
    def triangle_to_grid_map(self):
        """
        Array mapping every triangle index to a tuple grid cell ID, i.e. (ix, iy).

        :return: ndarray with shape (nx*ny*2, 2)
        """
        return self._triangle_to_grid_map

    @property
    def triangle_index_lookup(self):
        """
        Discrete2DMesh instance that looks up a triangle index at any 2D point.

        Useful for mapping from a 2D point -> triangle cell -> parent SOLPS mesh cell

        :return: Discrete2DMesh instance
        """
        return self._tri_index_loopup

    def __getstate__(self):
        state = {
            'r': self._r,
            'z': self._z,
            'vol': self._vol
        }
        return state

    def plot_mesh(self):
        """
        Plot the mesh grid geometry to a matplotlib figure.
        """
        fig, ax = plt.subplots()
        patches = []
        for triangle in self.triangles:
            vertices = self.vertex_coordinates[triangle]
            patches.append(Polygon(vertices, closed=True))
        p = PatchCollection(patches, facecolors='none', edgecolors='b')
        ax.add_collection(p)
        ax.axis('equal')

        # Code for plotting vessel geometry if available
        # if self.vessel is not None:
        #     for i in range(self.vessel.shape[0]):
        #         ax.plot([self.vessel[i, 0], self.vessel[i, 2]], [self.vessel[i, 1], self.vessel[i, 3]], 'k')
        #     for i in range(self.vessel.shape[0]):
        #         ax.plot([self.vessel[i, 0], self.vessel[i, 2]], [self.vessel[i, 1], self.vessel[i, 3]], 'or')

        return ax
