
# Copyright 2016-2021 Euratom
# Copyright 2016-2021 United Kingdom Atomic Energy Authority
# Copyright 2016-2021 Centro de Investigaciones Energéticas, Medioambientales y Tecnológicas
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
from warnings import warn

import numpy as np


class SOLPSMesh:
    """
    SOLPSMesh geometry object.

    The SOLPS mesh is quadrilateral. Each mesh cell is denoted by four vertices with one centre point. Vertices
    may be shared with neighbouring cells. The centre points should be unique.

    Raysect's mesh interpolator uses a different mesh scheme. Mesh cells are triangles and data values are stored at the
    triangle vertices. Therefore, each SOLPS quadrilateral cell is split into two triangular cells. The data points are
    later interpolated onto the vertex points.

    :param ndarray r: Array of cell vertex r coordinates, must be 3 dimensional. Example shape is (4 x 32 x 98).
    :param ndarray z: Array of cell vertex z coordinates, must be 3 dimensional. Example shape is (4 x 32 x 98).
    :param ndarray vol: Array of cell volumes in m-3. Example shape is (32 x 98).
    :param ndarray neighbix: Array of poloidal indeces of neighbouring cells in order: left, bottom, right, top,
                             must be 3 dimensional. Example shape is (4 x 32 x 98).
                             In SOLPS notation: left/right - poloidal prev./next, bottom/top - radial prev./next.
                             Cell indexing starts with 0 and -1 means no neighbour.
    :param ndarray neighbiy: Array of radial indeces of neighbouring cells in order: left, bottom, right, top,
                             must be 3 dimensional. Example shape is (4 x 32 x 98).
    """

    # TODO Make neighbix and neighbix optional in the future, as they can be reconstructed with _triangle_to_grid_map

    def __init__(self, r, z, vol, neighbix, neighbiy):

        if r.shape != z.shape:
            raise ValueError('Shape of r array: {0} mismatch the shape of z array: {1}.'.format(r.shape, z.shape))

        if vol.shape != r.shape[1:]:
            raise ValueError('Shape of vol array: {0} mismatch the grid dimentions: {1}.'.format(vol.shape, r.shape[1:]))

        if neighbix.shape != r.shape:
            raise ValueError('Shape of neighbix array must be {0}, but it is  {1}.'.format(r.shape, neighbix.shape))

        if neighbiy.shape != r.shape:
            raise ValueError('Shape of neighbix array must be {0}, but it is  {}.'.format(r.shape, neighbiy.shape))

        self._r = r
        self._z = z

        self._vol = vol

        self._neighbix = neighbix.astype(int)
        self._neighbiy = neighbiy.astype(int)

        self._initial_setup()

    def _initial_setup(self):

        r = self._r
        z = self._z

        self._cr = r.sum(0) / 4.
        self._cz = z.sum(0) / 4.

        self._nx = r.shape[2]  # poloidal
        self._ny = r.shape[1]  # radial

        self.vessel = None

        # Calculating poloidal basis vector
        self._poloidal_basis_vector = np.zeros((2, self._ny, self._nx))
        vec_r = r[1] - r[0]
        vec_z = z[1] - z[0]
        vec_magn = np.sqrt(vec_r**2 + vec_z**2)
        self._poloidal_basis_vector[0] = np.divide(vec_r, vec_magn, out=np.zeros_like(vec_magn), where=(vec_magn > 0))
        self._poloidal_basis_vector[1] = np.divide(vec_z, vec_magn, out=np.zeros_like(vec_magn), where=(vec_magn > 0))

        # Calculating radial contact areas
        self._radial_area = np.pi * (r[1] + r[0]) * vec_magn

        # Calculating radial basis vector
        self._radial_basis_vector = np.zeros((2, self._ny, self._nx))
        vec_r = r[2] - r[0]
        vec_z = z[2] - z[0]
        vec_magn = np.sqrt(vec_r**2 + vec_z**2)
        self._radial_basis_vector[0] = np.divide(vec_r, vec_magn, out=np.zeros_like(vec_magn), where=(vec_magn > 0))
        self._radial_basis_vector[1] = np.divide(vec_z, vec_magn, out=np.zeros_like(vec_magn), where=(vec_magn > 0))

        # Calculating poloidal contact areas
        self._poloidal_area = np.pi * (r[2] + r[0]) * vec_magn

        # For convertion from Cartesian to poloidal
        # TODO Make it work with triangle cells
        self._inv_det = 1. / (self._poloidal_basis_vector[0] * self._radial_basis_vector[1] -
                              self._poloidal_basis_vector[1] * self._radial_basis_vector[0])

        # Finding unique vertices
        vertices = np.array([r.flatten(), z.flatten()]).T
        self._vertex_coords, unique_vertices = np.unique(vertices, axis=0, return_inverse=True)
        self._num_vertices = self._vertex_coords.shape[0]

        # Work out the extent of the mesh.
        self._mesh_extent = {"minr": r.min(), "maxr": r.max(), "minz": z.min(), "maxz": z.max()}

        # Pull out the index number for each unique vertex in this rectangular cell.
        # Unusual vertex indexing is based on SOLPS output, see Matlab code extract from David Moulton.
        ng = self._nx * self._ny  # grid size
        ym, xm = np.meshgrid(np.arange(self._ny, dtype=np.int32), np.arange(self._nx, dtype=np.int32), indexing='ij')

        # add quadrangle b2 grid
        self._quadrangles = np.zeros((ng, 4), dtype=np.int32)
        self._quadrangles[:, 0] = unique_vertices[0:ng]
        self._quadrangles[:, 1] = unique_vertices[2 * ng: 3 * ng]
        self._quadrangles[:, 2] = unique_vertices[3 * ng: 4 * ng]
        self._quadrangles[:, 3] = unique_vertices[ng: 2 * ng]

        # add mapping from quadrangles to the b2 grid
        self._quadrangle_to_grid_map = np.zeros((ng, 2), dtype=np.int32)
        self._quadrangle_to_grid_map[:, 0] = ym.flatten()
        self._quadrangle_to_grid_map[:, 1] = xm.flatten()

        # Number of triangles must be equal to number of rectangle centre points times 2.
        self._num_tris = ng * 2
        self._triangles = np.zeros((self._num_tris, 3), dtype=np.int32)
        self._triangles[0::2, 0] = self._quadrangles[:, 0]
        self._triangles[0::2, 1] = self._quadrangles[:, 1]
        self._triangles[0::2, 2] = self._quadrangles[:, 2]
        # Split the quad cell into two triangular cells.
        self._triangles[1::2, 0] = self._quadrangles[:, 2]
        self._triangles[1::2, 1] = self._quadrangles[:, 3]
        self._triangles[1::2, 2] = self._quadrangles[:, 0]

        # Each triangle cell is mapped to the tuple ID (ix, iy) of its parent mesh cell.
        self._triangle_to_grid_map = np.zeros((self._num_tris, 2), dtype=np.int32)
        self._triangle_to_grid_map[::2, 0] = ym.flatten()
        self._triangle_to_grid_map[::2, 1] = xm.flatten()
        self._triangle_to_grid_map[1::2] = self._triangle_to_grid_map[::2]

    @property
    def nx(self):
        """Number of grid cells in the poloidal direction."""
        return self._nx

    @property
    def ny(self):
        """Number of grid cells in the radial direction."""
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
        """Volume of each grid cell in m-3."""
        return self._vol

    @property
    def neighbix(self):
        """Poloidal indeces of neighbouring cells in order: left, bottom, right, top."""
        return self._neighbix

    @property
    def neighbiy(self):
        """Radial indeces of neighbouring cells in order: left, bottom, right, top."""
        return self._neighbiy

    @property
    def radial_area(self):
        """Radial contact area in m-2."""
        return self._radial_area

    @property
    def poloidal_area(self):
        """Poloidal contact area in m-2."""
        return self._poloidal_area

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
    def quadrangles(self):
        """Array of quadrangle vertex indices with (num_thiangles, 3) shape."""
        return self._quadrangles

    @property
    def poloidal_basis_vector(self):
        """
        Array of 2D poloidal basis vectors for grid cells.

        For each cell there is a poloidal and radial basis vector.

        Any vector on the poloidal grid can be converted to cartesian with the following transformation.
        bx = (p_x  r_x) ( b_p )
        by   (p_y  r_y) ( b_r )

        :return: ndarray with shape (2, ny, nx).
        """
        return self._poloidal_basis_vector

    @property
    def radial_basis_vector(self):
        """
        Array of 2D radial basis vectors for grid cells.

        For each cell there is a poloidal and radial basis vector.

        Any vector on the poloidal grid can be converted to cartesian with the following transformation.
        bx = (p_x  r_x) ( b_p )
        by   (p_y  r_y) ( b_r )

        :return: ndarray with shape (2, ny, nx).
        """
        return self._radial_basis_vector

    @property
    def triangle_to_grid_map(self):
        """
        Array mapping every triangle index to a tuple grid cell ID, i.e. (iy, ix).

        :return: ndarray with shape (nx*ny*2, 2)
        """
        return self._triangle_to_grid_map

    @property
    def quadrangle_to_grid_map(self):
        """
        Array mapping every quadrangle index to a tuple grid cell ID, i.e. (iy, ix).

        :return: ndarray with shape (nx*ny, 2)
        """
        return self._quadrangle_to_grid_map

    def __getstate__(self):
        state = {
            'r': self._r,
            'z': self._z,
            'vol': self._vol,
            'neighbix': self._neighbix,
            'neighbiy': self._neighbiy
        }
        return state

    def __setstate__(self, state):
        self._r = state['r']
        self._z = state['z']
        self._vol = state['vol']
        self._neighbix = state['neighbix']
        self._neighbiy = state['neighbiy']
        self._initial_setup()

    def to_cartesian(self, vec_pol):
        """
        Converts the 2D vector defined on mesh from poloidal to cartesian coordinates.
        :param ndarray vec_pol: Array of 2D vector with with shape (2, ny, nx).
            [0, :, :] - poloidal component, [1, :, :] - radial component

        :return: ndarray with shape (2, ny, nx)
        """
        vec_cart = np.zeros((2, self._ny, self._nx))
        vec_cart[0] = self._poloidal_basis_vector[0] * vec_pol[0] + self._radial_basis_vector[0] * vec_pol[1]
        vec_cart[1] = self._poloidal_basis_vector[1] * vec_pol[0] + self._radial_basis_vector[1] * vec_pol[1]

        return vec_cart

    def to_poloidal(self, vec_cart):
        """
        Converts the 2D vector defined on mesh from cartesian to poloidal coordinates.
        :param ndarray vector_on_mesh: Array of 2D vector with with shape (2, ny, nx).
            [0, :, :] - R component, [1, :, :] - Z component

        :return: ndarray with shape (2, ny, nx)
        """
        vec_pol = np.zeros((2, self._ny, self._nx))
        vec_pol[0] = self._inv_det * (self._radial_basis_vector[1] * vec_cart[0] - self._radial_basis_vector[0] * vec_cart[1])
        vec_pol[1] = self._inv_det * (self._poloidal_basis_vector[0] * vec_cart[1] - self._poloidal_basis_vector[1] * vec_cart[0])

        return vec_pol

    def plot_triangle_mesh(self, solps_data=None, ax=None):
        """
        Plot the triangle mesh grid geometry to a matplotlib figure.

        :param solps_data: Data array defined on the SOLPS mesh
        """
        warn("plot_triangle_mesh method is deprecated, use functions from plot module.", DeprecationWarning)
        from .plot import plot_triangle_mesh
        ax = plot_triangle_mesh(self, solps_data, ax)
        return ax

    def plot_quadrangle_mesh(self, solps_data=None, ax=None):
        """
        Plot the quadrangle mesh grid geometry to a matplotlib figure.

        :param solps_data: Data array defined on the SOLPS mesh
        """
        warn("plot_quadrangle_mesh method is deprecated, use functions from plot module.", DeprecationWarning)
        from .plot import plot_quadrangle_mesh
        ax = plot_quadrangle_mesh(self, solps_data, ax)
        return ax
