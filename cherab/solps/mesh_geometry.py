
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
from collections import namedtuple

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
from matplotlib.collections import PatchCollection
from raysect.core.math.interpolators import Discrete2DMesh

INFINITY = 1E99


class SOLPSMesh:
    """
    SOLPSMesh geometry object.

    The SOLPS mesh is rectangular. Each mesh cell is denoted by four vertices with one centre point. Vertices
    may be shared with neighbouring cells. The centre points should be unique.

    Raysect's mesh interpolator uses a different mesh scheme. Mesh cells are triangles and data values are stored at the
    triangle vertices. Therefore, each SOLPS rectangular cell is split into two triangular cells. The data points are
    later interpolated onto the vertex points.

    :param ndarray cr_r: Array of cell vertex r coordinates, must be 3 dimensional. Example shape is (98 x 32 x 4).
    :param ndarray cr_z: Array of cell vertex z coordinates, must be 3 dimensional. Example shape is (98 x 32 x 4).
    :param ndarray vol: Array of cell volumes. Example shape is (98 x 32).
    """

    def __init__(self, cr_r, cr_z, vol):

        self._cr = None
        self._cz = None
        self._poloidal_grid_basis = None

        nx = cr_r.shape[0]
        ny = cr_r.shape[1]
        self._nx = nx
        self._ny = ny

        self._r = cr_r
        self._z = cr_z

        self._vol = vol

        # Iterate through the arrays from MDS plus to pull out unique vertices
        unique_vertices = {}
        vertex_id = 0
        for i in range(nx):
            for j in range(ny):
                for k in range(4):
                    vertex = (cr_r[i, j, k], cr_z[i, j, k])
                    try:
                        unique_vertices[vertex]
                    except KeyError:
                        unique_vertices[vertex] = vertex_id
                        vertex_id += 1

        # Load these unique vertices into a numpy array for later use in Raysect's mesh interpolator object.
        self.num_vertices = len(unique_vertices)
        self.vertex_coords = np.zeros((self.num_vertices, 2), dtype=np.float64)
        for vertex, vertex_id in unique_vertices.items():
            self.vertex_coords[vertex_id, :] = vertex

        # Work out the extent of the mesh.
        rmin = cr_r.flatten().min()
        rmax = cr_r.flatten().max()
        zmin = cr_z.flatten().min()
        zmax = cr_z.flatten().max()
        self.mesh_extent = {"minr": rmin, "maxr": rmax, "minz": zmin, "maxz": zmax}

        # Number of triangles must be equal to number of rectangle centre points times 2.
        self.num_tris = nx * ny * 2
        self.triangles = np.zeros((self.num_tris, 3), dtype=np.int32)

        self._triangle_to_grid_map = np.zeros((nx*ny*2, 2), dtype=np.int32)
        tri_index = 0
        for i in range(nx):
            for j in range(ny):
                # Pull out the index number for each unique vertex in this rectangular cell.
                # Unusual vertex indexing is based on SOLPS output, see Matlab code extract from David Moulton.
                # cell_r = [r(i,j,1),r(i,j,3),r(i,j,4),r(i,j,2)];
                v1_id = unique_vertices[(cr_r[i, j, 0], cr_z[i, j, 0])]
                v2_id = unique_vertices[(cr_r[i, j, 2], cr_z[i, j, 2])]
                v3_id = unique_vertices[(cr_r[i, j, 3], cr_z[i, j, 3])]
                v4_id = unique_vertices[(cr_r[i, j, 1], cr_z[i, j, 1])]

                # Split the quad cell into two triangular cells.
                # Each triangle cell is mapped to the tuple ID (ix, iy) of its parent mesh cell.
                self.triangles[tri_index, :] = (v1_id, v2_id, v3_id)
                self._triangle_to_grid_map[tri_index, :] = (i, j)
                tri_index += 1
                self.triangles[tri_index, :] = (v3_id, v4_id, v1_id)
                self._triangle_to_grid_map[tri_index, :] = (i, j)
                tri_index += 1

        tri_indices = np.arange(self.num_tris, dtype=np.int32)
        self._tri_index_loopup = Discrete2DMesh(self.vertex_coords, self.triangles, tri_indices)

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
    def vol(self):
        """Volume/area of each grid cell."""
        return self._vol

    @property
    def poloidal_grid_basis(self):
        """
        Array of 2D basis vectors for grid cells.

        For each cell there is a parallel and radial basis vector.

        Any vector on the poloidal grid can be converted to cartesian with the following transformation.
        bx = (p_x  r_x) ( b_p )
        by   (p_y  r_y) ( b_r )

        :return: ndarray with shape (nx, ny, 2) where the two basis vectors are [parallel, radial] respectively.
        """
        return self._poloidal_grid_basis

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
            'cr_r': self._r,
            'cr_z': self._z,
            'vol': self._vol,
        }
        return state

    def plot_mesh(self):
        """
        Plot the mesh grid geometry to a matplotlib figure.
        """
        fig, ax = plt.subplots()
        patches = []
        for triangle in self.triangles:
            vertices = self.vertex_coords[triangle]
            patches.append(Polygon(vertices, closed=True))
        p = PatchCollection(patches, facecolors='none', edgecolors='b')
        ax.add_collection(p)
        ax.axis('equal')
        return ax

        # Code for plotting vessel geometry if available
        # for i in range(vessel.shape[0]):
        #     plt.plot([vessel[i, 0], vessel[i, 2]], [vessel[i, 1], vessel[i, 3]], 'k')
        # for i in range(vessel.shape[0]):
        #     plt.plot([vessel[i, 0], vessel[i, 2]], [vessel[i, 1], vessel[i, 3]], 'or')
