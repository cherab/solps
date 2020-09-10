# cython: language_level=3

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

from libc.math cimport atan2, M_PI
import numpy as np
from numpy cimport ndarray

from raysect.core.math.vector cimport Vector3D, new_vector3d
from raysect.core.math.transform cimport rotate_z

from cherab.core.math.mappers cimport AxisymmetricMapper
from cherab.core.math.function cimport Function3D, Discrete2DMesh, VectorFunction3D
cimport cython

cdef double RAD_TO_DEG = 360 / (2*M_PI)


cdef class SOLPSFunction3D(Function3D):

    cdef:
        AxisymmetricMapper _triangle_index_lookup
        int[:,:] _triangle_to_grid_map
        double[:,:] _grid_values

    def __init__(self, Discrete2DMesh triangle_index_lookup, int[:,:] triangle_to_grid_map, double[:,:] grid_values):

        # todo: this is unsafe - converting an int to a double and performing operations - rounding errors
        self._triangle_index_lookup = AxisymmetricMapper(triangle_index_lookup)
        self._triangle_to_grid_map = triangle_to_grid_map
        self._grid_values = grid_values

    @cython.boundscheck(False)
    @cython.wraparound(False)
    @cython.initializedcheck(False)
    cdef double evaluate(self, double x, double y, double z) except? -1e999:

        cdef:
            int tri_index  # Index of the underlying mesh triangle
            int ix  # SOLPS grid x coordinate
            int iy  # SOLPS grid y coordinate

        try:
            tri_index = <int> self._triangle_index_lookup.evaluate(x, y, z)
        except ValueError:
            return 0.0  # Return zero if outside of mesh bounds

        ix, iy = self._triangle_to_grid_map[tri_index, :]

        return self._grid_values[ix, iy]


cdef class SOLPSVectorFunction3D(VectorFunction3D):

    cdef:
        AxisymmetricMapper _triangle_index_lookup
        int[:,:] _triangle_to_grid_map
        double[:,:,:] _grid_vectors

    def __init__(self, Discrete2DMesh triangle_index_lookup, object triangle_to_grid_map, object grid_vectors):

        # todo: this is unsafe - converting an int to a double and performing operations - rounding errors
        self._triangle_index_lookup = AxisymmetricMapper(triangle_index_lookup)
        self._triangle_to_grid_map = triangle_to_grid_map
        self._grid_vectors = grid_vectors

    @cython.boundscheck(False)
    @cython.wraparound(False)
    @cython.initializedcheck(False)
    cdef Vector3D evaluate(self, double x, double y, double z):

        cdef:
            int tri_index  # Index of the underlying mesh triangle
            int ix  # SOLPS grid x coordinate
            int iy  # SOLPS grid y coordinate
            double vx, vy, vz
            Vector3D v

        try:
            tri_index = <int> self._triangle_index_lookup.evaluate(x, y, z)
        except ValueError:
            return Vector3D(0.0, 0.0, 0.0)  # Return zero vector if outside of mesh bounds.

        ix, iy = self._triangle_to_grid_map[tri_index, :]

        # print(self._grid_vectors.shape)
        # Lookup vector for this grid cell.
        vx = self._grid_vectors[0, ix, iy]
        vy = self._grid_vectors[1, ix, iy]
        vz = self._grid_vectors[2, ix, iy]
        v = new_vector3d(vx, vy, vz)

        # Rotate vector field around the z-axis.
        return v.transform(rotate_z(atan2(y, x) * RAD_TO_DEG))
