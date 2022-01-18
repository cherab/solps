# cython: language_level=3

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

cimport numpy as np
from raysect.core.math.vector cimport Vector3D
from raysect.core.math.function.float.function2d.interpolate.common cimport MeshKDTree2D
from cherab.core.math.function cimport Function2D, VectorFunction2D


cdef class SOLPSFunction2D(Function2D):

    cdef:
        MeshKDTree2D _kdtree
        np.ndarray _grid_data, _triangle_to_grid_map
        np.int32_t[:,::1] _triangle_to_grid_map_mv
        double[:,::1] _grid_data_mv

    cdef double evaluate(self, double x, double y) except? -1e999

cdef class SOLPSVectorFunction2D(VectorFunction2D):

    cdef:
        MeshKDTree2D _kdtree
        np.ndarray _grid_vectors, _triangle_to_grid_map
        np.int32_t[:,::1] _triangle_to_grid_map_mv
        double[:,:,::1] _grid_vectors_mv

    cdef Vector3D evaluate(self, double x, double y)

