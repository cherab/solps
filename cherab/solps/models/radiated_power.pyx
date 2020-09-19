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


from math import pi

from raysect.core import translate
from raysect.primitive import Cylinder
from raysect.optical.material.emitter.inhomogeneous import NumericalIntegrator

cimport cython


cdef class SOLPSTotalRadiatedPower(InhomogeneousVolumeEmitter):

    def __init__(self, Function3D total_radiation, double vertical_offset=0.0, step=0.01):
        super().__init__(NumericalIntegrator(step=step))

        self.vertical_offset = vertical_offset
        self.total_rad = total_radiation

    def __call__(self, x, y, z):

        return self.total_rad.evaluate(x, y, z)  # this returns 0 if outside the mesh

    @cython.boundscheck(False)
    @cython.wraparound(False)
    @cython.initializedcheck(False)
    cpdef Spectrum emission_function(self, Point3D point, Vector3D direction, Spectrum spectrum,
                                     World world, Ray ray, Primitive primitive,
                                     AffineMatrix3D world_to_local, AffineMatrix3D local_to_world):

        cdef:
            double offset_z, wvl_range

        offset_z = point.z + self.vertical_offset

        wvl_range = ray.max_wavelength - ray.min_wavelength
        spectrum.samples_mv[:] = self.total_rad.evaluate(point.x, point.y, offset_z) / (4 * pi * wvl_range * spectrum.bins)

        return spectrum


def solps_total_radiated_power(world, solps_simulation, step=0.01):
    if solps_simulation.total_radiation_f3d is None:
        raise RuntimeError('Total radiation is not available for this simulation.')

    mesh = solps_simulation.mesh
    outer_radius = mesh.mesh_extent['maxr']
    inner_radius = mesh.mesh_extent['minr'] - 0.001
    plasma_height = mesh.mesh_extent['maxz'] - mesh.mesh_extent['minz']
    lower_z = mesh.mesh_extent['minz']

    main_plasma_cylinder = Cylinder(outer_radius, plasma_height, parent=world,
                                    material=SOLPSTotalRadiatedPower(solps_simulation.total_radiation_f3d, vertical_offset=lower_z, step=step),
                                    transform=translate(0, 0, lower_z))

    return main_plasma_cylinder
