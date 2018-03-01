
# Copyright 2014-2017 United Kingdom Atomic Energy Authority
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

    def __init__(self, object solps_simulation, double vertical_offset=0.0, step=0.01):
        super().__init__(NumericalIntegrator(step=step))

        self.vertical_offset = vertical_offset
        self.inside_simulation = solps_simulation.inside_volume_mesh
        self.total_rad = solps_simulation.total_radiation_volume

    def __call__(self, x, y , z):

        if self.inside_simulation.evaluate(x, y, z) < 1.0:
            return 0.0

        return self.total_rad.evaluate(x, y, z)

    @cython.boundscheck(False)
    @cython.wraparound(False)
    @cython.initializedcheck(False)
    cpdef Spectrum emission_function(self, Point3D point, Vector3D direction, Spectrum spectrum,
                                     World world, Ray ray, Primitive primitive,
                                     AffineMatrix3D world_to_local, AffineMatrix3D local_to_world):

        cdef:
            double offset_z, wvl_range

        offset_z =  point.z + self.vertical_offset
        if self.inside_simulation.evaluate(point.x, point.y, offset_z) < 1.0:
            return spectrum

        wvl_range = ray.max_wavelength - ray.min_wavelength
        spectrum.samples_mv[:] = self.total_rad.evaluate(point.x, point.y, offset_z) / (4 * pi * wvl_range * spectrum.bins)

        return spectrum

def solps_total_radiated_power(world, solps_simulation, step=0.01):
    mesh = solps_simulation.mesh
    outer_radius = mesh.mesh_extent['maxr']
    inner_radius = mesh.mesh_extent['minr'] - 0.001
    plasma_height = mesh.mesh_extent['maxz'] - mesh.mesh_extent['minz']
    lower_z = mesh.mesh_extent['minz']

    main_plasma_cylinder = Cylinder(outer_radius, plasma_height, parent=world,
                                    material=SOLPSTotalRadiatedPower(solps_simulation, vertical_offset=lower_z, step=step),
                                    transform=translate(0, 0, lower_z))

    return main_plasma_cylinder


