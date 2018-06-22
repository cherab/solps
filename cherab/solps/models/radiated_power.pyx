
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


cdef class SOLPSTotalRadiatedPower(InhomogeneousVolumeEmitter):

    def __init__(self, object solps_simulation, step=0.01):
        super().__init__(NumericalIntegrator(step=step))

        self.inside_simulation = solps_simulation.inside_mesh
        self.total_rad = solps_simulation.total_radiation

    def __call__(self, x, y , z):

        if self.inside_simulation.evaluate(x, y, z) < 1.0:
            return 0.0

        return self.total_rad.evaluate(x, y, z)

    cpdef Spectrum emission_function(self, Point3D point, Vector3D direction, Spectrum spectrum,
                                     World world, Ray ray, Primitive primitive,
                                     AffineMatrix3D world_to_local, AffineMatrix3D local_to_world):

        cdef:
            double x, y, z, wvl_range
            Point3D world_point

        world_point = point.transform(local_to_world)
        x = world_point.x
        y = world_point.y
        z = world_point.z

        if self.inside_simulation.evaluate(x, y, z) < 1.0:
            return spectrum

        wvl_range = ray.max_wavelength - ray.min_wavelength
        spectrum.samples[0] = self.total_rad.evaluate(x, y, z) / (4 * pi) / wvl_range

        return spectrum


def solps_total_radiated_power(world, solps_simulation, step=0.01):
    mesh = solps_simulation.mesh
    outer_radius = mesh.mesh_extent['maxr']
    inner_radius = mesh.mesh_extent['minr'] - 0.001
    plasma_height = mesh.mesh_extent['maxz'] - mesh.mesh_extent['minz']
    lower_z = mesh.mesh_extent['minz']

    main_plasma_cylinder = Cylinder(outer_radius, plasma_height, parent=world,
                                    material=SOLPSTotalRadiatedPower(solps_simulation, step=step),
                                    transform=translate(0, 0, lower_z))

    return main_plasma_cylinder


