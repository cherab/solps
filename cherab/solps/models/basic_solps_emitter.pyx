
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

from raysect.optical cimport World, Primitive, Ray, Spectrum, SpectralFunction, Point3D, Vector3D, AffineMatrix3D
from raysect.optical.material.emitter cimport InhomogeneousVolumeEmitter

from cherab.core.math.function cimport Function3D
from cherab.core.math.mappers cimport AxisymmetricMapper
from cherab.core.distribution cimport DistributionFunction
from cherab.core.atomic.line cimport Line
from cherab.core.model.spectra cimport add_gaussian_line, thermal_broadening


# RECIP_4_PI = 1 / (4 * pi)
DEF RECIP_4_PI = 0.7853981633974483


cdef class BasicSolpsEmitter(InhomogeneousVolumeEmitter):

    cdef public Function3D emitter
    cdef public DistributionFunction electron_distribution
    cdef public Line line

    def __init__(self, Line line, DistributionFunction electron_distribution, emitter_2D_grid, step=0.01):

        super().__init__(step)
        self.line = line
        self.electron_distribution = electron_distribution
        self.emitter = AxisymmetricMapper(emitter_2D_grid)


    cpdef Spectrum emission_function(self, Point3D point, Vector3D direction, Spectrum spectrum,
                                     World world, Ray ray, Primitive primitive,
                                     AffineMatrix3D to_local, AffineMatrix3D to_world):

        cdef double x, y, z, te, radiance, sigma

        x, y, z = point.transform(to_world)

        te = self.electron_distribution.effective_temperature(x, y, z)  # electron temperature t_e(x, y, z)
        radiance = RECIP_4_PI * self.emitter.evaluate(x, y, z)
        sigma = thermal_broadening(self.line.wavelength, te, self.line.element.atomic_weight)
        return add_gaussian_line(radiance, self.line.wavelength, sigma, spectrum)

