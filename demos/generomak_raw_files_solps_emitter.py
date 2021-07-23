
# Copyright 2014-2021 United Kingdom Atomic Energy Authority
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

import os
import matplotlib.pyplot as plt
import numpy as np

from raysect.core.math import Point3D, Vector3D, translate, rotate_basis, rotate_z
from raysect.optical import World, Spectrum
from raysect.optical.observer import PinholeCamera, PowerPipeline2D

from cherab.core.model import TotalRadiatedPower
from cherab.openadas import OpenADAS
from cherab.generomak.machine import load_first_wall
from cherab.solps import load_solps_from_raw_output
from cherab.solps.models import make_solps_emitter

# Unlike generomak_raw_files_image_plasma.py, here the total radiation power density
# is cached on the SOLPS mesh before imaging.
# The plasma object is not connected to the scenegraph. Instead, the emitter is created
# with make_solps_emitter(). This approach significantly speeds up the simulation.

###############################################################################
# Load the simulation and create a plasma object from it.
###############################################################################
demos_directory = os.path.dirname(__file__)
simulation_directory = os.path.join(demos_directory, 'data', 'raw')
print('Loading simulation...')
sim = load_solps_from_raw_output(simulation_directory)

print('Creating plasma...')
plasma = sim.create_plasma()

# For each species in the plasma, add the total radiation model.
plasma.atomic_data = OpenADAS(permit_extrapolation=True)
for species in plasma.composition:
    if species.charge < species.element.atomic_number:
        plasma.models.add(TotalRadiatedPower(species.element, species.charge))

###############################################################################
# Cache the total radiation power density on SOLPS mesh.
###############################################################################
print('Caching total radiation power density on SOLPS mesh...')
# Total radiation does not depend on wavelength, therefore Spectrum can be initialised with any values.
spectrum = Spectrum(min_wavelength=375., max_wavelength=740., bins=1)
# Calculating total radiation power density on SOLPS mesh
direction = Vector3D(0, 0, 1)
total_radiation = np.zeros((sim.mesh.ny, sim.mesh.nx))
for model in plasma.models:
    for ix in range(total_radiation.shape[1]):
        for iy in range(total_radiation.shape[0]):
            point = Point3D(sim.mesh.cr[iy, ix], 0, sim.mesh.cz[iy, ix])
            total_radiation[iy, ix] += 4. * np.pi * model.emission(point, direction, spectrum.new_spectrum()).total()
sim.total_radiation = total_radiation  # this creates sim.total_radiation_f2d and sim.total_radiation_f3d

# Make the emitter from sim.total_radiation_f2d() function
solps_emitter = make_solps_emitter(sim.mesh, sim.total_radiation_f2d)

###############################################################################
# Image the plasma with a camera.
###############################################################################
world = World()
# Do not add plasma object to the scenegraph! Add the emitter instead.
solps_emitter.parent = world

# Load the generomak first wall
load_first_wall(world)

# A wide-angle pinhole camera looking horizontally towards the centre of the device.
camera = PinholeCamera((256, 256))  # 4x more pixels than in generomak_raw_files_image_plasma.py
camera.parent = world
camera.pixel_samples = 100  # 10x more samples than in generomak_raw_files_image_plasma.py
camera.transform = (rotate_z(22.5)
                    * translate(1.5 * sim.mesh.mesh_extent['maxr'], 0, 0)
                    * rotate_basis(Vector3D(-1, 0, 0), Vector3D(0, 0, 1)))
camera.fov = 75
# The TotalRadiatedPower model is not spectrally resolved. So use a monochromatic
# pipeline to image the plasma.
camera.pipelines = [PowerPipeline2D(display_unsaturated_fraction=0.9999)]
camera.spectral_bins = 1

print('Imaging plasma...')
plt.ion()
camera.observe()
plt.ioff()
plt.show()
