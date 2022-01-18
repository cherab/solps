
# Copyright 2014-2020 United Kingdom Atomic Energy Authority
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

from raysect.core.math import Vector3D, translate, rotate_basis, rotate_z
from raysect.optical import World
from raysect.optical.observer import PinholeCamera, PowerPipeline2D
from raysect.optical.material import AbsorbingSurface
from raysect.primitive import Cylinder

from cherab.core.model import TotalRadiatedPower
from cherab.openadas import OpenADAS
from cherab.generomak.machine import load_first_wall
from cherab.solps import load_solps_from_raw_output


###############################################################################
# Load the simulation and create a plasma object from it.
###############################################################################
demos_directory = os.path.dirname(__file__)
simulation_directory = os.path.join(demos_directory, 'data', 'raw')
print('Loading simulation...')
sim = load_solps_from_raw_output(simulation_directory)

print('Creating plasma...')
plasma = sim.create_plasma()

###############################################################################
# Image the plasma with a camera.
###############################################################################
print('Imaging plasma...')
world = World()
plasma.parent = world

# Load the generomak first wall
load_first_wall(world)


# For each species in the plasma, model the total emission from that species.
plasma.atomic_data = OpenADAS(permit_extrapolation=True)
for species in plasma.composition:
    if species.charge < species.element.atomic_number:
        plasma.models.add(TotalRadiatedPower(species.element, species.charge))

# A wide-angle pinhole camera looking horizontally towards the centre of the device.
camera = PinholeCamera((128, 128))
camera.parent = world
camera.pixel_samples = 10
camera.transform = (rotate_z(22.5)
                    * translate(1.5 * sim.mesh.mesh_extent['maxr'], 0, 0)
                    * rotate_basis(Vector3D(-1, 0, 0), Vector3D(0, 0, 1)))
camera.fov = 75
# The TotalRadiatedPower model is not spectrally resolved. So use a monochromatic
# pipeline to image the plasma.
camera.pipelines = [PowerPipeline2D()]
camera.spectral_bins = 1

plt.ion()
camera.observe()

plt.ioff()
plt.show()
