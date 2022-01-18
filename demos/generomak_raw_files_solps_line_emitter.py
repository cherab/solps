
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

"""
The SOLPSLineEmitter calculates the radiation emitted in spectral lines using custom
radiation power density functions. This emitter is useful when the power density has
already been calculated with atomic data other than ADAS (e.g. with EIRENE), or
when the power density needs to be cached on the SOLPS mesh for better performance.
"""

import os

import numpy as np
from matplotlib.colors import SymLogNorm
import matplotlib.pyplot as plt

from raysect.core.math import Point3D, Vector3D, rotate_z
from raysect.optical import World, Spectrum
from raysect.optical.observer import SpectralRadiancePipeline0D

from cherab.core.model import ExcitationLine, RecombinationLine
from cherab.core.atomic import Line, hydrogen
from cherab.openadas import OpenADAS
from cherab.generomak.machine import load_first_wall
from cherab.tools.observers import FibreOpticGroup, SpectroscopicFibreOptic
from cherab.solps import load_solps_from_raw_output
from cherab.solps.models.line_emitter import SOLPSLineEmitter


def plot_emission_and_sightlines(solps_simulation, fibre_optic_group, sightline_length=3.4):
    # plot H-alpha power density
    ax = sim.mesh.plot_quadrangle_mesh(solps_data=sim.halpha_total_radiation)
    linthresh = np.percentile(np.unique(sim.halpha_total_radiation), 10)
    ax.collections[0].set_norm(SymLogNorm(linthresh=linthresh))
    ax.get_figure().colorbar(ax.collections[0], aspect=40)
    ax.get_figure().set_size_inches((5.5, 10))
    ax.set_title("H-alpha power density [W m-3]")

    # plot lines of sight
    for sight_line in fibre_optic_group.sight_lines:
        origin = sight_line.origin
        direction = sight_line.direction
        radius = sight_line.radius
        angle = np.deg2rad(sight_line.acceptance_angle)
        end = origin + sightline_length * direction
        radius_end = radius + np.tan(angle) * sightline_length
        ro = np.sqrt(origin.x**2 + origin.y**2)
        zo = origin.z
        re = np.sqrt(end.x**2 + end.y**2)
        ze = end.z
        theta = 0.5 * np.pi - np.arctan2(zo - ze, ro - re)
        rr = (ro + radius * np.cos(theta), re + radius_end * np.cos(theta))
        rl = (ro - radius * np.cos(theta), re - radius_end * np.cos(theta))
        zr = (zo + radius * np.sin(theta), ze + radius_end * np.sin(theta))
        zl = (zo - radius * np.sin(theta), ze - radius_end * np.sin(theta))
        ax.plot(rr, zr, color='k', lw=0.75)
        ax.plot(rl, zl, color='k', lw=0.75)
        ax.plot((ro, re), (zo, ze), ls='--', color='C7', lw=0.75)

    return ax


###############################################################################
# Load the simulation and create a plasma object from it.
###############################################################################
demos_directory = os.path.dirname(__file__)
simulation_directory = os.path.join(demos_directory, 'data', 'raw')
print('Loading simulation...')
sim = load_solps_from_raw_output(simulation_directory)
print('Creating plasma...')
plasma = sim.create_plasma()

# Adding H-alpha excitation and recombination models
plasma.atomic_data = OpenADAS(permit_extrapolation=True)
h_alpha = Line(hydrogen, 0, (3, 2))
plasma.models = [ExcitationLine(h_alpha), RecombinationLine(h_alpha)]

# Caching the H-alpha radiation power density on the SOLPS mesh
min_wavelength = 654.
max_wavelength = 658.
print('Caching H-alpha radiation power density on SOLPS mesh...')
direction = Vector3D(0, 0, 1)
halpha_radiation = np.zeros_like(sim.electron_density)  # total radiation density on SOLPS mesh
spectrum = Spectrum(min_wavelength, max_wavelength, 1)
for ix in range(halpha_radiation.shape[1]):
    for iy in range(halpha_radiation.shape[0]):
        point = Point3D(sim.mesh.cr[iy, ix], 0, sim.mesh.cz[iy, ix])
        for model in plasma.models:
            halpha_radiation[iy, ix] += 4. * np.pi * model.emission(point, direction, spectrum.new_spectrum()).total()

sim.halpha_total_radiation = halpha_radiation

# Replace plasma models with the SOLPSLineEmitter
plasma.models = [SOLPSLineEmitter(sim.halpha_total_radiation_f2d, h_alpha)]

###############################################################################
# Observe the plasma with a group of optical fibres.
###############################################################################
world = World()
plasma.parent = world

# Load the generomak first wall
load_first_wall(world)

# A group of optical fibres observing the divertor.
group = FibreOpticGroup(parent=world, name='Divertor LoS array')
group.transform = rotate_z(22.5)
origin = Point3D(2.3, 0, 1.25)
angles = [-63.8, -66.5, -69.2, -71.9, -74.6]
direction_r = -np.cos(np.deg2rad(angles))
direction_z = np.sin(np.deg2rad(angles))
for i in range(len(angles)):
    group.add_sight_line(SpectroscopicFibreOptic(origin, Vector3D(direction_r[i], 0, direction_z[i]), name='{}'.format(i + 1)))
group.acceptance_angle = 1.4
group.radius = 0.001
group.pixel_samples = 5000
group.connect_pipelines(((SpectralRadiancePipeline0D, 'SpectralPipeline', None),))
group.min_wavelength = 655.5
group.max_wavelength = 656.9
group.spectral_bins = 256
group.display_progress = False

plt.ion()
# Plotting the projection of sight lines on the poloidal plane.
plot_emission_and_sightlines(sim, group)

print('Observing plasma...')
group.observe()

group.plot_spectra('SpectralPipeline', in_photons=True)
group.plot_total_signal('SpectralPipeline')
plt.ioff()
plt.show()
