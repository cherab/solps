
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

import sys
import os
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from scipy import linalg

from raysect.core.math import Vector3D, translate, rotate_basis
from raysect.optical import World
from raysect.optical.observer import PinholeCamera
from raysect.optical.material import AbsorbingSurface
from raysect.primitive import Cylinder

from cherab.core.atomic.elements import hydrogen, carbon
from cherab.core.math.samplers import sample2d_grid, sample3d, sample3d_grid, samplevector3d_grid
from cherab.core.model import TotalRadiatedPower
from cherab.openadas import OpenADAS
from cherab.solps import load_solps_from_raw_output


###############################################################################
# Load the simulation.
###############################################################################
demos_directory = os.path.dirname(__file__)
simulation_directory = os.path.join(demos_directory, 'generomak_raw_files')
try:
    print('Loading simulation...')
    sim = load_solps_from_raw_output(simulation_directory)
except RuntimeError:
    print('ERROR: simulation files not found. Please extract the raw '
          'files from the zip archive before running this demo:')
    print('> unzip {0}/generomak_raw_files.zip -d {0}/generomak_raw_files'.format(demos_directory))
    sys.exit(1)

###############################################################################
# Create a Cherab plasma from the simulation
###############################################################################
mesh = sim.mesh
plasma = sim.create_plasma()

me = mesh.mesh_extent
xl, xu = (me['minr'], me['maxr'])
zl, zu = (me['minz'], me['maxz'])
nx = 500
ny = 500

# Extract information about the species in the plasma.
h0 = plasma.composition.get(hydrogen, 0)
h1 = plasma.composition.get(hydrogen, 1)
c0 = plasma.composition.get(carbon, 0)
c1 = plasma.composition.get(carbon, 1)
c2 = plasma.composition.get(carbon, 2)
c3 = plasma.composition.get(carbon, 3)
c4 = plasma.composition.get(carbon, 4)
c5 = plasma.composition.get(carbon, 5)
c6 = plasma.composition.get(carbon, 6)

###############################################################################
# Sample some poloidal profiles from the simulation.
###############################################################################
print('Sampling profiles...')
# The distributions are 3D, so we perform a sample in 3D space with only a
# single y value to get a poloidal profile.
xsamp, _, zsamp, ne_samples = sample3d(plasma.electron_distribution.density,
                                       (xl, xu, nx), (0, 0, 1), (zl, zu, ny))
ne_samples = ne_samples.squeeze()
te_samples = sample3d_grid(plasma.electron_distribution.effective_temperature,
                           xsamp, [0], zsamp).squeeze()
h0_samples = sample3d_grid(h0.distribution.density, xsamp, [0], zsamp).squeeze()
h1_samples = sample3d_grid(h1.distribution.density, xsamp, [0], zsamp).squeeze()
c0_samples = sample3d_grid(c0.distribution.density, xsamp, [0], zsamp).squeeze()
c1_samples = sample3d_grid(c1.distribution.density, xsamp, [0], zsamp).squeeze()
c2_samples = sample3d_grid(c2.distribution.density, xsamp, [0], zsamp).squeeze()
c3_samples = sample3d_grid(c3.distribution.density, xsamp, [0], zsamp).squeeze()
c4_samples = sample3d_grid(c4.distribution.density, xsamp, [0], zsamp).squeeze()
c5_samples = sample3d_grid(c5.distribution.density, xsamp, [0], zsamp).squeeze()
c6_samples = sample3d_grid(c6.distribution.density, xsamp, [0], zsamp).squeeze()
# Magnitude of main ion velocity vector.
h1_velocity = samplevector3d_grid(h1.distribution.bulk_velocity, xsamp, [0], zsamp).squeeze()
h1_speed = linalg.norm(h1_velocity, axis=-1)
# Mask determining whether a point is inside or outside the simulation mesh.
# This is a 2D function: see sim.inside_volume_mesh for the 3D equivalent.
inside_samples = sample2d_grid(sim.inside_mesh, xsamp, zsamp)

# Plot the simulation mesh.
plt.ion()
mesh.plot_mesh()
plt.title('Mesh geometry')

# Plot sampled values.
plt.figure()
plt.imshow(ne_samples.T, extent=[xl, xu, zl, zu], origin='lower', norm=LogNorm())
plt.colorbar()
plt.xlim(xl, xu)
plt.ylim(zl, zu)
plt.title('Electron density [m-3]')

plt.figure()
plt.imshow(te_samples.T, extent=[xl, xu, zl, zu], origin='lower')
plt.colorbar()
plt.xlim(xl, xu)
plt.ylim(zl, zu)
plt.title('Electron temperature [eV]')

plt.figure()
plt.imshow(h0_samples.T, extent=[xl, xu, zl, zu], origin='lower', norm=LogNorm())
plt.colorbar()
plt.xlim(xl, xu)
plt.ylim(zl, zu)
plt.title('H0 density [m-3]')

plt.figure()
plt.imshow(h1_samples.T, extent=[xl, xu, zl, zu], origin='lower', norm=LogNorm())
plt.colorbar()
plt.xlim(xl, xu)
plt.ylim(zl, zu)
plt.title('H+ density [m-3]')

plt.figure()
plt.imshow(c0_samples.T, extent=[xl, xu, zl, zu], origin='lower', norm=LogNorm())
plt.colorbar()
plt.xlim(xl, xu)
plt.ylim(zl, zu)
plt.title('CI density [m-3]')

plt.figure()
plt.imshow(c1_samples.T, extent=[xl, xu, zl, zu], origin='lower', norm=LogNorm())
plt.colorbar()
plt.xlim(xl, xu)
plt.ylim(zl, zu)
plt.title('CII density [m-3]')

plt.figure()
plt.imshow(c2_samples.T, extent=[xl, xu, zl, zu], origin='lower', norm=LogNorm())
plt.colorbar()
plt.xlim(xl, xu)
plt.ylim(zl, zu)
plt.title('CIII density [m-3]')

plt.figure()
plt.imshow(c3_samples.T, extent=[xl, xu, zl, zu], origin='lower', norm=LogNorm())
plt.colorbar()
plt.xlim(xl, xu)
plt.ylim(zl, zu)
plt.title('CIV density [m-3]')

plt.figure()
plt.imshow(c4_samples.T, extent=[xl, xu, zl, zu], origin='lower', norm=LogNorm())
plt.colorbar()
plt.xlim(xl, xu)
plt.ylim(zl, zu)
plt.title('CV density [m-3]')

plt.figure()
plt.imshow(c5_samples.T, extent=[xl, xu, zl, zu], origin='lower', norm=LogNorm())
plt.colorbar()
plt.xlim(xl, xu)
plt.ylim(zl, zu)
plt.title('CVI density [m-3]')

plt.figure()
plt.imshow(c6_samples.T, extent=[xl, xu, zl, zu], origin='lower', norm=LogNorm())
plt.colorbar()
plt.xlim(xl, xu)
plt.ylim(zl, zu)
plt.title('CVII density [m-3]')

plt.figure()
plt.imshow(h1_speed.T, extent=[xl, xu, zl, zu], origin='lower')
plt.colorbar()
plt.xlim(xl, xu)
plt.ylim(zl, zu)
plt.title('H+ speed [m/s]')

plt.figure()
plt.imshow(inside_samples.T, extent=[xl, xu, zl, zu], origin='lower')
plt.colorbar()
plt.xlim(xl, xu)
plt.ylim(zl, zu)
plt.title('Inside/Outside test')

input('Press Return to continue...')
plt.close('all')

###############################################################################
# Image the plasma with a camera.
###############################################################################
print('Imaging plasma...')
world = World()
plasma.parent = world

# Cherab does not yet have a wall geometry specified for Generomak (see
# https://github.com/cherab/core/issues/212), so we'll just make our own.
# This will consist of a cylindrical center column and nothing else, which
# will allow us to position an overview camera to image the whole plasma.
centre_column_radius = 0.9 * me['minr']
centre_column_height = me['maxz'] - me['minz']
Cylinder(radius=centre_column_radius, height=centre_column_height,
         transform=translate(0, 0, me['minz']),
         material=AbsorbingSurface(), name='Centre column', parent=world)

# A wide-angle pinhole camera looking horizontally towards the centre of the device.
camera = PinholeCamera((128, 128))
camera.parent = world
camera.pixel_samples = 10
camera.transform = (translate(2 * me['maxr'], 0, 0)
                    * rotate_basis(Vector3D(-1, 0, 0), Vector3D(0, 0, 1)))
camera.fov = 60

# For each species in the plasma, model the total emission from that species.
plasma.atomic_data = OpenADAS(permit_extrapolation=True)
for species in plasma.composition:
    if species.charge < species.element.atomic_number:
        plasma.models.add(TotalRadiatedPower(species.element, species.charge))

camera.observe()

input('Press Return to finish...')
