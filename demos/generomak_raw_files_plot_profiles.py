
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
from matplotlib.colors import LogNorm
import numpy as np
from scipy import linalg

from cherab.core.atomic.elements import hydrogen, carbon
from cherab.core.math.samplers import sample2d, sample2d_grid
from cherab.core.math.samplers import sample3d, sample3d_grid, samplevector3d_grid
from cherab.solps import load_solps_from_raw_output


###############################################################################
# Load the simulation.
###############################################################################
demos_directory = os.path.dirname(__file__)
simulation_directory = os.path.join(demos_directory, 'data', 'raw')
print('Loading simulation...')
sim = load_solps_from_raw_output(simulation_directory)

###############################################################################
# Sample some poloidal profiles from the simulation.
###############################################################################
mesh = sim.mesh
me = mesh.mesh_extent
xl, xu = (me['minr'], me['maxr'])
zl, zu = (me['minz'], me['maxz'])
nx = 500
nz = 500

print('Sampling profiles...')
xsamp, zsamp, ne_samples = sample2d(sim.electron_density_f2d, (xl, xu, nx), (zl, zu, nz))
te_samples = sample2d_grid(sim.electron_temperature_f2d, xsamp, zsamp)
h0_samples = sample2d_grid(sim.species_density_f2d[('hydrogen', 0)], xsamp, zsamp)
h1_samples = sample2d_grid(sim.species_density_f2d[('hydrogen', 1)], xsamp, zsamp)
c0_samples = sample2d_grid(sim.species_density_f2d[('carbon', 0)], xsamp, zsamp)
c1_samples = sample2d_grid(sim.species_density_f2d[('carbon', 1)], xsamp, zsamp)
c2_samples = sample2d_grid(sim.species_density_f2d[('carbon', 2)], xsamp, zsamp)
c3_samples = sample2d_grid(sim.species_density_f2d[('carbon', 3)], xsamp, zsamp)
c4_samples = sample2d_grid(sim.species_density_f2d[('carbon', 4)], xsamp, zsamp)
c5_samples = sample2d_grid(sim.species_density_f2d[('carbon', 5)], xsamp, zsamp)
c6_samples = sample2d_grid(sim.species_density_f2d[('carbon', 6)], xsamp, zsamp)
# Cartesian velocity is a 3D profile.
h1_velocity = samplevector3d_grid(sim.velocities_cartesian[('hydrogen', 1)],
                                  xsamp, [0], zsamp).squeeze()
h1_speed = linalg.norm(h1_velocity, axis=-1)
# Mask determining whether a point is inside or outside the simulation mesh in
# the poloidal plane. See sim.inside_volume_mesh for the 3D equivalent.
inside_samples = sample2d_grid(sim.inside_mesh, xsamp, zsamp)

###############################################################################
# Create a Cherab plasma from the simulation and sample quantities.
###############################################################################
print('Creating plasma...')
plasma = sim.create_plasma()

# Extract information about the species in the plasma. For brevity we'll only
# use the main ion and a single impurity charge state in this demo.
h0 = plasma.composition.get(hydrogen, 0)
h1 = plasma.composition.get(hydrogen, 1)
c0 = plasma.composition.get(carbon, 0)

# The distributions are 3D, so we perform a sample in 3D space with only a
# single y value to get a poloidal profile.
xsamp, _, zsamp, ne_plasma = sample3d(plasma.electron_distribution.density,
                                      (xl, xu, nx), (0, 0, 1), (zl, zu, nz))
ne_plasma = ne_plasma.squeeze()
te_plasma = sample3d_grid(plasma.electron_distribution.effective_temperature,
                          xsamp, [0], zsamp).squeeze()
h0_plasma = sample3d_grid(h0.distribution.density, xsamp, [0], zsamp).squeeze()
h1_plasma = sample3d_grid(h1.distribution.density, xsamp, [0], zsamp).squeeze()
c0_plasma = sample3d_grid(c0.distribution.density, xsamp, [0], zsamp).squeeze()
h1_plasma_velocity = samplevector3d_grid(h1.distribution.bulk_velocity, xsamp, [0], zsamp).squeeze()
h1_plasma_speed = linalg.norm(h1_velocity, axis=-1)

# Compare sampled quantities from the plasma with those from the simulation object.
print('Comparing plasma and simulation sampled quantities...')
np.testing.assert_equal(ne_plasma, ne_samples)
np.testing.assert_equal(te_plasma, te_samples)
np.testing.assert_equal(h0_plasma, h0_samples)
np.testing.assert_equal(h1_plasma, h1_samples)
np.testing.assert_equal(c0_plasma, c0_samples)
np.testing.assert_equal(h1_plasma_speed, h1_speed)
print('Plasma and simulation sampled quantities are identical.')

###############################################################################
# Plot the sampled vales.
###############################################################################
mesh.plot_mesh()
plt.title('Mesh geometry')

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

plt.show()
