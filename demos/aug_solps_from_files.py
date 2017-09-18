
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

import matplotlib.pyplot as plt
import numpy as np

from cherab.core.atomic.elements import deuterium
from cherab.solps import SOLPSSimulation

# Change to your local solps output files directory
sim = SOLPSSimulation.load_from_output_files('/home/mcarr/mst1/aug_2016/solps_testcase', debug=True)

mesh = sim.mesh
plasma = sim.plasma

me = mesh.mesh_extent
xl, xu = (me['minr'], me['maxr'])
yl, yu = (me['minz'], me['maxz'])

d0 = plasma.composition.get(deuterium, 0)
d1 = plasma.composition.get(deuterium, 1)
te_samples = np.zeros((500, 500))
ne_samples = np.zeros((500, 500))
ni_samples = np.zeros((500, 500))
n1_samples = np.zeros((500, 500))
inside_samples = np.zeros((500, 500))
xrange = np.linspace(xl, xu, 500)
yrange = np.linspace(yl, yu, 500)

for i, x in enumerate(xrange):
    for j, y in enumerate(yrange):
        ne_samples[j, i] = plasma.electron_distribution.density(x, 0.0, y)
        te_samples[j, i] = plasma.electron_distribution.effective_temperature(x, 0.0, y)
        ni_samples[j, i] = d0.distribution.density(x, 0.0, y)
        n1_samples[j, i] = d1.distribution.density(x, 0.0, y)
        inside_samples[j, i] = sim.inside_mesh(x, 0.0, y)

plt.figure()
plt.imshow(ne_samples, extent=[xl, xu, yl, yu], origin='lower')
plt.colorbar()
plt.xlim(xl, xu)
plt.ylim(yl, yu)
plt.title("electron density")
plt.figure()
plt.imshow(te_samples, extent=[xl, xu, yl, yu], origin='lower')
plt.colorbar()
plt.xlim(xl, xu)
plt.ylim(yl, yu)
plt.title("electron temperature")
plt.figure()
plt.imshow(ni_samples, extent=[xl, xu, yl, yu], origin='lower')
plt.colorbar()
plt.xlim(xl, xu)
plt.ylim(yl, yu)
plt.title("d0 density")
plt.figure()
plt.imshow(n1_samples, extent=[xl, xu, yl, yu], origin='lower')
plt.colorbar()
plt.xlim(xl, xu)
plt.ylim(yl, yu)
plt.title("d1 density")
plt.figure()
plt.imshow(inside_samples, extent=[xl, xu, yl, yu], origin='lower')
plt.colorbar()
plt.xlim(xl, xu)
plt.ylim(yl, yu)
plt.title("Inside/Outside test")
plt.show()
