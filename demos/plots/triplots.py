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
from matplotlib.tri import Triangulation

from cherab.solps import load_solps_from_raw_output


# Load the simulation.
demos_directory = os.path.dirname(os.path.dirname(__file__))
simulation_directory = os.path.join(demos_directory, 'data', 'raw')
print('Loading simulation...')

sim = load_solps_from_raw_output(simulation_directory)
mesh = sim.mesh

# prepare data for triangulation plots using matplotlib.tri
tri = Triangulation(mesh.vertex_coordinates[:, 0], mesh.vertex_coordinates[:, 1], mesh.triangles)
ion_temperature_tri = sim.ion_temperature[mesh.triangle_to_grid_map[:, 0], mesh.triangle_to_grid_map[:, 1]]


# plot mesh
fig_mesh = plt.figure()
ax_mesh = fig_mesh.add_subplot(111)
ax_mesh.set_title("Mesh")
ax_mesh.triplot(tri, lw=0.2)
ax_mesh.set_aspect('equal')
ax_mesh.set_xlabel("R [m]")
ax_mesh.set_ylabel("z [m]")


# plot ion temperature
fig_ion_temperature = plt.figure()
ax_ion_temperature = fig_ion_temperature.add_subplot(111)
tpc = ax_ion_temperature.tripcolor(tri, ion_temperature_tri)
fig_ion_temperature.colorbar(tpc, ax=ax_ion_temperature, label="Ion Temperature [eV]")
ax_ion_temperature.set_title("Ion Temperature")
ax_ion_temperature.set_aspect('equal')
ax_ion_temperature.set_xlabel("R [m]")
ax_ion_temperature.set_ylabel("z [m]")

plt.show()
