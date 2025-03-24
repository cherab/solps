
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
from matplotlib.collections import PolyCollection

from cherab.solps import load_solps_from_raw_output
from cherab.solps.plot import plot_quadrangle_mesh, plot_triangle_mesh


# Load the simulation.
demos_directory = os.path.dirname(os.path.dirname(__file__))
simulation_directory = os.path.join(demos_directory, 'data', 'raw')
print('Loading simulation...')
sim = load_solps_from_raw_output(simulation_directory)
mesh = sim.mesh

# plot quadrangle and triangle meshes
# plot the quadrangle b2 mesh
ax = plot_quadrangle_mesh(mesh)
ax.set_title("Quadrangle B2 Mesh")
ax.get_figure().set_size_inches((10, 20))

#plot the quadrangle b2 mesh with b2 ion temperature values
ax = plot_quadrangle_mesh(mesh, solps_data=sim.ion_temperature)
ax.get_figure().colorbar(ax.collections[0], aspect=40)
ax.get_figure().set_size_inches((10, 20))
ax.set_title("B2 Ion Temperature [eV]")
plt.show()

# axes can also be passed as an argument
fig_pass, ax = plt.subplots(figsize=(10, 20))
ax = plot_triangle_mesh(mesh, solps_data=sim.ion_temperature, ax=ax)
ax.get_figure().colorbar(ax.collections[0], aspect=40)
ax.get_figure().set_size_inches((10, 20))
ax.set_title("Cherab Triangle Mesh with Ion Temperature [eV]")
plt.show()

# The following part only illustrates what is done within the plot methods

# get the neccessary data from SOLPSSimulation
vertices = sim.mesh.vertex_coordinates
quadrangles = sim.mesh.quadrangles
mesh_extent = sim.mesh.mesh_extent
quadrangle_to_grid_map = sim.mesh.quadrangle_to_grid_map

# plot the quadrangle b2 mesh
collection_mesh = PolyCollection(vertices[quadrangles], facecolor="none")

fig_mesh, ax = plt.subplots(figsize=(10, 20))
ax.add_collection(collection_mesh)
ax.set_xlim(mesh_extent["minr"], mesh_extent["maxr"])
ax.set_ylim(mesh_extent["minz"], mesh_extent["maxz"])
ax.set_aspect(1)
ax.set_ylabel("R [m]")
ax.set_ylabel("Z [m]")
ax.set_title("Quadrangle B2 Mesh")

# plot the quadrangle b2 mesh with b2 ion temperature values
collection_values = PolyCollection(vertices[quadrangles])
collection_values.set_array(sim.ion_temperature[quadrangle_to_grid_map[:,0], quadrangle_to_grid_map[:,1]])

fig_mesh, ax = plt.subplots(figsize=(10, 20))
ax.add_collection(collection_values)
ax.set_xlim(mesh_extent["minr"], mesh_extent["maxr"])
ax.set_ylim(mesh_extent["minz"], mesh_extent["maxz"])
ax.set_aspect(1)
ax.set_ylabel("R [m]")
ax.set_ylabel("Z [m]")
ax.set_title("B2 Ion Temperature")

plt.show()