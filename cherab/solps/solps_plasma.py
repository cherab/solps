
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

import re
import numpy as np
import matplotlib.pyplot as plt
from scipy.constants import atomic_mass, electron_mass

# Raysect imports
from raysect.core.math.interpolators import Discrete2DMesh
from raysect.core import translate, Vector3D

# CHERAB core imports
from cherab.core import Plasma, Species, Maxwellian
from cherab.core.math.mappers import AxisymmetricMapper
from cherab.core.atomic.elements import hydrogen, deuterium, helium, beryllium, carbon, nitrogen, oxygen, neon, \
    argon, krypton, xenon

# This SOLPS package imports
from .solps_3d_functions import SOLPSFunction3D, SOLPSVectorFunction3D


Q = 1.602E-19

_species_symbol_map = {
    'D': deuterium,
    'C': carbon,
    'He': helium,
    'N': nitrogen,
    'Ne': neon,
    'Ar': argon,
    'Kr': krypton,
    'Xe': xenon,
}

_SPECIES_REGEX = '([a-zA-z]+)\+?([0-9]+)'


class SOLPSSimulation:

    def __init__(self, mesh):

        self.mesh = mesh

        self._electron_temperature = None
        self._electron_density = None
        self._species_list = None
        self._species_density = None
        self._rad_par_flux = None
        self._radial_area = None
        self._b2_neutral_densities = None
        self._velocities_parallel = None
        self._velocities_radial = None
        self._velocities_toroidal = None
        self._velocities_cartesian = None
        self._inside_mesh = None
        self._total_rad = None
        self._b_field_vectors = None
        self._b_field_vectors_cartesian = None
        self._parallel_velocities = None
        self._radial_velocities = None
        self._eirene_model = None
        self._b2_model = None
        self._eirene = None

    @property
    def electron_temperature(self):
        """
        Simulated electron temperatures at each mesh cell.
        :return:
        """
        return self._electron_temperature

    @property
    def electron_density(self):
        """
        Simulated electron densities at each mesh cell.
        :return:
        """
        return self._electron_density

    @property
    def species_list(self):
        """
        Text list of species elements present in the simulation.
        :return:
        """
        return self._species_list

    @property
    def species_density(self):
        """
        Array of species densities at each mesh cell.
        :return:
        """
        return self._species_density

    @property
    def radial_particle_flux(self):
        """
        Blah
        :return:
        """
        return self._rad_par_flux

    @property
    def radial_area(self):
        """
        Blah
        :return:
        """
        return self._radial_area

    @property
    def b2_neutral_densities(self):
        """
        Neutral atom densities from B2
        :return:
        """
        return self._b2_neutral_densities

    @property
    def velocities_parallel(self):
        return self._velocities_parallel

    @property
    def velocities_radial(self):
        return self._velocities_radial

    @property
    def velocities_toroidal(self):
        return self._velocities_toroidal

    @property
    def velocities_cartesian(self):
        return self._velocities_cartesian

    @property
    def inside_mesh(self):
        """
        Function3D for testing if point p is inside the simulation mesh.
        """
        if self._inside_mesh is None:
            raise RuntimeError("Inside mesh test not available for this simulation")
        else:
            return self._inside_mesh

    @property
    def total_radiation(self):
        """
        Total radiation Function3D

        This is not calculated from the CHERAB emission models, instead it comes from the SOLPS output data.
        Is calculated from the sum of all integrated line emission and all Bremmstrahlung. The signals used are 'RQRAD'
        and 'RQBRM'. Final output is in W/str?
        """
        if self._total_rad is None:
            raise RuntimeError("Total radiation not available for this simulation")
        else:
            return self._total_rad

    @total_radiation.setter
    def total_radiation(self, value):

        radiation_data = np.array(value)

        if not radiation_data.shape == (self.mesh.nx, self.mesh.ny):
            raise ValueError("Total radiation data array must have same shape as mesh (nx, ny).")

        mapped_radiation_data = _map_data_onto_triangles(radiation_data)
        radiation_mesh_2d = Discrete2DMesh(self.mesh.vertex_coords, self.mesh.triangles, mapped_radiation_data, limit=False)
        self._total_rad = AxisymmetricMapper(radiation_mesh_2d)

    @property
    def parallel_velocities(self):
        """
        Plasma velocity field at each mesh cell. Equivalent to 'UA' in SOLPS.
        """
        if self._parallel_velocities is None:
            raise RuntimeError("Parallel velocities not available for this simulation")
        else:
            return self._parallel_velocities

    @property
    def radial_velocities(self):
        """
        Calculated radial velocity components for each species.
        """
        if self._parallel_velocities is None:
            raise RuntimeError("Radial velocities not available for this simulation")
        else:
            return self._radial_velocities

    @property
    def b_field(self):
        """
        Magnetic B field at each mesh cell in mesh cell coordinates (b_parallel, b_radial b_toroidal).
        """
        if self._b_field_vectors is None:
            raise RuntimeError("Magnetic field not available for this simulation")
        else:
            return self._b_field_vectors

    @property
    def b_field_cartesian(self):
        """
        Magnetic B field at each mesh cell in cartesian coordinates (Bx, By, Bz).
        """
        if self._b_field_vectors_cartesian is None:
            raise RuntimeError("Magnetic field not available for this simulation")
        else:
            return self._b_field_vectors_cartesian

    @property
    def eirene_simulation(self):
        """
        Data from an underlying EIRENE simulation.

        :rtype: Eirene
        """
        if self._eirene is None:
            raise RuntimeError("EIRENE simulation data not available for this SOLPS simulation")
        else:
            return self._eirene

    def plot_electrons(self):
        """ Make a plot of the electron temperature and density in the SOLPS mesh plane. """

        me = self.mesh.mesh_extent
        xl, xu = (me['minr'], me['maxr'])
        yl, yu = (me['minz'], me['maxz'])

        te_samples = np.zeros((500, 500))
        ne_samples = np.zeros((500, 500))

        xrange = np.linspace(xl, xu, 500)
        yrange = np.linspace(yl, yu, 500)

        plasma = self.plasma
        for i, x in enumerate(xrange):
            for j, y in enumerate(yrange):
                ne_samples[j, i] = plasma.electron_distribution.density(x, 0.0, y)
                te_samples[j, i] = plasma.electron_distribution.effective_temperature(x, 0.0, y)

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

    def plot_species_density(self, species, ionisation):
        """
        Make a plot of the requested species density in the SOLPS mesh plane.

        :param Element species: The species to plot.
        :param int ionisation: The charge state of the species to plot.
        """

        species_dist = self.plasma.get_species(species, ionisation)

        me = self.mesh.mesh_extent
        xl, xu = (me['minr'], me['maxr'])
        yl, yu = (me['minz'], me['maxz'])
        species_samples = np.zeros((500, 500))

        xrange = np.linspace(xl, xu, 500)
        yrange = np.linspace(yl, yu, 500)

        for i, x in enumerate(xrange):
            for j, y in enumerate(yrange):
                species_samples[j, i] = species_dist.distribution.density(x, 0.0, y)

        plt.figure()
        plt.imshow(species_samples, extent=[xl, xu, yl, yu], origin='lower')
        plt.colorbar()
        plt.xlim(xl, xu)
        plt.ylim(yl, yu)
        plt.title("Species {} - stage {} - density".format(species.name, ionisation))

    def plot_pec_emission_lines(self, emission_lines, title="", vmin=None, vmax=None, log=False):
        """
        Make a plot of the given PEC emission lines

        :param list emission_lines: List of PEC emission lines.
        :param str title: The title of the plot.
        :param float vmin: The minimum value for clipping the plots (default=None).
        :param float vmax: The maximum value for clipping the plots (default=None).
        :param bool log: Toggle a log plot for the data (default=False).
        """

        me = self.mesh.mesh_extent
        xl, xu = (me['minr'], me['maxr'])
        yl, yu = (me['minz'], me['maxz'])
        emission_samples = np.zeros((500, 500))

        xrange = np.linspace(xl, xu, 500)
        yrange = np.linspace(yl, yu, 500)

        for i, x in enumerate(xrange):
            for j, y in enumerate(yrange):
                for emitter in emission_lines:
                    emission_samples[j, i] += emitter.emission_at_point(x, 0.0, y)

        if log:
            emission_samples = np.log(emission_samples)
        plt.figure()
        plt.imshow(emission_samples, extent=[xl, xu, yl, yu], origin='lower', vmin=vmin, vmax=vmax)
        plt.colorbar()
        plt.xlim(xl, xu)
        plt.ylim(yl, yu)
        plt.title(title)

    def plot_radiated_power(self):
        """
        Make a plot of the given PEC emission lines

        :param list emission_lines: List of PEC emission lines.
        :param str title: The title of the plot.
        """

        mesh = self.mesh
        me = mesh.mesh_extent
        total_rad = self.total_radiation

        xl, xu = (me['minr'], me['maxr'])
        yl, yu = (me['minz'], me['maxz'])

        # tri_index_lookup = mesh.triangle_index_lookup

        emission_samples = np.zeros((500, 500))

        xrange = np.linspace(xl, xu, 500)
        yrange = np.linspace(yl, yu, 500)

        for i, x in enumerate(xrange):
            for j, y in enumerate(yrange):

                try:
                    # k, l = mesh.triangle_to_grid_map[int(tri_index_lookup(x, y))]
                    # emission_samples[i, j] = total_rad[k, l]
                    emission_samples[j, i] = total_rad(x, 0, y)

                except ValueError:
                    continue

        plt.figure()
        plt.imshow(emission_samples, extent=[xl, xu, yl, yu], origin='lower')
        plt.colorbar()
        plt.xlim(xl, xu)
        plt.ylim(yl, yu)
        plt.title("Radiated Power (W/m^3)")

    def create_plasma(self, parent=None, transform=None, name=None):
        """
        Make a CHERAB plasma object from this SOLPS simulation.

        :rtype: Plasma
        """

        mesh = self.mesh
        name = name or "SOLPS Plasma"
        plasma = Plasma(parent=parent, transform=transform, name=name)
        # TODO - add plasma geometry

        tri_index_lookup = self.mesh.triangle_index_lookup
        tri_to_grid = self.mesh.triangle_to_grid_map

        if isinstance(self._b_field_vectors, np.ndarray):
            plasma.b_field = SOLPSVectorFunction3D(tri_index_lookup, tri_to_grid, self._b_field_vectors)
        else:
            print('Warning! No magnetic field data available for this simulation.')

        # Create electron species
        triangle_data = _map_data_onto_triangles(self._electron_temperature)
        electron_te_interp = Discrete2DMesh(mesh.vertex_coords, mesh.triangles, triangle_data, limit=False)
        electron_temp = AxisymmetricMapper(electron_te_interp)
        triangle_data = _map_data_onto_triangles(self._electron_density)
        electron_ne_interp = Discrete2DMesh.instance(electron_te_interp, triangle_data)
        electron_dens = AxisymmetricMapper(electron_ne_interp)
        electron_velocity = lambda x, y, z: Vector3D(0, 0, 0)
        plasma.electron_distribution = Maxwellian(electron_dens, electron_temp, electron_velocity, electron_mass)

        if not isinstance(self.velocities_cartesian, np.ndarray):
            print('Warning! No velocity field data available for this simulation.')

        b2_neutral_i = 0  # counter for B2 neutrals
        for k, sp in enumerate(self.species_list):

            # Identify the species based on its symbol
            symbol, charge = re.match(_SPECIES_REGEX, sp).groups()
            charge = int(charge)
            species_type = _species_symbol_map[symbol]

            # If neutral and B" atomic density available,  use B2 density, otherwise use fluid species density.
            if isinstance(self.b2_neutral_densities, np.ndarray) and charge == 0:
                species_dens_data = self.b2_neutral_densities[:, :, b2_neutral_i]
                b2_neutral_i += 1
            else:
                species_dens_data = self.species_density[:, :, k]

            triangle_data = _map_data_onto_triangles(species_dens_data)
            dens = AxisymmetricMapper(Discrete2DMesh.instance(electron_te_interp, triangle_data))
            # dens = SOLPSFunction3D(tri_index_lookup, tri_to_grid, species_dens_data)

            # Create the velocity vector lookup function
            if isinstance(self.velocities_cartesian, np.ndarray):
                velocity = SOLPSVectorFunction3D(tri_index_lookup, tri_to_grid, self.velocities_cartesian[:, :, k, :])
            else:
                velocity = lambda x, y, z: Vector3D(0, 0, 0)

            distribution = Maxwellian(dens, electron_temp, velocity, species_type.atomic_weight * atomic_mass)
            plasma.composition.add(Species(species_type, charge, distribution))

        return plasma


def _map_data_onto_triangles(solps_dataset):
    """
    Reshape a SOLPS data array so that it matches the triangles in the SOLPS mesh.

    :param ndarray solps_dataset: Given SOLPS dataset, typically of shape (98 x 32).
    :return: New 1D ndarray with shape (98*32*2)
    """

    solps_mesh_shape = solps_dataset.shape
    triangle_data = np.zeros(solps_mesh_shape[0] * solps_mesh_shape[1] * 2, dtype=np.float64)

    tri_index = 0
    for i in range(solps_mesh_shape[0]):
        for j in range(solps_mesh_shape[1]):

            # Same data
            triangle_data[tri_index] = solps_dataset[i, j]
            tri_index += 1
            triangle_data[tri_index] = solps_dataset[i, j]
            tri_index += 1

    return triangle_data
