
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
import os
from math import sqrt
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
from .mesh_geometry import SOLPSMesh
from .b2.parse_b2_block_file import load_b2f_file
from .solps_3d_functions import SOLPSFunction3D, SOLPSVectorFunction3D
from .eirene import Eirene


Q = 1.602E-19

# key is nuclear charge Z and atomic mass AMU
_popular_species = {
    (1, 2): deuterium,
    (6, 12.011): carbon,
    (2, 4.003): helium,
    (7, 14.007): nitrogen,
    (18, 39.948): argon,
    (36, 83.798): krypton,
    (54, 131.293): xenon
}

_species_symbol_map = {
    'D': deuterium,
    'C': carbon,
    'He': helium,
    'N': nitrogen,
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

    def create_plasma(self):
        """
        Make a CHERAB plasma object from this SOLPS simulation.

        :rtype: Plasma
        """

        mesh = self.mesh
        plasma = Plasma(name="Plasma")

        tri_index_lookup = self.mesh.triangle_index_lookup
        tri_to_grid = self.mesh.triangle_to_grid_map

        plasma.b_field = SOLPSVectorFunction3D(tri_index_lookup, tri_to_grid, self.b_field)

        # Create electron species
        triangle_data = _map_data_onto_triangles(self._electron_temperature)
        electron_te_interp = Discrete2DMesh(mesh.vertex_coords, mesh.triangles, triangle_data, limit=False)
        electron_temp = AxisymmetricMapper(electron_te_interp)
        triangle_data = _map_data_onto_triangles(self._electron_density)
        electron_ne_interp = Discrete2DMesh.instance(electron_te_interp, triangle_data)
        electron_dens = AxisymmetricMapper(electron_ne_interp)
        electron_velocity = lambda x, y, z: Vector3D(0, 0, 0)
        plasma.electron_distribution = Maxwellian(electron_dens, electron_temp, electron_velocity, electron_mass)

        b2_neutral_i = 0  # counter for B2 neutrals
        for k, sp in enumerate(self.species_list):

            # Identify the species based on its symbol
            symbol, charge = re.match(_SPECIES_REGEX, sp).groups()
            charge = int(charge)
            species_type = _species_symbol_map[symbol]

            # If neutral and B" atomic density available,  use B2 density, otherwise use fluid species density.
            # if self.b2_neutral_densities and charge == 0:
            if charge == 0:
                species_dens_data = self.b2_neutral_densities[:, :, b2_neutral_i]
                b2_neutral_i += 1
            else:
                species_dens_data = self.species_density[:, :, k]

            triangle_data = _map_data_onto_triangles(species_dens_data)
            dens = AxisymmetricMapper(Discrete2DMesh.instance(electron_te_interp, triangle_data))
            # dens = SOLPSFunction3D(tri_index_lookup, tri_to_grid, species_dens_data)

            # Create the velocity vector lookup function
            velocity = SOLPSVectorFunction3D(tri_index_lookup, tri_to_grid, self.velocities_cartesian[:, :, k, :])

            distribution = Maxwellian(dens, electron_temp, velocity, species_type.atomic_weight * atomic_mass)
            plasma.composition.add(Species(species_type, charge, distribution))

        return plasma


# reshape method when loading from files
def _load_solps_vertex_signal(mds_connection, signal, mesh, template_interpolator=None, integrate_range=None):

    if integrate_range:
        lower, upper = integrate_range
        signal_data = np.swapaxes(mds_connection.get(signal).data(), 0, 2)  # (x, 32, 98) => (98, 32, x)
        signal_data = signal_data[:, :, lower:upper]
        signal_data = np.sum(signal_data, axis=2)
    else:
        signal_data = np.swapaxes(mds_connection.get(signal).data(), 0, 1)  # (32, 98) => (98, 32)
    triangle_data = _map_data_onto_triangles(signal_data)

    if template_interpolator:
        return Discrete2DMesh.instance(template_interpolator, triangle_data)
    else:
        return Discrete2DMesh(mesh.vertex_coords, mesh.triangles, triangle_data, limit=False)


# reshape method when loading from files
def _reshape_solps_data(signal_data, mesh, template_interpolator=None):

    triangle_data = _map_data_onto_triangles(signal_data)

    if template_interpolator:
        return Discrete2DMesh.instance(template_interpolator, triangle_data)
    else:
        return Discrete2DMesh(mesh.vertex_coords, mesh.triangles, triangle_data, limit=False)


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


def load_solps_from_mdsplus(mds_server, ref_number):
    """
    Load a SOLPS simulation from a MDSplus server.

    :param str mds_server: Server address.
    :param int ref_number: Simulation reference number.
    :rtype: SOLPSSimulation
    """

    from MDSplus import Connection as MDSConnection

    # Setup connection to server
    conn = MDSConnection(mds_server)
    conn.openTree('solps', ref_number)

    # Load SOLPS mesh geometry and lookup arrays
    mesh = SOLPSMesh.load_from_mdsplus(conn)
    sim = SOLPSSimulation(mesh)
    ni = mesh.nx
    nj = mesh.ny

    ##########################
    # Magnetic field vectors #
    raw_b_field = np.swapaxes(conn.get('\SOLPS::TOP.SNAPSHOT.B').data(), 0, 2)
    b_field_vectors_cartesian = np.zeros((ni, nj, 3))
    b_field_vectors = np.zeros((ni, nj, 3))
    for i in range(ni):
        for j in range(nj):
            bparallel = raw_b_field[i, j, 0]
            bradial = raw_b_field[i, j, 1]
            btoroidal = raw_b_field[i, j, 2]
            b_field_vectors[i, j] = (bparallel, bradial, btoroidal)

            pv = mesh.poloidal_grid_basis[i, j, 0]  # parallel basis vector
            rv = mesh.poloidal_grid_basis[i, j, 1]  # radial basis vector

            bx = pv.x * bparallel + rv.x * bradial  # component of B along poloidal x
            by = pv.y * bparallel + rv.y * bradial  # component of B along poloidal y
            b_field_vectors_cartesian[i, j] = (bx, btoroidal, by)
    sim._b_field_vectors = b_field_vectors
    sim._b_field_vectors_cartesian = b_field_vectors_cartesian

    # Load electron species
    sim._electron_temperature = np.swapaxes(conn.get('\SOLPS::TOP.SNAPSHOT.TE').data(), 0, 1)  # (32, 98) => (98, 32)
    sim._electron_density = np.swapaxes(conn.get('\SOLPS::TOP.SNAPSHOT.NE').data(), 0, 1)  # (32, 98) => (98, 32)

    ############################
    # Load each plasma species #
    ############################

    # Master list of species, e.g. ['D0', 'D+1', 'C0', 'C+1', ...
    sim._species_list = conn.get('\SOLPS::TOP.IDENT.SPECIES').data().decode('UTF-8').split()
    sim._species_density = np.swapaxes(conn.get('\SOLPS::TOP.SNAPSHOT.NA').data(), 0, 2)
    sim._rad_par_flux = np.swapaxes(conn.get('\SOLPS::TOP.SNAPSHOT.FNAY').data(), 0, 2)  # radial particle flux
    sim._radial_area = np.swapaxes(conn.get('\SOLPS::TOP.SNAPSHOT.SY').data(), 0, 1)  # radial contact area

    # Load the neutral atom density from B2
    neutral_dens_data = np.swapaxes(conn.get('\SOLPS::TOP.SNAPSHOT.DAB2').data(), 0, 2)
    sim._b2_neutral_densities = neutral_dens_data

    sim._velocities_parallel = np.swapaxes(conn.get('\SOLPS::TOP.SNAPSHOT.UA').data(), 0, 2)
    sim._velocities_radial = np.zeros((ni, nj, len(sim.species_list)))
    sim._velocities_toroidal = np.zeros((ni, nj, len(sim.species_list)))
    sim._velocities_cartesian = np.zeros((ni, nj, len(sim.species_list), 3), dtype=np.float64)

    ################################################
    # Calculate the species' velocity distribution #
    b2_neutral_i = 0  # counter for B2 neutrals
    for k, sp in enumerate(sim.species_list):

        # Identify the species based on its symbol
        symbol, charge = re.match(_SPECIES_REGEX, sp).groups()
        charge = int(charge)

        # If neutral and B" atomic density available,  use B2 density, otherwise use fluid species density.
        # if sim.b2_neutral_densities and charge == 0:
        if charge == 0:
            species_dens_data = sim.b2_neutral_densities[:, :, b2_neutral_i]
            b2_neutral_i += 1
        else:
            species_dens_data = sim.species_density[:, :, k]

        for i in range(ni):
            for j in range(nj):
                # Load grid basis vectors
                pv = mesh.poloidal_grid_basis[i, j, 0]  # parallel basis vector
                rv = mesh.poloidal_grid_basis[i, j, 1]  # radial basis vector

                # calculate field component ratios for velocity conversion
                bparallel = b_field_vectors[i, j, 0]
                btoroidal = b_field_vectors[i, j, 2]
                bplane = sqrt(bparallel**2 + btoroidal**2)
                parallel_to_toroidal_ratio = bparallel * btoroidal / (bplane**2)

                # Calculate toroidal and radial velocity components
                v_parallel = sim.velocities_parallel[i, j, k]  # straight from SOLPS 'UA' variable
                v_toroidal = v_parallel * parallel_to_toroidal_ratio
                sim.velocities_toroidal[i, j, k] = v_toroidal
                # Special case for edge of mesh, no radial velocity expected.
                try:
                    if species_dens_data[i, j] == 0:
                        v_radial = 0.0
                    else:
                        v_radial = sim.radial_particle_flux[i, j, k] / sim.radial_area[i, j] / species_dens_data[i, j]
                except IndexError:
                    v_radial = 0.0
                sim.velocities_radial[i, j, k] = v_radial

                # Convert velocities to cartesian coordinates
                vx = pv.x * v_parallel + rv.x * v_radial  # component of v along poloidal x
                vy = pv.y * v_parallel + rv.y * v_radial  # component of v along poloidal y
                sim.velocities_cartesian[i, j, k, :] = (vx, v_toroidal, vy)

    # Make Mesh Interpolator function for inside/outside mesh test.
    inside_outside_data = np.ones(mesh.num_tris)
    inside_outside = AxisymmetricMapper(Discrete2DMesh(mesh.vertex_coords, mesh.triangles, inside_outside_data))
    sim._inside_mesh = inside_outside

    ###############################
    # Load extra data from server #
    ###############################

    ####################
    # Integrated power #
    vol = np.swapaxes(conn.get('\SOLPS::TOP.SNAPSHOT.VOL').data(), 0, 1)  # TODO - this should be a mesh property
    linerad = np.swapaxes(conn.get('\SOLPS::TOP.SNAPSHOT.RQRAD').data(), 0, 2)
    linerad = np.sum(linerad, axis=2)
    brmrad = np.swapaxes(conn.get('\SOLPS::TOP.SNAPSHOT.RQBRM').data(), 0, 2)
    brmrad = np.sum(brmrad, axis=2)

    total_rad_data = np.zeros(vol.shape)
    ni, nj = vol.shape
    for i in range(ni):
        for j in range(nj):
            total_rad_data[i, j] = (linerad[i, j] + brmrad[i, j]) / vol[i, j]
    sim._total_rad = total_rad_data

    return sim


# # Code based on script by Felix Reimold (2016)
# @staticmethod
# def load_from_output_files(simulation_path, debug=False):
#     """
#     Load a SOLPS simulation from raw SOLPS output files.
#
#     Required files include:
#     * mesh description file (b2fgmtry)
#     * B2 plasma state (b2fstate)
#     * Eirene output file (fort.44)
#
#     :param str simulation_path: String path to simulation directory.
#     :rtype: SOLPSSimulation
#     """
#
#     if not os.path.isdir(simulation_path):
#         RuntimeError("simulation_path must be a valid directory")
#
#     mesh_file_path = os.path.join(simulation_path, 'b2fgmtry')
#     b2_state_file = os.path.join(simulation_path, 'b2fstate')
#     eirene_fort44_file = os.path.join(simulation_path, "fort.44")
#
#     if not os.path.isfile(mesh_file_path):
#         raise RuntimeError("No B2 b2fgmtry file found in SOLPS output directory")
#
#     if not(os.path.isfile(b2_state_file)):
#         RuntimeError("No B2 b2fstate file found in SOLPS output directory")
#
#     if not(os.path.isfile(eirene_fort44_file)):
#         RuntimeError("No EIRENE fort.44 file found in SOLPS output directory")
#
#     plasma = Plasma(name="Plasma")
#     velocity = lambda x, y, z: Vector3D(0, 0, 1)
#
#     # Load SOLPS mesh geometry
#     mesh = SOLPSMesh.load_from_files(mesh_file_path=mesh_file_path, debug=debug)
#
#     header_dict, sim_info_dict, mesh_data_dict = load_b2f_file(b2_state_file, debug=debug)
#
#     # TODO: add code to load SOLPS velocities and magnetic field from files
#
#     # Load electron species
#     electron_interp = _reshape_solps_data(mesh_data_dict['te']/Q, mesh)
#     electron_temp = AxisymmetricMapper(electron_interp)
#     electron_dens = AxisymmetricMapper(_reshape_solps_data(mesh_data_dict['ne'], mesh))
#     electrons = Maxwellian(electron_dens, electron_temp, velocity, electron_mass)
#     plasma.electron_distribution = electrons
#
#     ##########################################
#     # Load each plasma species in simulation #
#     ##########################################
#
#     for i in range(len(sim_info_dict['zn'])):
#
#         zn = int(sim_info_dict['zn'][i])  # Nuclear charge
#         am = float(sim_info_dict['am'][i])  # Atomic mass
#         charge = int(sim_info_dict['zamax'][i])  # Ionisation/charge
#         species = _popular_species[(zn, am)]
#
#         # load a species density distribution, use electrons for temperature, ignore flow velocity
#         signal_data = mesh_data_dict['na'][:, :, i]
#         triangle_data = _map_data_onto_triangles(signal_data)
#         dens = AxisymmetricMapper(Discrete2DMesh.instance(electron_interp, triangle_data))
#         distribution = Maxwellian(dens, electron_temp, velocity, species.atomic_weight * atomic_mass)
#         plasma.composition.add(Species(species, charge, distribution))
#
#     # Make Mesh Interpolator function for inside/outside mesh test.
#     inside_outside = AxisymmetricMapper(Discrete2DMesh.instance(electron_interp, np.ones(mesh.num_tris)))
#
#     # plasma.inside_outside = inside_outside
#
#     sim = SOLPSSimulation(mesh, plasma)
#     sim._inside_mesh = inside_outside
#
#     # Load total radiated power from EIRENE output file
#     eirene = Eirene(eirene_fort44_file)
#     sim._eirene = eirene
#
#     # Note EIRENE data grid is slightly smaller than SOLPS grid, for example (98, 38) => (96, 36)
#     # Need to pad EIRENE data to fit inside larger B2 array
#     nx = mesh.nx
#     ny = mesh.ny
#     eradt_raw_data = eirene.eradt.sum(2)
#     eradt_data = np.zeros((nx, ny))
#     eradt_data[1:nx-1, 1:ny-1] = eradt_raw_data
#     eradt_data = _map_data_onto_triangles(eradt_data)
#     sim._total_rad = AxisymmetricMapper(Discrete2DMesh.instance(electron_interp, eradt_data))
#
#     return sim
