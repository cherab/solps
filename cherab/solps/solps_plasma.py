
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

    def __init__(self, mesh, plasma):

        self.mesh = mesh
        self.plasma = plasma

        self._inside_mesh = None
        self._total_rad = None
        self._b_field_vectors = None
        self._parallel_velocities = None
        self._radial_velocities = None
        self._eirene_model = None
        self._b2_model = None
        self._eirene = None

    @property
    def inside_mesh(self):
        """
        Function3D for testing if point p is inside the simulation mesh.
        """
        if not self._inside_mesh:
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
        if not self._total_rad:
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
        if not self._parallel_velocities:
            raise RuntimeError("Parallel velocities not available for this simulation")
        else:
            return self._parallel_velocities

    @property
    def radial_velocities(self):
        """
        Calculated radial velocity components for each species.
        """
        if not self._parallel_velocities:
            raise RuntimeError("Radial velocities not available for this simulation")
        else:
            return self._radial_velocities

    @property
    def b_field(self):
        """
        Magnetic B field at each mesh cell.
        """
        if not self._b_field_vectors:
            raise RuntimeError("Parallel velocities not available for this simulation")
        else:
            return self._b_field_vectors

    @property
    def eirene_simulation(self):
        """
        Data from an underlying EIRENE simulation.

        :rtype: Eirene
        """
        if not self._eirene:
            raise RuntimeError("EIRENE simulation data not available for this SOLPS simulation")
        else:
            return self._eirene

    @staticmethod
    def load_from_mdsplus(mds_server, ref_number, use_b2_neutral_densities=False):
        """
        Load a SOLPS simulation from MDSplus server.

        :param str mds_server: Server address.
        :param int ref_number: Simulation reference number.
        :rtype: SOLPSSimulation
        """

        from MDSplus import Connection as MDSConnection

        # Setup connection to server
        conn = MDSConnection(mds_server)
        conn.openTree('solps', ref_number)

        plasma = Plasma(name="Plasma")

        # Load SOLPS mesh geometry and lookup arrays
        mesh = SOLPSMesh.load_from_mdsplus(conn)
        tri_index_lookup = mesh.triangle_index_lookup
        tri_to_grid = mesh.triangle_to_grid_map

        ni = mesh.nx
        nj = mesh.ny

        ##########################
        # Magnetic field vectors #
        raw_b_field = np.swapaxes(conn.get('\SOLPS::TOP.SNAPSHOT.B').data(), 0, 2)
        b_field_vectors = np.zeros((ni, nj, 3))
        for i in range(ni):
            for j in range(nj):
                bparallel = raw_b_field[i, j, 0]
                bradial = raw_b_field[i, j, 1]
                btoroidal = raw_b_field[i, j, 2]

                pv = mesh.poloidal_grid_basis[i, j, 0]  # parallel basis vector
                rv = mesh.poloidal_grid_basis[i, j, 1]  # radial basis vector

                bx = pv.x * bparallel + rv.x * bradial  # component of B along poloidal x
                by = pv.y * bparallel + rv.y * bradial  # component of B along poloidal y
                b_field_vectors[i, j] = (bx, btoroidal, by)

        b_field = SOLPSVectorFunction3D(tri_index_lookup, tri_to_grid, b_field_vectors)

        # Load electron species
        electron_interp = _load_solps_vertex_signal(conn, '\SOLPS::TOP.SNAPSHOT.TE', mesh)
        electron_temp = AxisymmetricMapper(electron_interp)
        electron_dens = AxisymmetricMapper(_load_solps_vertex_signal(conn, '\SOLPS::TOP.SNAPSHOT.NE', mesh, template_interpolator=electron_interp))
        electron_velocity = lambda x, y, z: Vector3D(0, 0, 0)
        electrons = Maxwellian(electron_dens, electron_temp, electron_velocity, electron_mass)
        plasma.electron_distribution = electrons

        ############################
        # Load each plasma species #
        ############################

        species_list = conn.get('\SOLPS::TOP.IDENT.SPECIES').data().decode('UTF-8').split()  # ['D0', 'D+1', 'C0', 'C+1', ...
        species_density = np.swapaxes(conn.get('\SOLPS::TOP.SNAPSHOT.NA').data(), 0, 2)
        species_parallel_velocity = np.swapaxes(conn.get('\SOLPS::TOP.SNAPSHOT.UA').data(), 0, 2)
        rad_par_flux = np.swapaxes(conn.get('\SOLPS::TOP.SNAPSHOT.FNAY').data(), 0, 2)  # radial particle flux
        radial_area = np.swapaxes(conn.get('\SOLPS::TOP.SNAPSHOT.SY').data(), 0, 1)  # radial contact area
        radial_velocities = np.zeros((ni, nj, len(species_list)))

        # Load the neutral atom density from B2
        if use_b2_neutral_densities:
            neutral_dens_raw = conn.get('\SOLPS::TOP.SNAPSHOT.DAB2').data()
            if neutral_dens_raw:
                neutral_dens_data = np.swapaxes(conn.get('\SOLPS::TOP.SNAPSHOT.DAB2').data(), 0, 2)
                neutral_i = 0  # counter for neutrals
            else:
                print("Could not load B2 neutrals 'DAB2'.")
                use_b2_neutral_densities = False

        for k, sp in enumerate(species_list):

            # Identify the species based on its symbol
            symbol, charge = re.match(_SPECIES_REGEX, sp).groups()
            charge = int(charge)
            species_type = _species_symbol_map[symbol]

            if use_b2_neutral_densities:
                # If neutral, use B2 atomic density, otherwise use fluid species density.
                if charge == 0:
                    species_dens_data = neutral_dens_data[:, :, ni]
                    neutral_i += 1
                else:
                    species_dens_data = species_density[:, :, k]
            else:
                species_dens_data = species_density[:, :, k]

            triangle_data = _map_data_onto_triangles(species_dens_data)
            dens = AxisymmetricMapper(Discrete2DMesh.instance(electron_interp, triangle_data))
            # Create the density lookup function
            # dens = SOLPSFunction3D(tri_index_lookup, tri_to_grid, species_dens_data)

            ###########################################
            # Load the species' velocity distribution #
            velocity_field_vectors = np.zeros((ni, nj, 3), dtype=np.float64)
            for i in range(ni):
                for j in range(nj):
                    # Load grid basis vectors
                    pv = mesh.poloidal_grid_basis[i, j, 0]  # parallel basis vector
                    rv = mesh.poloidal_grid_basis[i, j, 1]  # radial basis vector

                    # calculate field component ratios for velocity conversion
                    bparallel = raw_b_field[i, j, 0]
                    btoroidal = raw_b_field[i, j, 2]
                    bplane = sqrt(bparallel**2 + btoroidal**2)
                    parallel_to_toroidal_ratio = bparallel * btoroidal / (bplane**2)

                    v_parallel = species_parallel_velocity[i, j, k]  # straight from SOLPS 'UA' variable
                    v_toroidal = v_parallel * parallel_to_toroidal_ratio
                    # Special case for edge of mesh, no radial velocity expected.
                    try:
                        v_radial = rad_par_flux[i, j, k] / radial_area[i, j] / species_density[i, j, k]
                    except IndexError:
                        v_radial = 0.0
                    radial_velocities[i, j, k] = v_radial

                    vx = pv.x * v_parallel + rv.x * v_radial  # component of v along poloidal x
                    vy = pv.y * v_parallel + rv.y * v_radial  # component of v along poloidal y
                    velocity_field_vectors[i, j, :] = (vx, v_toroidal, vy)

            # Create the velocity vector lookup function
            velocity = SOLPSVectorFunction3D(tri_index_lookup, tri_to_grid, velocity_field_vectors)

            distribution = Maxwellian(dens, electron_temp, velocity, species_type.atomic_weight * atomic_mass)
            plasma.composition.add(Species(species_type, charge, distribution))

        # Make Mesh Interpolator function for inside/outside mesh test.
        inside_outside_data = np.ones(mesh.num_tris)
        inside_outside = AxisymmetricMapper(Discrete2DMesh.instance(electron_interp, inside_outside_data))
        # TODO - need to check its ok to remove this feature
        # plasma.inside_outside = inside_outside

        # Creat simulation instance and add calculated parameters
        sim = SOLPSSimulation(mesh, plasma)
        sim._inside_mesh = inside_outside
        sim._b_field_vectors = b_field_vectors
        sim._parallel_velocities = species_parallel_velocity
        sim._radial_velocities = radial_velocities

        ###############################
        # Load extra data from server #
        ###############################

        ####################
        # Integrated power #
        vol = np.swapaxes(conn.get('\SOLPS::TOP.SNAPSHOT.VOL').data(), 0, 1)
        linerad = np.swapaxes(conn.get('\SOLPS::TOP.SNAPSHOT.RQRAD').data(), 0, 2)
        linerad = np.sum(linerad, axis=2)
        brmrad = np.swapaxes(conn.get('\SOLPS::TOP.SNAPSHOT.RQBRM').data(), 0, 2)
        brmrad = np.sum(brmrad, axis=2)

        total_rad_data = np.zeros(vol.shape)
        ni, nj = vol.shape
        for i in range(ni):
            for j in range(nj):
                total_rad_data[i, j] = (linerad[i, j] + brmrad[i, j]) / vol[i, j]
        tri_total_rad = _map_data_onto_triangles(total_rad_data)
        total_rad = AxisymmetricMapper(Discrete2DMesh.instance(electron_interp, tri_total_rad))
        sim._total_rad = total_rad
        # TODO - need to resolve difference between cherab type plasma interface and storing
        # underlying SOLPS data in original form.
        sim.total_rad_data = total_rad_data

        return sim

    # Code based on script by Felix Reimold (2016)
    @staticmethod
    def load_from_output_files(simulation_path, debug=False):
        """
        Load a SOLPS simulation from raw SOLPS output files.

        Required files include:
        * mesh description file (b2fgmtry)
        * B2 plasma state (b2fstate)
        * Eirene output file (fort.44)

        :param str simulation_path: String path to simulation directory.
        :rtype: SOLPSSimulation
        """

        if not os.path.isdir(simulation_path):
            RuntimeError("simulation_path must be a valid directory")

        mesh_file_path = os.path.join(simulation_path, 'b2fgmtry')
        b2_state_file = os.path.join(simulation_path, 'b2fstate')
        eirene_fort44_file = os.path.join(simulation_path, "fort.44")

        if not os.path.isfile(mesh_file_path):
            raise RuntimeError("No B2 b2fgmtry file found in SOLPS output directory")

        if not(os.path.isfile(b2_state_file)):
            RuntimeError("No B2 b2fstate file found in SOLPS output directory")

        if not(os.path.isfile(eirene_fort44_file)):
            RuntimeError("No EIRENE fort.44 file found in SOLPS output directory")

        plasma = Plasma(name="Plasma")
        velocity = lambda x, y, z: Vector3D(0, 0, 1)

        # Load SOLPS mesh geometry
        mesh = SOLPSMesh.load_from_files(mesh_file_path=mesh_file_path, debug=debug)

        header_dict, sim_info_dict, mesh_data_dict = load_b2f_file(b2_state_file, debug=debug)

        # TODO: add code to load SOLPS velocities and magnetic field from files

        # Load electron species
        electron_interp = _reshape_solps_data(mesh_data_dict['te']/Q, mesh)
        electron_temp = AxisymmetricMapper(electron_interp)
        electron_dens = AxisymmetricMapper(_reshape_solps_data(mesh_data_dict['ne'], mesh))
        electrons = Maxwellian(electron_dens, electron_temp, velocity, electron_mass)
        plasma.electron_distribution = electrons

        ##########################################
        # Load each plasma species in simulation #
        ##########################################

        for i in range(len(sim_info_dict['zn'])):

            zn = int(sim_info_dict['zn'][i])  # Nuclear charge
            am = float(sim_info_dict['am'][i])  # Atomic mass
            charge = int(sim_info_dict['zamax'][i])  # Ionisation/charge
            species = _popular_species[(zn, am)]

            # load a species density distribution, use electrons for temperature, ignore flow velocity
            signal_data = mesh_data_dict['na'][:, :, i]
            triangle_data = _map_data_onto_triangles(signal_data)
            dens = AxisymmetricMapper(Discrete2DMesh.instance(electron_interp, triangle_data))
            distribution = Maxwellian(dens, electron_temp, velocity, species.atomic_weight * atomic_mass)
            plasma.composition.add(Species(species, charge, distribution))

        # Make Mesh Interpolator function for inside/outside mesh test.
        inside_outside = AxisymmetricMapper(Discrete2DMesh.instance(electron_interp, np.ones(mesh.num_tris)))

        # plasma.inside_outside = inside_outside

        sim = SOLPSSimulation(mesh, plasma)
        sim._inside_mesh = inside_outside

        # Load total radiated power from EIRENE output file
        eirene = Eirene(eirene_fort44_file)
        sim._eirene = eirene

        # Note EIRENE data grid is slightly smaller than SOLPS grid, for example (98, 38) => (96, 36)
        # Need to pad EIRENE data to fit inside larger B2 array
        nx = mesh.nx
        ny = mesh.ny
        eradt_raw_data = eirene.eradt.sum(2)
        eradt_data = np.zeros((nx, ny))
        eradt_data[1:nx-1, 1:ny-1] = eradt_raw_data
        eradt_data = _map_data_onto_triangles(eradt_data)
        sim._total_rad = AxisymmetricMapper(Discrete2DMesh.instance(electron_interp, eradt_data))

        return sim

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
