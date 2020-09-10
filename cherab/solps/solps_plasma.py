
# Copyright 2016-2018 Euratom
# Copyright 2016-2018 United Kingdom Atomic Energy Authority
# Copyright 2016-2018 Centro de Investigaciones Energéticas, Medioambientales y Tecnológicas
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

import pickle
import numpy as np
import matplotlib.pyplot as plt
from scipy.constants import atomic_mass, electron_mass

# Raysect imports
from raysect.core import translate, Point3D, Vector3D, Node, AffineMatrix3D
from raysect.primitive import Cylinder
from raysect.optical import Spectrum

# CHERAB core imports
from cherab.core import Plasma, Species, Maxwellian
from cherab.core.math.mappers import AxisymmetricMapper, VectorAxisymmetricMapper
from cherab.core.atomic.elements import lookup_isotope, lookup_element

# This SOLPS package imports
from cherab.solps.eirene import Eirene
from .solps_2d_functions import SOLPSFunction2D, SOLPSVectorFunction2D
from .mesh_geometry import SOLPSMesh


class SOLPSSimulation:

    def __init__(self, mesh, species_list):

        # Mesh and species_list cannot be changed after initialisation
        if isinstance(mesh, SOLPSMesh):
            self._mesh = mesh
        else:
            raise ValueError('Argument "mesh" must be a SOLPSMesh instance.')

        # Make Mesh Interpolator function for inside/outside mesh test.
        inside_outside_data = np.ones((self._mesh.ny, self._mesh.nx))
        self._inside_mesh = SOLPSFunction2D(mesh.vertex_coordinates, mesh.triangles, mesh.triangle_to_grid_map, inside_outside_data)

        # Creating a sample SOLPSVectorFunction2D for KDtree to use later
        sample_vector = np.ones((3, self._mesh.ny, self._mesh.nx))
        self._sample_vector_f2d = SOLPSVectorFunction2D(mesh.vertex_coordinates, mesh.triangles, mesh.triangle_to_grid_map, sample_vector)

        if not len(species_list):
            raise ValueError('Argument "species_list" must contain at least one species.')
        self._species_list = tuple(species_list)  # adding additional species is not allowed
        self._neutral_list = tuple([sp for sp in self._species_list if sp[1] == 0])

        self._electron_temperature = None
        self._electron_temperature_f2d = None
        self._electron_temperature_f3d = None
        self._electron_density = None
        self._electron_density_f2d = None
        self._electron_density_f3d = None
        self._ion_temperature = None
        self._ion_temperature_f2d = None
        self._ion_temperature_f3d = None
        self._neutral_temperature = None
        self._neutral_temperature_f2d = None
        self._neutral_temperature_f3d = None
        self._species_density = None
        self._species_density_f2d = None
        self._species_density_f3d = None
        self._velocities = None
        self._velocities_cartesian = None
        self._velocities_cartesian_f2d = None
        self._velocities_cartesian_f3d = None
        self._total_radiation = None
        self._total_radiation_f2d = None
        self._total_radiation_f3d = None
        self._b_field = None
        self._b_field_cartesian = None
        self._b_field_cartesian_f2d = None
        self._b_field_cartesian_f3d = None
        self._eirene_model = None  # what is this for?
        self._b2_model = None  # what is this for?
        self._eirene = None  # do we need this in SOLPSSimulation?

    @property
    def mesh(self):
        """
        SOLPSMesh instance.
        :return:
        """
        return self._mesh

    @property
    def species_list(self):
        """
        Tuple of species elements in the form (species name, charge).
        :return:
        """
        return self._species_list

    @property
    def electron_temperature(self):
        """
        Simulated electron temperatures at each mesh cell.
        :return:
        """
        return self._electron_temperature

    @property
    def electron_temperature_f2d(self):
        """
        Simulated electron temperatures at each mesh cell.
        :return:
        """
        return self._electron_temperature_f2d

    @property
    def electron_temperature_f3d(self):
        """
        Simulated electron temperatures at each mesh cell.
        :return:
        """
        return self._electron_temperature_f3d

    @electron_temperature.setter
    def electron_temperature(self, value):
        _check_array("electron_temperature", value, (self.mesh.ny, self.mesh.nx))
        self._electron_temperature = value
        self._electron_temperature_f2d = SOLPSFunction2D.instance(self._inside_mesh, value)
        self._electron_temperature_f3d = AxisymmetricMapper(self._electron_temperature_f2d)

    @property
    def ion_temperature(self):
        """
        Simulated ion temperatures at each mesh cell.
        :return:
        """
        return self._ion_temperature

    @property
    def ion_temperature_f2d(self):
        """
        Simulated ion temperatures at each mesh cell.
        :return:
        """
        return self._ion_temperature_f2d

    @property
    def ion_temperature_f3d(self):
        """
        Simulated ion temperatures at each mesh cell.
        :return:
        """
        return self._ion_temperature_f3d

    @ion_temperature.setter
    def ion_temperature(self, value):
        _check_array("ion_temperature", value, (self.mesh.ny, self.mesh.nx))
        self._ion_temperature = value
        self._ion_temperature_f2d = SOLPSFunction2D.instance(self._inside_mesh, value)
        self._ion_temperature_f3d = AxisymmetricMapper(self._ion_temperature_f2d)

    @property
    def neutral_temperature(self):
        """
        Array of neutral atom (effective) temperature at each mesh cell.
        :return:
        """
        return self._neutral_temperature

    @property
    def neutral_temperature_f2d(self):
        """
        Array of neutral atom (effective) temperature at each mesh cell.
        :return:
        """
        return self._neutral_temperature_f2d

    @property
    def neutral_temperature_f3d(self):
        """
        Array of neutral atom (effective) temperature at each mesh cell.
        :return:
        """
        return self._neutral_temperature_f3d

    @neutral_temperature.setter
    def neutral_temperature(self, value):
        _check_array("neutral_temperature", value, (len(self._neutral_list), self.mesh.ny, self.mesh.nx))
        self._neutral_temperature = value
        self._neutral_temperature_f2d = {}
        self._neutral_temperature_f3d = {}
        for k, sp in enumerate(self._neutral_list):
            self._neutral_temperature_f2d[k] = SOLPSFunction2D.instance(self._inside_mesh, value[k])
            self._neutral_temperature_f2d[sp] = self._neutral_temperature_f2d[k]
            self._neutral_temperature_f3d[k] = AxisymmetricMapper(self._neutral_temperature_f2d[k])
            self._neutral_temperature_f3d[sp] = self._neutral_temperature_f3d[k]

    @property
    def electron_density(self):
        """
        Simulated electron densities at each mesh cell.
        :return:
        """
        return self._electron_density

    @property
    def electron_density_f2d(self):
        """
        Simulated electron densities at each mesh cell.
        :return:
        """
        return self._electron_density_f2d

    @property
    def electron_density_f3d(self):
        """
        Simulated electron densities at each mesh cell.
        :return:
        """
        return self._electron_density_f3d

    @electron_density.setter
    def electron_density(self, value):
        _check_array("electron_density", value, (self.mesh.ny, self.mesh.nx))
        self._electron_density = value
        self._electron_density_f2d = SOLPSFunction2D.instance(self._inside_mesh, value)
        self._electron_density_f3d = AxisymmetricMapper(self._electron_density_f2d)

    @property
    def species_density(self):
        """
        Array of species densities at each mesh cell.
        :return:
        """
        return self._species_density

    @property
    def species_density_f2d(self):
        """
        Array of species densities at each mesh cell.
        :return:
        """
        return self._species_density_f2d

    @property
    def species_density_f3d(self):
        """
        Array of species densities at each mesh cell.
        :return:
        """
        return self._species_density_f3d

    @species_density.setter
    def species_density(self, value):
        _check_array("species_density", value, (len(self._species_list), self.mesh.ny, self.mesh.nx))
        self._species_density = value
        self._species_density_f2d = {}
        self._species_density_f3d = {}
        for k, sp in enumerate(self._species_list):
            self._species_density_f2d[k] = SOLPSFunction2D.instance(self._inside_mesh, value[k])
            self._species_density_f2d[sp] = self._species_density_f2d[k]
            self._species_density_f3d[k] = AxisymmetricMapper(self._species_density_f2d[k])
            self._species_density_f3d[sp] = self._species_density_f3d[k]

    @property
    def velocities(self):
        """
        Velocities in poloidal coordinates (v_poloidal, v_radial, v_toroidal) for each species densities at each mesh cell.
        :return:
        """
        return self._velocities

    @velocities.setter
    def velocities(self, value):
        _check_array("velocities", value, (len(self.species_list), 3, self.mesh.ny, self.mesh.nx))

        # Converting to Cartesian coordinates
        velocities_cartesian = np.zeros(value.shape)
        velocities_cartesian[:, 2] = value[:, 2]
        for k in range(value.shape[0]):
            velocities_cartesian[k, :2] = self.mesh.to_cartesian(value[k, :2])

        self._velocities = value
        self._velocities_cartesian = velocities_cartesian
        self._velocities_cartesian_f2d = {}
        self._velocities_cartesian_f3d = {}
        for k, sp in enumerate(self._species_list):
            self._velocities_cartesian_f2d[k] = SOLPSVectorFunction2D.instance(self._sample_vector_f2d, velocities_cartesian[k])
            self._velocities_cartesian_f2d[sp] = self._velocities_cartesian_f2d[k]
            self._velocities_cartesian_f3d[k] = VectorAxisymmetricMapper(self._velocities_cartesian_f2d[k])
            self._velocities_cartesian_f3d[sp] = self._velocities_cartesian_f3d[k]

    @property
    def velocities_cartesian(self):
        """
        Velocities in Cartesian (v_r, v_z, v_toroidal) coordinates for each species densities at each mesh cell.
        :return:
        """
        return self._velocities_cartesian

    @property
    def velocities_cartesian_f2d(self):
        """
        Velocities in Cartesian (v_r, v_z, v_toroidal) coordinates for each species densities at each mesh cell.
        :return:
        """
        return self._velocities_cartesian_f2d

    @property
    def velocities_cartesian_f3d(self):
        """
        Velocities in Cartesian (v_r, v_z, v_toroidal) coordinates for each species densities at each mesh cell.
        :return:
        """
        return self._velocities_cartesian_f3d

    @velocities_cartesian.setter
    def velocities_cartesian(self, value):
        _check_array("velocities_cartesian", value, (len(self.species_list), 3, self.mesh.ny, self.mesh.nx))

        # Converting to poloidal coordinates
        velocities = np.zeros(value.shape)
        velocities[:, 2] = value[:, 2]
        for k in range(value.shape[0]):
            velocities[k, :2] = self.mesh.to_poloidal(value[k, :2])

        self._velocities_cartesian = value
        self._velocities = velocities
        self._velocities_cartesian_f2d = {}
        self._velocities_cartesian_f3d = {}
        for k, sp in enumerate(self._species_list):
            self._velocities_cartesian_f2d[k] = SOLPSVectorFunction2D.instance(self._sample_vector_f2d, value[k])
            self._velocities_cartesian_f2d[sp] = self._velocities_cartesian_f2d[k]
            self._velocities_cartesian_f3d[k] = VectorAxisymmetricMapper(self._velocities_cartesian_f2d[k])
            self._velocities_cartesian_f3d[sp] = self._velocities_cartesian_f3d[k]

    @property
    def inside_mesh(self):
        """
        Function2D for testing if point p is inside the simulation mesh.
        """
        return self._inside_mesh

    @property
    def inside_volume_mesh(self):
        """
        Function3D for testing if point p is inside the simulation mesh.
        """
        return AxisymmetricMapper(self._inside_mesh)

    @property
    def total_radiation(self):
        """
        Total radiation

        This is not calculated from the CHERAB emission models, instead it comes from the SOLPS output data.
        Is calculated from the sum of all integrated line emission and all Bremmstrahlung. The signals used are 'RQRAD'
        and 'RQBRM'. Final output is in W/str?
        """
        if self._total_radiation is None:
            raise RuntimeError("Total radiation not available for this simulation.")
        else:
            return self._total_radiation

    @property
    def total_radiation_f2d(self):
        """
        Total radiation

        This is not calculated from the CHERAB emission models, instead it comes from the SOLPS output data.
        Is calculated from the sum of all integrated line emission and all Bremmstrahlung. The signals used are 'RQRAD'
        and 'RQBRM'. Final output is in W/str?
        """
        if self._total_radiation_f2d is None:
            raise RuntimeError("Total radiation not available for this simulation.")
        else:
            return self._total_radiation_f2d

    @property
    def total_radiation_f3d(self):
        """
        Total radiation

        This is not calculated from the CHERAB emission models, instead it comes from the SOLPS output data.
        Is calculated from the sum of all integrated line emission and all Bremmstrahlung. The signals used are 'RQRAD'
        and 'RQBRM'. Final output is in W/str?
        """
        if self._total_radiation_f3d is None:
            raise RuntimeError("Total radiation not available for this simulation.")
        else:
            return self._total_radiation_f3d

    @total_radiation.setter
    def total_radiation(self, value):
        _check_array("total_radiation", value, (self.mesh.ny, self.mesh.nx))
        self._total_radiation = value
        self._total_radiation_f2d = SOLPSFunction2D.instance(self._inside_mesh, value)
        self._total_radiation_f3d = AxisymmetricMapper(self._total_radiation_f2d)

    @property
    def b_field(self):
        """
        Magnetic B field at each mesh cell in mesh cell coordinates (b_poloidal, b_radial, b_toroidal).
        """
        if self._b_field is None:
            raise RuntimeError("Magnetic field not available for this simulation.")
        else:
            return self._b_field

    @b_field.setter
    def b_field(self, value):
        _check_array("b_field", value, (3, self.mesh.ny, self.mesh.nx))

        # Converting to cartesian system
        b_field_cartesian = np.zeros(value.shape)
        b_field_cartesian[2] = value[2]
        b_field_cartesian[:2] = self.mesh.to_cartesian(value[:2])

        self._b_field_cartesian = b_field_cartesian
        self._b_field = value
        self._b_field_cartesian_f2d = SOLPSVectorFunction2D.instance(self._sample_vector_f2d, b_field_cartesian)
        self._b_field_cartesian_f3d = VectorAxisymmetricMapper(self._b_field_cartesian_f2d)

    @property
    def b_field_cartesian(self):
        """
        Magnetic B field at each mesh cell in Cartesian coordinates (B_r, B_z, B_toroidal).
        """
        if self._b_field_cartesian is None:
            raise RuntimeError("Magnetic field not available for this simulation.")
        else:
            return self._b_field_cartesian

    @property
    def b_field_cartesian_f2d(self):
        """
        Magnetic B field at each mesh cell in Cartesian coordinates (B_r, B_z, B_toroidal).
        """
        if self._b_field_cartesian_f2d is None:
            raise RuntimeError("Magnetic field not available for this simulation.")
        else:
            return self._b_field_cartesian_f2d

    @property
    def b_field_cartesian_f3d(self):
        """
        Magnetic B field at each mesh cell in Cartesian coordinates (B_r, B_z, B_toroidal).
        """
        if self._b_field_cartesian_f3d is None:
            raise RuntimeError("Magnetic field not available for this simulation.")
        else:
            return self._b_field_cartesian_f3d

    @b_field_cartesian.setter
    def b_field_cartesian(self, value):
        _check_array("b_field_cartesian", value, (3, self.mesh.ny, self.mesh.nx))

        # Converting to poloidal system
        b_field = np.zeros(value.shape)
        b_field[2] = value[2]
        b_field[:2] = self.mesh.to_poloidal(value[:2])

        self._b_field = b_field
        self._b_field_cartesian = value
        self._b_field_cartesian_f2d = SOLPSVectorFunction2D.instance(self._sample_vector_f2d, value)
        self._b_field_cartesian_f3d = VectorAxisymmetricMapper(self._b_field_cartesian_f2d)

    @property
    def eirene_simulation(self):
        """
        Data from an underlying EIRENE simulation.

        :rtype: Eirene
        """
        if self._eirene is None:
            raise RuntimeError("EIRENE simulation data not available for this SOLPS simulation.")
        else:
            return self._eirene

    @eirene_simulation.setter
    def eirene_simulation(self, value):
        if not isinstance(value, Eirene):
            raise ValueError('Attribute "eirene_simulation" must be an Eirene instance.')

        self._eirene = value

    def __getstate__(self):
        state = {
            'mesh': self._mesh.__getstate__(),
            'species_list': self._species_list,
            'inside_mesh': self._inside_mesh,
            'sample_vector_f2d': self._sample_vector_f2d,
            'electron_temperature': self._electron_temperature,
            'ion_temperature': self._ion_temperature,
            'neutral_temperature': self._neutral_temperature,
            'electron_density': self._electron_density,
            'species_density': self._species_density,
            'velocities_cartesian': self._velocities_cartesian,
            'total_radiation': self._total_radiation,
            'b_field_cartesian': self._b_field_cartesian,
            'eirene_model': self._eirene_model,
            'b2_model': self._b2_model,
            'eirene': self._eirene
        }
        return state

    def __setstate__(self, state):
        self._mesh = SOLPSMesh(**state['mesh'])
        self._species_list = state['species_list']
        self._neutral_list = tuple([sp for sp in self._species_list if sp[1] == 0])
        self._inside_mesh = state['inside_mesh']
        self._sample_vector_f2d = state['sample_vector_f2d']
        self._electron_temperature = None
        self._electron_temperature_f2d = None
        self._electron_temperature_f3d = None
        self._electron_density = None
        self._electron_density_f2d = None
        self._electron_density_f3d = None
        self._ion_temperature = None
        self._ion_temperature_f2d = None
        self._ion_temperature_f3d = None
        self._neutral_temperature = None
        self._neutral_temperature_f2d = None
        self._neutral_temperature_f3d = None
        self._species_density = None
        self._species_density_f2d = None
        self._species_density_f3d = None
        self._velocities = None
        self._velocities_cartesian = None
        self._velocities_cartesian_f2d = None
        self._velocities_cartesian_f3d = None
        self._total_radiation = None
        self._total_radiation_f2d = None
        self._total_radiation_f3d = None
        self._b_field = None
        self._b_field_cartesian = None
        self._b_field_cartesian_f2d = None
        self._b_field_cartesian_f3d = None
        if state['electron_temperature'] is not None:
            self.electron_temperature = state['electron_temperature']  # will create _f2d() and _f3d()
        if state['ion_temperature'] is not None:
            self.ion_temperature = state['ion_temperature']
        if state['neutral_temperature'] is not None:
            self.neutral_temperature = state['neutral_temperature']
        if state['electron_density'] is not None:
            self.electron_density = state['electron_density']
        if state['species_density'] is not None:
            self.species_density = state['species_density']
        if state['velocities_cartesian'] is not None:
            self.velocities_cartesian = state['velocities_cartesian']
        if state['total_radiation'] is not None:
            self.total_radiation = state['total_radiation']
        if state['b_field_cartesian'] is not None:
            self.b_field_cartesian = state['b_field_cartesian']
        self._eirene_model = state['eirene_model']
        self._b2_model = state['b2_model']
        self._eirene = state['eirene']

    def save(self, filename):

        file_handle = open(filename, 'wb')
        pickle.dump(self.__getstate__(), file_handle)
        file_handle.close()

    # def plot_electrons(self):
    #     """ Make a plot of the electron temperature and density in the SOLPS mesh plane. """
    #
    #     me = self.mesh.mesh_extent
    #     xl, xu = (me['minr'], me['maxr'])
    #     yl, yu = (me['minz'], me['maxz'])
    #
    #     te_samples = np.zeros((500, 500))
    #     ne_samples = np.zeros((500, 500))
    #
    #     xrange = np.linspace(xl, xu, 500)
    #     yrange = np.linspace(yl, yu, 500)
    #
    #     plasma = self.plasma
    #     for i, x in enumerate(xrange):
    #         for j, y in enumerate(yrange):
    #             ne_samples[j, i] = plasma.electron_distribution.density(x, 0.0, y)
    #             te_samples[j, i] = plasma.electron_distribution.effective_temperature(x, 0.0, y)
    #
    #     plt.figure()
    #     plt.imshow(ne_samples, extent=[xl, xu, yl, yu], origin='lower')
    #     plt.colorbar()
    #     plt.xlim(xl, xu)
    #     plt.ylim(yl, yu)
    #     plt.title("electron density")
    #     plt.figure()
    #     plt.imshow(te_samples, extent=[xl, xu, yl, yu], origin='lower')
    #     plt.colorbar()
    #     plt.xlim(xl, xu)
    #     plt.ylim(yl, yu)
    #     plt.title("electron temperature")
    #
    # def plot_species_density(self, species, ionisation):
    #     """
    #     Make a plot of the requested species density in the SOLPS mesh plane.
    #
    #     :param Element species: The species to plot.
    #     :param int ionisation: The charge state of the species to plot.
    #     """
    #
    #     species_dist = self.plasma.get_species(species, ionisation)
    #
    #     me = self.mesh.mesh_extent
    #     xl, xu = (me['minr'], me['maxr'])
    #     yl, yu = (me['minz'], me['maxz'])
    #     species_samples = np.zeros((500, 500))
    #
    #     xrange = np.linspace(xl, xu, 500)
    #     yrange = np.linspace(yl, yu, 500)
    #
    #     for i, x in enumerate(xrange):
    #         for j, y in enumerate(yrange):
    #             species_samples[j, i] = species_dist.distribution.density(x, 0.0, y)
    #
    #     plt.figure()
    #     plt.imshow(species_samples, extent=[xl, xu, yl, yu], origin='lower')
    #     plt.colorbar()
    #     plt.xlim(xl, xu)
    #     plt.ylim(yl, yu)
    #     plt.title("Species {} - stage {} - density".format(species.name, ionisation))
    #
    # def plot_pec_emission_lines(self, emission_lines, title="", vmin=None, vmax=None, log=False):
    #     """
    #     Make a plot of the given PEC emission lines
    #
    #     :param list emission_lines: List of PEC emission lines.
    #     :param str title: The title of the plot.
    #     :param float vmin: The minimum value for clipping the plots (default=None).
    #     :param float vmax: The maximum value for clipping the plots (default=None).
    #     :param bool log: Toggle a log plot for the data (default=False).
    #     """
    #
    #     me = self.mesh.mesh_extent
    #     xl, xu = (me['minr'], me['maxr'])
    #     yl, yu = (me['minz'], me['maxz'])
    #     emission_samples = np.zeros((500, 500))
    #
    #     xrange = np.linspace(xl, xu, 500)
    #     yrange = np.linspace(yl, yu, 500)
    #
    #     for i, x in enumerate(xrange):
    #         for j, y in enumerate(yrange):
    #             for emitter in emission_lines:
    #                 emission_samples[j, i] += emitter.emission(Point3D(x, 0.0, y), Vector3D(0, 0, 0), Spectrum(350, 700, 800)).total()
    #
    #     if log:
    #         emission_samples = np.log(emission_samples)
    #     plt.figure()
    #     plt.imshow(emission_samples, extent=[xl, xu, yl, yu], origin='lower', vmin=vmin, vmax=vmax)
    #     plt.colorbar()
    #     plt.xlim(xl, xu)
    #     plt.ylim(yl, yu)
    #     plt.title(title)
    #
    # def plot_radiated_power(self):
    #     """
    #     Make a plot of the given PEC emission lines
    #
    #     :param list emission_lines: List of PEC emission lines.
    #     :param str title: The title of the plot.
    #     """
    #
    #     mesh = self.mesh
    #     me = mesh.mesh_extent
    #     total_rad = self.total_radiation
    #
    #     xl, xu = (me['minr'], me['maxr'])
    #     yl, yu = (me['minz'], me['maxz'])
    #
    #     # tri_index_lookup = mesh.triangle_index_lookup
    #
    #     emission_samples = np.zeros((500, 500))
    #
    #     xrange = np.linspace(xl, xu, 500)
    #     yrange = np.linspace(yl, yu, 500)
    #
    #     for i, x in enumerate(xrange):
    #         for j, y in enumerate(yrange):
    #
    #             try:
    #                 # k, l = mesh.triangle_to_grid_map[int(tri_index_lookup(x, y))]
    #                 # emission_samples[i, j] = total_rad[k, l]
    #                 emission_samples[j, i] = total_rad(x, 0, y)
    #
    #             except ValueError:
    #                 continue
    #
    #     plt.figure()
    #     plt.imshow(emission_samples, extent=[xl, xu, yl, yu], origin='lower')
    #     plt.colorbar()
    #     plt.xlim(xl, xu)
    #     plt.ylim(yl, yu)
    #     plt.title("Radiated Power (W/m^3)")

    def create_plasma(self, parent=None, transform=None, name=None):
        """
        Make a CHERAB plasma object from this SOLPS simulation.

        :param Node parent: The plasma's parent node in the scenegraph, e.g. a World object.
        :param AffineMatrix3D transform: Affine matrix describing the location and orientation
        of the plasma in the world.
        :param str name: User friendly name for this plasma (default = "SOLPS Plasma").
        :rtype: Plasma
        """

        mesh = self.mesh
        name = name or "SOLPS Plasma"
        plasma = Plasma(parent=parent, transform=transform, name=name)
        radius = mesh.mesh_extent['maxr']
        height = mesh.mesh_extent['maxz'] - mesh.mesh_extent['minz']
        plasma.geometry = Cylinder(radius, height)
        plasma.geometry_transform = translate(0, 0, mesh.mesh_extent['minz'])

        try:
            plasma.b_field = self.b_field_cartesian_f3d
        except RuntimeError:
            print('Warning! No magnetic field data available for this simulation.')

        # Create electron species
        electron_velocity = lambda x, y, z: Vector3D(0, 0, 0)
        plasma.electron_distribution = Maxwellian(self.electron_density_f3d, self.electron_temperature_f3d, electron_velocity, electron_mass)
 
        if self.velocities_cartesian_f3d is None:
            print('Warning! No velocity field data available for this simulation.')

        if self.neutral_temperature_f3d is None:
            print('Warning! No neutral atom temperature data available for this simulation.')

        neutral_i = 0  # neutrals count
        for k, sp in enumerate(self.species_list):

            try:
                species_type = lookup_element(sp[0])
            except ValueError:
                species_type = lookup_isotope(sp[0])

            charge = sp[1]

            # Create the velocity vector lookup function
            if self.velocities_cartesian is not None:
                velocity = self.velocities_cartesian_f3d[k]
            else:
                velocity = lambda x, y, z: Vector3D(0, 0, 0)

            if charge or self.neutral_temperature is None:  # ions or neutral atoms (neutral temperature is not available)
                distribution = Maxwellian(self.species_density_f3d[k], self.ion_temperature_f3d, velocity,
                                          species_type.atomic_weight * atomic_mass)

            else:  # neutral atoms with neutral temperature
                distribution = Maxwellian(self.species_density_f3d[k], self._neutral_temperature_f3d[neutral_i], velocity,
                                          species_type.atomic_weight * atomic_mass)
                neutral_i += 1

            plasma.composition.add(Species(species_type, charge, distribution))

        return plasma


def _check_array(name, value, shape):
    if not isinstance(value, np.ndarray):
        raise ValueError('Attribute "%s" must be a numpy.ndarray' % name)
    if value.shape != shape:
        raise ValueError('Shape of "%s": %s mismatch the shape of SOLPS grid: %s.' % (name, value.shape, shape))


def prefer_element(isotope):
    """
    Return Element instance, if the element of this isotope has the same mass number.
    """
    el_mass_number = int(round(isotope.element.atomic_weight))
    if el_mass_number == isotope.mass_number:
        return isotope.element

    return isotope


def b2_flux_to_velocity(mesh, density, poloidal_flux, radial_flux, parallel_velocity, b_field_cartesian):
    """
    Calculates velocities of neutral particles using B2 particle fluxes defined at cell faces.

    :param SOLPSMesh mesh: SOLPS simulation mesh.
    :param ndarray density: Density of atoms in m-3. Must be 3 dimensiona array of
                            shape (num_atoms, mesh.ny, mesh.nx).
    :param ndarray poloidal_flux: Poloidal flux of atoms in s-1. Must be a 3 dimensional array of
                                  shape (num_atoms, mesh.ny, mesh.nx).
    :param ndarray radial_flux: Radial flux of atoms in s-1. Must be a 3 dimensional array of
                                shape (num_atoms, mesh.ny, mesh.nx).
    :param ndarray parallel_velocity: Parallel velocity of atoms in m/s. Must be a 3 dimensional
                                      array of shape (num_atoms, mesh.ny, mesh.nx).
                                      Parallel velocity is a velocity projection on magnetic
                                      field direction.
    :param ndarray b_field_cartesian: Magnetic field in Cartesian (R, Z, phi) coordinates.
                                      Must be a 3 dimensional array of shape (3, mesh.ny, mesh.nx).

    :return: Velocities of atoms in (R, Z, phi) coordinates as a 4-dimensional ndarray of
             shape (num_atoms, 3, mesh.ny, mesh.nx)
    """

    nx = mesh.nx  # poloidal
    ny = mesh.ny  # radial
    ns = density.shape[0]  # number of species

    _check_array('density', density, (ns, ny, nx))
    _check_array('poloidal_flux', poloidal_flux, (ns, ny, nx))
    _check_array('radial_flux', radial_flux, (ns, ny, nx))
    _check_array('parallel_velocity', parallel_velocity, (ns, ny, nx))
    _check_array('b_field_cartesian', b_field_cartesian, (3, ny, nx))

    poloidal_area = mesh.poloidal_area[None]
    radial_area = mesh.radial_area[None]
    leftix = mesh.neighbix[0]  # poloidal prev.
    leftiy = mesh.neighbiy[0]
    bottomix = mesh.neighbix[1]  # radial prev.
    bottomiy = mesh.neighbiy[1]
    rightix = mesh.neighbix[2]   # poloidal next.
    rightiy = mesh.neighbiy[2]
    topix = mesh.neighbix[3]  # radial next.
    topiy = mesh.neighbiy[3]

    # Converting s-1 --> m-2 s-1
    poloidal_flux = np.divide(poloidal_flux, poloidal_area, out=np.zeros_like(poloidal_flux), where=poloidal_area > 0)
    radial_flux = np.divide(radial_flux, radial_area, out=np.zeros_like(radial_flux), where=radial_area > 0)

    # Obtaining left velocity
    dens_neighb = density[:, leftiy, leftix]  # density in the left neighbouring cell
    has_neighbour = ((leftix > -1) * (leftiy > -1))[None]  # check if has left neighbour
    neg_flux = (poloidal_flux < 0) * (density > 0)  # will use density in this cell if flux is negative
    pos_flux = (poloidal_flux > 0) * (dens_neighb > 0) * has_neighbour  # will use density in neighbouring cell if flux is positive
    velocity_left = np.divide(poloidal_flux, density, out=np.zeros((ns, ny, nx)), where=neg_flux)
    velocity_left = np.divide(poloidal_flux, dens_neighb, out=velocity_left, where=pos_flux)
    velocity_left = velocity_left[:, None] * mesh.poloidal_basis_vector[None]  # to vector in Cartesian

    # Obtaining bottom velocity
    dens_neighb = density[:, bottomiy, bottomix]
    has_neighbour = ((bottomix > -1) * (bottomiy > -1))[None]
    neg_flux = (radial_flux < 0) * (density > 0)
    pos_flux = (poloidal_flux > 0) * (dens_neighb > 0) * has_neighbour
    velocity_bottom = np.divide(radial_flux, density, out=np.zeros((ns, ny, nx)), where=neg_flux)
    velocity_bottom = np.divide(radial_flux, dens_neighb, out=velocity_bottom, where=pos_flux)
    velocity_bottom = velocity_bottom[:, None] * mesh.radial_basis_vector[None]  # to Cartesian

    # Obtaining right and top velocities
    velocity_right = velocity_left[:, :, rightiy, rightix]
    velocity_right[:, :, (rightix < 0) + (rightiy < 0)] = 0

    velocity_top = velocity_bottom[:, :, topiy, topix]
    velocity_top[:, :, (topix < 0) + (topiy < 0)] = 0

    vcart = np.zeros((ns, 3, ny, nx))  # velocities in Cartesian coordinates

    # Projection of velocity on RZ-plane
    vcart[:, :2] = 0.25 * (velocity_bottom + velocity_left + velocity_top + velocity_right)

    # Obtaining toroidal velocity
    b = b_field_cartesian[None]
    bmagn = np.sqrt((b * b).sum(1))
    vcart[:, 2] = (parallel_velocity * bmagn - vcart[:, 0] * b[:, 0] - vcart[:, 1] * b[:, 1]) / b[:, 2]

    return vcart


def eirene_flux_to_velocity(mesh, density, poloidal_flux, radial_flux, parallel_velocity, b_field_cartesian):
    """
    Calculates velocities of neutral particles using Eirene particle fluxes defined at cell centre.

    :param SOLPSMesh mesh: SOLPS simulation mesh.
    :param ndarray density: Density of atoms in m-3. Must be 3 dimensiona array of
                            shape (num_atoms, mesh.ny, mesh.nx).
    :param ndarray poloidal_flux: Poloidal flux of atoms in m-2 s-1. Must be a 3 dimensional array of
                                  shape (num_atoms, mesh.ny, mesh.nx).
    :param ndarray radial_flux: Radial flux of atoms in m-2 s-1. Must be a 3 dimensional array of
                                shape (num_atoms, mesh.ny, mesh.nx).
    :param ndarray parallel_velocity: Parallel velocity of atoms in m/s. Must be a 3 dimensional
                                      array of shape (num_atoms, mesh.ny, mesh.nx).
                                      Parallel velocity is a velocity projection on magnetic
                                      field direction.
    :param ndarray b_field_cartesian: Magnetic field in Cartesian (R, Z, phi) coordinates.
                                      Must be a 3 dimensional array of shape (3, mesh.ny, mesh.nx).

    :return: Velocities of atoms in (R, Z, phi) coordinates as a 4-dimensional ndarray of
             shape (mesh.nx, mesh.ny, num_atoms, 3)
    """

    nx = mesh.nx  # poloidal
    ny = mesh.ny  # radial
    ns = density.shape[0]  # number of neutral atoms

    _check_array('density', density, (ns, ny, nx))
    _check_array('poloidal_flux', poloidal_flux, (ns, ny, nx))
    _check_array('radial_flux', radial_flux, (ns, ny, nx))
    _check_array('parallel_velocity', parallel_velocity, (ns, ny, nx))
    _check_array('b_field_cartesian', b_field_cartesian, (3, ny, nx))

    # Obtaining velocity
    poloidal_velocity = np.divide(poloidal_flux, density, out=np.zeros_like(density), where=(density > 0))
    radial_velocity = np.divide(radial_flux, density, out=np.zeros_like(density), where=(density > 0))

    vcart = np.zeros((ns, 3, ny, nx))  # velocities in Cartesian coordinates

    # Projection of velocity on RZ-plane
    vcart[:, :2] = (poloidal_velocity[:, None] * mesh.poloidal_basis_vector[None] + radial_velocity[:, None] * mesh.radial_basis_vector[None])

    # Obtaining toroidal velocity
    b = b_field_cartesian[None]
    bmagn = np.sqrt((b * b).sum(1))
    vcart[:, 2] = (parallel_velocity * bmagn - vcart[:, 0] * b[:, 0] - vcart[:, 1] * b[:, 1]) / b[:, 2]

    return vcart
