
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
from raysect.core.math.function.float import Discrete2DMesh
from raysect.core import translate, Point3D, Vector3D, Node, AffineMatrix3D
from raysect.primitive import Cylinder
from raysect.optical import Spectrum

# CHERAB core imports
from cherab.core import Plasma, Species, Maxwellian
from cherab.core.math.mappers import AxisymmetricMapper
from cherab.core.atomic.elements import lookup_isotope, lookup_element

# This SOLPS package imports
from cherab.solps.eirene import Eirene
from .solps_3d_functions import SOLPSFunction3D, SOLPSVectorFunction3D
from .mesh_geometry import SOLPSMesh


# TODO: Implement *_as_f2d() and *_as_f3d() interpolators for plasma parameters

class SOLPSSimulation:

    def __init__(self, mesh, species_list):

        # Mesh and species_list cannot be changed after initialisation
        if isinstance(mesh, SOLPSMesh):
            self._mesh = mesh
        else:
            raise ValueError('Argument "mesh" must be a SOLPSMesh instance.')

        # Make Mesh Interpolator function for inside/outside mesh test.
        inside_outside_data = np.ones(mesh.num_triangles)
        inside_outside = AxisymmetricMapper(Discrete2DMesh(mesh.vertex_coordinates, mesh.triangles, inside_outside_data, limit=False))
        self._inside_mesh = inside_outside

        if not len(species_list):
            raise ValueError('Argument "species_list" must contain at least one species.')
        self._species_list = tuple(species_list)  # adding additional species is not allowed

        self._electron_temperature = None
        self._electron_density = None
        self._ion_temperature = None
        self._neutral_temperature = None
        self._species_density = None
        self._velocities = None
        self._velocities_cartesian = None
        self._total_radiation = None
        self._b_field_vectors = None
        self._b_field_vectors_cartesian = None
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

    @electron_temperature.setter
    def electron_temperature(self, value):
        _check_array("electron_temperature", value, (self.mesh.ny, self.mesh.nx))

        self._electron_temperature = value

    @property
    def ion_temperature(self):
        """
        Simulated ion temperatures at each mesh cell.
        :return:
        """
        return self._ion_temperature

    @ion_temperature.setter
    def ion_temperature(self, value):
        _check_array("ion_temperature", value, (self.mesh.ny, self.mesh.nx))

        self._ion_temperature = value

    @property
    def neutral_temperature(self):
        """
        Array of neutral atom (effective) temperature at each mesh cell.
        :return:
        """
        return self._neutral_temperature

    @neutral_temperature.setter
    def neutral_temperature(self, value):
        num_neutrals = len([sp for sp in self.species_list if sp[1] == 0])
        _check_array("neutral_temperature", value, (num_neutrals, self.mesh.ny, self.mesh.nx))

        self._neutral_temperature = value

    @property
    def electron_density(self):
        """
        Simulated electron densities at each mesh cell.
        :return:
        """
        return self._electron_density

    @electron_density.setter
    def electron_density(self, value):
        _check_array("electron_density", value, (self.mesh.ny, self.mesh.nx))

        self._electron_density = value

    @property
    def species_density(self):
        """
        Array of species densities at each mesh cell.
        :return:
        """
        return self._species_density

    @species_density.setter
    def species_density(self, value):
        _check_array("species_density", value, (len(self.species_list), self.mesh.ny, self.mesh.nx))

        self._species_density = value

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

        self._velocities_cartesian = velocities_cartesian
        self._velocities = value

    @property
    def velocities_cartesian(self):
        """
        Velocities in Cartesian (v_r, v_z, v_toroidal) coordinates for each species densities at each mesh cell.
        :return:
        """
        return self._velocities_cartesian

    @velocities_cartesian.setter
    def velocities_cartesian(self, value):
        _check_array("velocities_cartesian", value, (len(self.species_list), 3, self.mesh.ny, self.mesh.nx))

        # Converting to poloidal coordinates
        velocities = np.zeros(value.shape)
        velocities[:, 2] = value[:, 2]
        for k in range(value.shape[0]):
            velocities[k, :2] = self.mesh.to_poloidal(value[k, :2])

        self._velocities = value
        self._velocities_cartesian = value

    @property
    def inside_volume_mesh(self):
        """
        Function3D for testing if point p is inside the simulation mesh.
        """
        if self._inside_mesh is None:
            raise RuntimeError("Inside mesh test not available for this simulation.")
        else:
            return self._inside_mesh

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

    @total_radiation.setter
    def total_radiation(self, value):
        _check_array("total_radiation", value, (self.mesh.ny, self.mesh.nx))

        self._total_radiation = value

    # TODO: decide is this a 2D or 3D interface?
    @property
    def total_radiation_volume(self):
        """
        Total radiation volume.

        This is not calculated from the CHERAB emission models, instead it comes from the SOLPS output data.
        Is calculated from the sum of all integrated line emission and all Bremmstrahlung. The signals used are 'RQRAD'
        and 'RQBRM'. Final output is in W/str?

        :returns: Function3D
        """

        mapped_radiation_data = _map_data_onto_triangles(self._total_rad)
        radiation_mesh_2d = Discrete2DMesh(self.mesh.vertex_coordinates, self.mesh.triangles, mapped_radiation_data, limit=False)
        # return AxisymmetricMapper(radiation_mesh_2d)
        return radiation_mesh_2d

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

    @property
    def b_field_cartesian(self):
        """
        Magnetic B field at each mesh cell in Cartesian coordinates (B_r, B_z, B_toroidal).
        """
        if self._b_field_cartesian is None:
            raise RuntimeError("Magnetic field not available for this simulation.")
        else:
            return self._b_field_cartesian

    @b_field_cartesian.setter
    def b_field_cartesian(self, value):
        _check_array("b_field_cartesian", value, (3, self.mesh.ny, self.mesh.nx))

        # Converting to poloidal system
        b_field = np.zeros(value.shape)
        b_field[2] = value[2]
        b_field[:2] = self.mesh.to_poloidal(value[:2])

        self._b_field = b_field
        self._b_field_cartesian = value

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
            'electron_temperature': self._electron_temperature,
            'ion_temperature': self._ion_temperature,
            'neutral_temperature': self._neutral_temperature,
            'electron_density': self._electron_density,
            'species_list': self._species_list,
            'species_density': self._species_density,
            'velocities': self._velocities,
            'velocities_cartesian': self._velocities_cartesian,
            'inside_mesh': self._inside_mesh,
            'total_radiation': self._total_radiation,
            'b_field': self._b_field,
            'b_field_cartesian': self._b_field_cartesian,
            'eirene_model': self._eirene_model,
            'b2_model': self._b2_model,
            'eirene': self._eirene
        }
        return state

    def __setstate__(self, state):
        self._mesh = SOLPSMesh(**state['mesh'])
        self._electron_temperature = state['electron_temperature']
        self._ion_temperature = state['ion_temperature']
        self._neutral_temperature = state['neutral_temperature']
        self._electron_density = state['electron_density']
        self._species_list = state['species_list']
        self._species_density = state['species_density']
        self._velocities = state['velocities']
        self._velocities_cartesian = state['velocities_cartesian']
        self._inside_mesh = state['inside_mesh']
        self._total_radiation = state['total_radiation']
        self._b_field = state['b_field']
        self._b_field_cartesian = state['b_field_cartesian']
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

        tri_index_lookup = self.mesh.triangle_index_lookup
        tri_to_grid = self.mesh.triangle_to_grid_map

        try:
            plasma.b_field = SOLPSVectorFunction3D(tri_index_lookup, tri_to_grid, self.b_field_cartesian)
        except RuntimeError:
            print('Warning! No magnetic field data available for this simulation.')

        # Create electron species
        triangle_data = _map_data_onto_triangles(self.electron_temperature)
        electron_te_interp = Discrete2DMesh(mesh.vertex_coordinates, mesh.triangles, triangle_data, limit=False)
        electron_temp = AxisymmetricMapper(electron_te_interp)
        triangle_data = _map_data_onto_triangles(self.electron_density)
        electron_dens = AxisymmetricMapper(Discrete2DMesh.instance(electron_te_interp, triangle_data))
        electron_velocity = lambda x, y, z: Vector3D(0, 0, 0)
        plasma.electron_distribution = Maxwellian(electron_dens, electron_temp, electron_velocity, electron_mass)

        # Ion temperature
        triangle_data = _map_data_onto_triangles(self.ion_temperature)
        ion_temp = AxisymmetricMapper(Discrete2DMesh.instance(electron_te_interp, triangle_data))

        if self.velocities_cartesian is None:
            print('Warning! No velocity field data available for this simulation.')

        if self.neutral_temperature is None:
            print('Warning! No neutral atom temperature data available for this simulation.')

        neutral_i = 0  # neutrals count
        for k, sp in enumerate(self.species_list):

            try:
                species_type = lookup_element(sp[0])
            except ValueError:
                species_type = lookup_isotope(sp[0])

            charge = sp[1]

            triangle_data = _map_data_onto_triangles(self.species_density[k])
            dens = AxisymmetricMapper(Discrete2DMesh.instance(electron_te_interp, triangle_data))

            # dens = SOLPSFunction3D(tri_index_lookup, tri_to_grid, self.species_density[k])

            # Create the velocity vector lookup function
            if self.velocities_cartesian is not None:
                velocity = SOLPSVectorFunction3D(tri_index_lookup, tri_to_grid, self.velocities_cartesian[k])
            else:
                velocity = lambda x, y, z: Vector3D(0, 0, 0)

            if charge or self.neutral_temperature is None:  # ions or neutral atoms (neutral temperature is not available)
                distribution = Maxwellian(dens, ion_temp, velocity, species_type.atomic_weight * atomic_mass)

            else:  # neutral atoms with neutral temperature
                triangle_data = _map_data_onto_triangles(self.neutral_temperature[neutral_i])
                neutral_temp = AxisymmetricMapper(Discrete2DMesh.instance(electron_te_interp, triangle_data))
                distribution = Maxwellian(dens, neutral_temp, velocity, species_type.atomic_weight * atomic_mass)
                neutral_i += 1

            plasma.composition.add(Species(species_type, charge, distribution))

        return plasma


def _map_data_onto_triangles(solps_dataset):
    """
    Reshape a SOLPS data array so that it matches the triangles in the SOLPS mesh.

    :param ndarray solps_dataset: Given SOLPS dataset, typically of shape (98 x 32).
    :return: New 1D ndarray with shape (98*32*2)
    """

    triangle_data = np.zeros(solps_dataset.size * 2, dtype=np.float64)

    triangle_data[::2] = solps_dataset.flatten()
    triangle_data[1::2] = triangle_data[::2]

    return triangle_data


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
