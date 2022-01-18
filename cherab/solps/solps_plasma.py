
# Copyright 2016-2021 Euratom
# Copyright 2016-2021 United Kingdom Atomic Energy Authority
# Copyright 2016-2021 Centro de Investigaciones Energéticas, Medioambientales y Tecnológicas
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
from scipy.constants import atomic_mass, electron_mass

# Raysect imports
from raysect.core import translate, Vector3D
from raysect.primitive import Cylinder

# CHERAB core imports
from cherab.core import Plasma, Species, Maxwellian
from cherab.core.math.function import ConstantVector3D
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

        if not len(species_list):
            raise ValueError('Argument "species_list" must contain at least one species.')
        self._species_list = tuple(species_list)  # adding additional species is not allowed

        self._initial_setup()

    def _initial_setup(self):

        # Make Mesh Interpolator function for inside/outside mesh test.
        inside_outside_data = np.ones((self._mesh.ny, self._mesh.nx))
        self._inside_mesh = SOLPSFunction2D(self._mesh.vertex_coordinates, self._mesh.triangles,
                                            self._mesh.triangle_to_grid_map, inside_outside_data)

        # Creating a sample SOLPSVectorFunction2D for KDtree to use later
        sample_vector = np.ones((3, self._mesh.ny, self._mesh.nx))
        self._sample_vector_f2d = SOLPSVectorFunction2D(self._mesh.vertex_coordinates, self._mesh.triangles,
                                                        self._mesh.triangle_to_grid_map, sample_vector)

        self._neutral_list = tuple([sp for sp in self._species_list if sp[1] == 0])

        self._electron_temperature = None
        self._electron_temperature_f2d = None
        self._electron_temperature_f3d = None
        self._electron_density = None
        self._electron_density_f2d = None
        self._electron_density_f3d = None
        self._electron_velocities = None
        self._electron_velocities_cylindrical = None
        self._electron_velocities_cylindrical_f2d = None
        self._electron_velocities_cartesian = None
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
        self._velocities_cylindrical = None
        self._velocities_cylindrical_f2d = None
        self._velocities_cartesian = None
        self._halpha_mol_radiation = None
        self._halpha_mol_radiation_f2d = None
        self._halpha_mol_radiation_f3d = None
        self._halpha_total_radiation = None
        self._halpha_total_radiation_f2d = None
        self._halpha_total_radiation_f3d = None
        self._total_radiation = None
        self._total_radiation_f2d = None
        self._total_radiation_f3d = None
        self._b_field = None
        self._b_field_cylindrical = None
        self._b_field_cylindrical_f2d = None
        self._b_field_cartesian = None
        self._eirene_model = None
        self._b2_model = None
        self._eirene = None

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
    def neutral_list(self):
        """
        Tuple of species elements in the form (species name, charge).
        :return:
        """
        return self._neutral_list

    @property
    def electron_temperature(self):
        """
        Simulated electron temperatures at each mesh cell.
        Array of shape (ny, nx).
        :return:
        """
        return self._electron_temperature

    @property
    def electron_temperature_f2d(self):
        """
        Function2D interpolator for electron temperature.
        Returns electron temperature at a given point (R, Z).
        :return:
        """
        return self._electron_temperature_f2d

    @property
    def electron_temperature_f3d(self):
        """
        Function3D interpolator for electron temperature.
        Returns electron temperature at a given point (x, y, z).
        :return:
        """
        return self._electron_temperature_f3d

    @electron_temperature.setter
    def electron_temperature(self, value):
        value = np.array(value, dtype=np.float64, copy=False)
        _check_shape("electron_temperature", value, (self.mesh.ny, self.mesh.nx))
        self._electron_temperature = value
        self._electron_temperature_f2d = SOLPSFunction2D.instance(self._inside_mesh, value)
        self._electron_temperature_f3d = AxisymmetricMapper(self._electron_temperature_f2d)

    @property
    def ion_temperature(self):
        """
        Simulated ion temperatures at each mesh cell.
        Array of shape (ny, nx).
        :return:
        """
        return self._ion_temperature

    @property
    def ion_temperature_f2d(self):
        """
        Function2D interpolator for ion temperature.
        Returns ion temperature at a given point (R, Z).
        :return:
        """
        return self._ion_temperature_f2d

    @property
    def ion_temperature_f3d(self):
        """
        Function3D interpolator for ion temperature.
        Returns ion temperature at a given point (x, y, z).
        :return:
        """
        return self._ion_temperature_f3d

    @ion_temperature.setter
    def ion_temperature(self, value):
        value = np.array(value, dtype=np.float64, copy=False)
        _check_shape("ion_temperature", value, (self.mesh.ny, self.mesh.nx))
        self._ion_temperature = value
        self._ion_temperature_f2d = SOLPSFunction2D.instance(self._inside_mesh, value)
        self._ion_temperature_f3d = AxisymmetricMapper(self._ion_temperature_f2d)

    @property
    def neutral_temperature(self):
        """
        Simulated neutral atom (effective) temperatures at each mesh cell.
        Array of shape (na, ny, nx).
        :return:
        """
        return self._neutral_temperature

    @property
    def neutral_temperature_f2d(self):
        """
        Dictionary of Function2D interpolators for neutral atom (effective) temperatures.
        Accessed by neutral atom index or neutral_list elements.
        E.g., neutral_temperature_f2d[0] or neutral_temperature_f2d[('deuterium', 0))].
        Each entry returns neutral atom (effective) temperature at a given point (R, Z).
        :return:
        """
        return self._neutral_temperature_f2d

    @property
    def neutral_temperature_f3d(self):
        """
        Dictionary of Function3D interpolators for neutral atom (effective) temperatures.
        Accessed by neutral atom index or neutral_list elements.
        E.g., neutral_temperature_f3d[0] or neutral_temperature_f3d[('deuterium', 0))].
        Each entry returns neutral atom (effective) temperature at a given point (x, y, z).
        :return:
        """
        return self._neutral_temperature_f3d

    @neutral_temperature.setter
    def neutral_temperature(self, value):
        value = np.array(value, dtype=np.float64, copy=False)
        _check_shape("neutral_temperature", value, (len(self._neutral_list), self.mesh.ny, self.mesh.nx))
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
        Array of shape (ny, nx)
        :return:
        """
        return self._electron_density

    @property
    def electron_density_f2d(self):
        """
        Function2D interpolator for electron density.
        Returns electron density at a given point (R, Z).
        :return:
        """
        return self._electron_density_f2d

    @property
    def electron_density_f3d(self):
        """
        Function3D interpolator for electron density.
        Returns electron density at a given point (x, y, z).
        :return:
        """
        return self._electron_density_f3d

    @electron_density.setter
    def electron_density(self, value):
        value = np.array(value, dtype=np.float64, copy=False)
        _check_shape("electron_density", value, (self.mesh.ny, self.mesh.nx))
        self._electron_density = value
        self._electron_density_f2d = SOLPSFunction2D.instance(self._inside_mesh, value)
        self._electron_density_f3d = AxisymmetricMapper(self._electron_density_f2d)

    @property
    def electron_velocities(self):
        """
        Electron velocities in poloidal coordinates (e_pol, e_rad, e_tor) at each mesh cell.
        Array of shape (3, ny, nx):
        [0, :, :] - poloidal, [1, :, :] - radial, [2, :, :] - toroidal.
        :return:
        """
        return self._electron_velocities

    @electron_velocities.setter
    def electron_velocities(self, value):
        value = np.array(value, dtype=np.float64, copy=False)
        _check_shape("electron_velocities", value, (3, self.mesh.ny, self.mesh.nx))

        # Converting to cylindrical coordinates
        velocities_cylindrical = np.zeros(value.shape)
        velocities_cylindrical[1] = -value[2]
        velocities_cylindrical[[0, 2]] = self.mesh.to_cartesian(value[:2])

        self._electron_velocities = value
        self._electron_velocities_cylindrical = velocities_cylindrical
        self._electron_velocities_cylindrical_f2d = SOLPSVectorFunction2D.instance(self._sample_vector_f2d, velocities_cylindrical)
        self._electron_velocities_cartesian = VectorAxisymmetricMapper(self._electron_velocities_cylindrical_f2d)

    @property
    def electron_velocities_cylindrical(self):
        """
        Electron velocities in cylindrical coordinates (R, phi, Z) at each mesh cell.
        Array of shape (3, ny, nx): [0, :, :] - R, [1, :, :] - phi, [2, :, :] - Z.
        :return:
        """
        return self._electron_velocities_cylindrical

    @electron_velocities_cylindrical.setter
    def electron_velocities_cylindrical(self, value):
        value = np.array(value, dtype=np.float64, copy=False)
        _check_shape("electron_velocities_cylindrical", value, (3, self.mesh.ny, self.mesh.nx))

        # Converting to poloidal coordinates
        velocities = np.zeros(value.shape)
        velocities[2] = -value[1]
        velocities[:2] = self.mesh.to_poloidal(value[[0, 2]])

        self._electron_velocities_cylindrical = value
        self._electron_velocities = velocities
        self._electron_velocities_cylindrical_f2d = SOLPSVectorFunction2D.instance(self._sample_vector_f2d, value)
        self._electron_velocities_cartesian = VectorAxisymmetricMapper(self._electron_velocities_cylindrical_f2d)

    @property
    def electron_velocities_cylindrical_f2d(self):
        """
        VectorFunction2D interpolator for electron velocities in cylindrical coordinates.
        Returns a vector of electron velocity at a given point (R, Z).
        :return:
        """
        return self._electron_velocities_cylindrical_f2d

    @property
    def electron_velocities_cartesian(self):
        """
        VectorFunction3D interpolator for electron velocities in Cartesian coordinates.
        Returns a vector of electron velocity at a given point (x, y, z).
        :return:
        """
        return self._electron_velocities_cartesian

    @property
    def species_density(self):
        """
        Simulated species densities at each mesh cell.
        Array of shape (ns, ny, nx).
        :return:
        """
        return self._species_density

    @property
    def species_density_f2d(self):
        """
        Dictionary of Function2D interpolators for species densities.
        Accessed by species index or species_list elements.
        E.g., species_density_f2d[1] or species_density_f2d[('deuterium', 1))].
        Each entry returns species density at a given point (R, Z).
        :return:
        """
        return self._species_density_f2d

    @property
    def species_density_f3d(self):
        """
        Dictionary of Function3D interpolators for species densities.
        Accessed by species index or species_list elements.
        E.g., species_density_f3d[1] or species_density_f3d[('deuterium', 1))].
        Each entry returns species density at a given point (x, y, z).
        :return:
        """
        return self._species_density_f3d

    @species_density.setter
    def species_density(self, value):
        value = np.array(value, dtype=np.float64, copy=False)
        _check_shape("species_density", value, (len(self._species_list), self.mesh.ny, self.mesh.nx))
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
        Species velocities in poloidal coordinates (e_pol, e_rad, e_tor) at each mesh cell.
        Array of shape (ns, 3, ny, nx):
        [:, 0, :, :] - poloidal, [:, 1, :, :] - radial, [:, 2, :, :] - toroidal.
        :return:
        """
        return self._velocities

    @velocities.setter
    def velocities(self, value):
        value = np.array(value, dtype=np.float64, copy=False)
        _check_shape("velocities", value, (len(self.species_list), 3, self.mesh.ny, self.mesh.nx))

        # Converting to cylindrical coordinates
        velocities_cylindrical = np.zeros(value.shape)
        velocities_cylindrical[:, 1] = -value[:, 2]
        for k in range(value.shape[0]):
            velocities_cylindrical[k, [0, 2]] = self.mesh.to_cartesian(value[k, :2])

        self._velocities = value
        self._velocities_cylindrical = velocities_cylindrical
        self._velocities_cylindrical_f2d = {}
        self._velocities_cartesian = {}
        for k, sp in enumerate(self._species_list):
            self._velocities_cylindrical_f2d[k] = SOLPSVectorFunction2D.instance(self._sample_vector_f2d, velocities_cylindrical[k])
            self._velocities_cylindrical_f2d[sp] = self._velocities_cylindrical_f2d[k]
            self._velocities_cartesian[k] = VectorAxisymmetricMapper(self._velocities_cylindrical_f2d[k])
            self._velocities_cartesian[sp] = self._velocities_cartesian[k]

    @property
    def velocities_cylindrical(self):
        """
        Species velocities in cylindrical coordinates (R, phi, Z) at each mesh cell.
        Array of shape (ns, 3, ny, nx): [:, 0, :, :] - R, [:, 1, :, :] - phi, [:, 2, :, :] - Z.
        :return:
        """
        return self._velocities_cylindrical

    @velocities_cylindrical.setter
    def velocities_cylindrical(self, value):
        value = np.array(value, dtype=np.float64, copy=False)
        _check_shape("velocities_cylindrical", value, (len(self.species_list), 3, self.mesh.ny, self.mesh.nx))

        # Converting to poloidal coordinates
        velocities = np.zeros(value.shape)
        velocities[:, 2] = -value[:, 1]
        for k in range(value.shape[0]):
            velocities[k, :2] = self.mesh.to_poloidal(value[k, [0, 2]])

        self._velocities_cylindrical = value
        self._velocities = velocities
        self._velocities_cylindrical_f2d = {}
        self._velocities_cartesian = {}
        for k, sp in enumerate(self._species_list):
            self._velocities_cylindrical_f2d[k] = SOLPSVectorFunction2D.instance(self._sample_vector_f2d, value[k])
            self._velocities_cylindrical_f2d[sp] = self._velocities_cylindrical_f2d[k]
            self._velocities_cartesian[k] = VectorAxisymmetricMapper(self._velocities_cylindrical_f2d[k])
            self._velocities_cartesian[sp] = self._velocities_cartesian[k]

    @property
    def velocities_cylindrical_f2d(self):
        """
        Dictionary of VectorFunction2D interpolators for species velocities
        in cylindrical coordinates.
        Accessed by species index or species_list elements.
        E.g., velocities_cylindrical_f2d[1] or velocities_cylindrical_f2d[('deuterium', 1))].
        Each entry returns a vector of species velocity at a given point (R, Z).
        :return:
        """
        return self._velocities_cylindrical_f2d

    @property
    def velocities_cartesian(self):
        """
        Dictionary of VectorFunction3D interpolators for species velocities
        in Cartesian coordinates.
        Accessed by species index or species_list elements.
        E.g., velocities_cartesian[1] or velocities_cartesian[('deuterium', 1))].
        Each entry returns a vector of species velocity at a given point (x, y, z).
        :return:
        """
        return self._velocities_cartesian

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
        Total radiation at each mesh cell.
        Array of shape (ny, nx).

        This is not calculated from the CHERAB emission models, instead it comes from the SOLPS output data.
        Is calculated from the sum of all integrated line emission and all Bremmstrahlung.
        The signals used are 'RQRAD' and 'RQBRM'. Final output is in W m-3.
        """

        return self._total_radiation

    @property
    def total_radiation_f2d(self):
        """
        Function2D interpolator for total radiation.
        Returns total radiation at a given point (R, Z).
        """

        return self._total_radiation_f2d

    @property
    def total_radiation_f3d(self):
        """
        Function3D interpolator for total radiation.
        Returns total radiation at a given point (x, y, z).
        """

        return self._total_radiation_f3d

    @total_radiation.setter
    def total_radiation(self, value):
        value = np.array(value, dtype=np.float64, copy=False)
        _check_shape("total_radiation", value, (self.mesh.ny, self.mesh.nx))
        self._total_radiation = value
        self._total_radiation_f2d = SOLPSFunction2D.instance(self._inside_mesh, value)
        self._total_radiation_f3d = AxisymmetricMapper(self._total_radiation_f2d)

    @property
    def halpha_mol_radiation(self):
        """
        Molecular H-alpha radiation at each mesh cell.
        Array of shape (ny, nx).

        This is calculated with a built-in EIRENE collissional radiative model (CRM).
        Final output is in W m-3.
        """

        return self._halpha_mol_radiation

    @property
    def halpha_mol_radiation_f2d(self):
        """
        Function2D interpolator for molecular H-alpha radiation.
        Returns total radiation at a given point (R, Z).
        """

        return self._halpha_mol_radiation_f2d

    @property
    def halpha_mol_radiation_f3d(self):
        """
        Function3D interpolator for molecular H-alpha radiation.
        Returns total radiation at a given point (x, y, z).
        """

        return self._halpha_mol_radiation_f3d

    @halpha_mol_radiation.setter
    def halpha_mol_radiation(self, value):
        value = np.array(value, dtype=np.float64, copy=False)
        _check_shape("halpha_mol_radiation", value, (self.mesh.ny, self.mesh.nx))
        self._halpha_mol_radiation = value
        self._halpha_mol_radiation_f2d = SOLPSFunction2D.instance(self._inside_mesh, value)
        self._halpha_mol_radiation_f3d = AxisymmetricMapper(self._halpha_mol_radiation_f2d)

    @property
    def halpha_total_radiation(self):
        """
        Total (molecular + atomic) H-alpha radiation at each mesh cell.
        Array of shape (ny, nx).

        This is calculated with a built-in EIRENE collissional radiative model (CRM).
        Final output is in W m-3.
        """

        return self._halpha_total_radiation

    @property
    def halpha_total_radiation_f2d(self):
        """
        Function2D interpolator for total H-alpha radiation.
        Returns total radiation at a given point (R, Z).
        """

        return self._halpha_total_radiation_f2d

    @property
    def halpha_total_radiation_f3d(self):
        """
        Function3D interpolator for total H-alpha radiation.
        Returns total radiation at a given point (x, y, z).
        """

        return self._halpha_total_radiation_f3d

    @halpha_total_radiation.setter
    def halpha_total_radiation(self, value):
        value = np.array(value, dtype=np.float64, copy=False)
        _check_shape("halpha_total_radiation", value, (self.mesh.ny, self.mesh.nx))
        self._halpha_total_radiation = value
        self._halpha_total_radiation_f2d = SOLPSFunction2D.instance(self._inside_mesh, value)
        self._halpha_total_radiation_f3d = AxisymmetricMapper(self._halpha_total_radiation_f2d)

    @property
    def b_field(self):
        """
        Magnetic B field in poloidal coordinates (e_pol, e_rad, e_tor) at each mesh cell.
        Array of shape (3, ny, nx): [0, :, :] - poloidal, [1, :, :] - radial, [2, :, :] - toroidal.
        """

        return self._b_field

    @b_field.setter
    def b_field(self, value):
        value = np.array(value, dtype=np.float64, copy=False)
        _check_shape("b_field", value, (3, self.mesh.ny, self.mesh.nx))

        # Converting to cylindrical system
        b_field_cylindrical = np.zeros(value.shape)
        b_field_cylindrical[1] = -value[2]
        b_field_cylindrical[[0, 2]] = self.mesh.to_cartesian(value[:2])

        self._b_field_cylindrical = b_field_cylindrical
        self._b_field = value
        self._b_field_cylindrical_f2d = SOLPSVectorFunction2D.instance(self._sample_vector_f2d, b_field_cylindrical)
        self._b_field_cartesian = VectorAxisymmetricMapper(self._b_field_cylindrical_f2d)

    @property
    def b_field_cylindrical(self):
        """
        Magnetic B field in poloidal coordinates (R, phi, Z) at each mesh cell.
        Array of shape (3, ny, nx): [0, :, :] - R, [1, :, :] - phi, [2, :, :] - Z.
        """

        return self._b_field_cylindrical

    @property
    def b_field_cylindrical_f2d(self):
        """
        VectorFunction2D interpolator for magnetic B field in cylindrical coordinates.
        Returns a vector of magnetic field at a given point (R, Z).
        """

        return self._b_field_cylindrical_f2d

    @property
    def b_field_cartesian(self):
        """
        VectorFunction3D interpolator for magnetic B field in Cartesian coordinates.
        Returns a vector of magnetic field at a given point (x, y, z).
        """

        return self._b_field_cartesian

    @b_field_cylindrical.setter
    def b_field_cylindrical(self, value):
        value = np.array(value, dtype=np.float64, copy=False)
        _check_shape("b_field_cylindrical", value, (3, self.mesh.ny, self.mesh.nx))

        # Converting to poloidal system
        b_field = np.zeros(value.shape)
        b_field[2] = -value[1]
        b_field[:2] = self.mesh.to_poloidal(value[[0, 2]])

        self._b_field = b_field
        self._b_field_cylindrical = value
        self._b_field_cylindrical_f2d = SOLPSVectorFunction2D.instance(self._sample_vector_f2d, value)
        self._b_field_cartesian = VectorAxisymmetricMapper(self._b_field_cylindrical_f2d)

    @property
    def eirene_simulation(self):
        """
        Data from an underlying EIRENE simulation.

        :rtype: Eirene
        """

        return self._eirene

    @eirene_simulation.setter
    def eirene_simulation(self, value):
        if not isinstance(value, Eirene):
            raise ValueError('Attribute "eirene_simulation" must be an Eirene instance.')

        self._eirene = value

    def __getstate__(self):
        state = {
            'mesh': self._mesh,
            'species_list': self._species_list,
            'electron_temperature': self._electron_temperature,
            'ion_temperature': self._ion_temperature,
            'neutral_temperature': self._neutral_temperature,
            'electron_density': self._electron_density,
            'species_density': self._species_density,
            'electron_velocities_cylindrical': self._electron_velocities_cylindrical,
            'velocities_cylindrical': self._velocities_cylindrical,
            'total_radiation': self._total_radiation,
            'halpha_mol_radiation': self._halpha_mol_radiation,
            'halpha_total_radiation': self._halpha_total_radiation,
            'b_field_cylindrical': self._b_field_cylindrical,
            'eirene_model': self._eirene_model,
            'b2_model': self._b2_model,
            'eirene': self._eirene
        }
        return state

    def __setstate__(self, state):
        self._mesh = state['mesh']
        self._species_list = state['species_list']
        self._initial_setup()
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
        if state['electron_velocities_cylindrical'] is not None:
            self.electron_velocities_cylindrical = state['electron_velocities_cylindrical']
        if state['velocities_cylindrical'] is not None:
            self.velocities_cylindrical = state['velocities_cylindrical']
        if state['total_radiation'] is not None:
            self.total_radiation = state['total_radiation']
        if state['halpha_mol_radiation'] is not None:
            self.halpha_mol_radiation = state['halpha_mol_radiation']
        if state['halpha_total_radiation'] is not None:
            self.halpha_total_radiation = state['halpha_total_radiation']
        if state['b_field_cylindrical'] is not None:
            self.b_field_cylindrical = state['b_field_cylindrical']
        self._eirene_model = state['eirene_model']
        self._b2_model = state['b2_model']
        self._eirene = state['eirene']

    def save(self, filename):
        """
        Saves SOLPSSimulation to file.
        """
        with open(filename, 'wb') as file_handle:
            pickle.dump(self, file_handle)

    @classmethod
    def load(cls, filename):
        """
        Loads SOLPSSimulation from file.
        """
        with open(filename, 'rb') as file_handle:
            sim = pickle.load(file_handle)

        return sim

    def create_plasma(self, parent=None, transform=None, name=None):
        """
        Make a CHERAB plasma object from this SOLPS simulation.

        :param Node parent: The plasma's parent node in the scenegraph, e.g. a World object.
        :param AffineMatrix3D transform: Affine matrix describing the location and orientation
        of the plasma in the world.
        :param str name: User friendly name for this plasma (default = "SOLPS Plasma").
        :rtype: Plasma
        """

        # Checking if the minimal required data is available to create a plasma object
        if self.electron_density_f3d is None:
            raise RuntimeError("Unable to create plasma object: electron density is not set.")
        if self.electron_temperature_f3d is None:
            raise RuntimeError("Unable to create plasma object: electron temperature is not set.")
        if self.species_density_f3d is None:
            raise RuntimeError("Unable to create plasma object: species density is not set.")
        if self.ion_temperature_f3d is None:
            raise RuntimeError("Unable to create plasma object: ion temperature is not set.")

        mesh = self.mesh
        name = name or "SOLPS Plasma"
        plasma = Plasma(parent=parent, transform=transform, name=name)
        radius = mesh.mesh_extent['maxr']
        height = mesh.mesh_extent['maxz'] - mesh.mesh_extent['minz']
        plasma.geometry = Cylinder(radius, height)
        plasma.geometry_transform = translate(0, 0, mesh.mesh_extent['minz'])

        if self.b_field_cartesian is None:
            print('Warning! No magnetic field data available for this simulation.')
        else:
            plasma.b_field = self.b_field_cartesian

        # Create electron species
        if self.electron_velocities_cartesian is None:
            print('Warning! No electron velocity data available for this simulation.')
            electron_velocity = ConstantVector3D(Vector3D(0, 0, 0))
        else:
            electron_velocity = self.electron_velocities_cartesian
        plasma.electron_distribution = Maxwellian(self.electron_density_f3d, self.electron_temperature_f3d, electron_velocity, electron_mass)

        if self.velocities_cartesian is None:
            print('Warning! No species velocities data available for this simulation.')

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
                velocity = self.velocities_cartesian[k]
            else:
                velocity = ConstantVector3D(Vector3D(0, 0, 0))

            if charge or self.neutral_temperature is None:  # ions or neutral atoms (neutral temperature is not available)
                distribution = Maxwellian(self.species_density_f3d[k], self.ion_temperature_f3d, velocity,
                                          species_type.atomic_weight * atomic_mass)

            else:  # neutral atoms with neutral temperature
                distribution = Maxwellian(self.species_density_f3d[k], self._neutral_temperature_f3d[neutral_i], velocity,
                                          species_type.atomic_weight * atomic_mass)
                neutral_i += 1

            plasma.composition.add(Species(species_type, charge, distribution))

        return plasma


def _check_shape(name, value, shape):
    if value.shape != shape:
        raise ValueError('Shape of "{0}": {1} mismatch the shape of SOLPS grid: {2}.'.format(name, value.shape, shape))


def prefer_element(isotope):
    """
    Return Element instance, if the element of this isotope has the same mass number.
    """
    el_mass_number = int(round(isotope.element.atomic_weight))
    if el_mass_number == isotope.mass_number:
        return isotope.element

    return isotope


def b2_flux_to_velocity(sim, poloidal_flux, radial_flux, parallel_velocity):
    """
    Calculates velocities of all species at cell centres using B2 particle fluxes defined
    at cell faces.

    :param SOLPSSimulation sim: SOLPSSimulation instance.
    :param ndarray poloidal_flux: Poloidal flux of species in s-1. Must be a 3 dimensional array of
                                  shape (num_species, mesh.ny, mesh.nx).
    :param ndarray radial_flux: Radial flux of species in s-1. Must be a 3 dimensional array of
                                shape (num_species, mesh.ny, mesh.nx).
    :param ndarray parallel_velocity: Parallel velocity of species in m/s. Must be a 3 dimensional
                                      array of shape (num_species, mesh.ny, mesh.nx).
                                      Parallel velocity is a velocity projection on magnetic
                                      field direction.

    :return ndarray: Velocities of all species in cylindrical coordinates.
                     Array of shape (num_species, 3, mesh.ny, mesh.nx).
    """
    if sim.b_field_cylindrical is None:
        raise RuntimeError('Attribute "b_field_cylindrical" is not set.')
    if sim.species_density is None:
        raise RuntimeError('Attribute "species_density" is not set.')

    mesh = sim.mesh
    b = sim.b_field_cylindrical

    poloidal_flux = np.array(poloidal_flux, dtype=np.float64, copy=False)
    radial_flux = np.array(radial_flux, dtype=np.float64, copy=False)
    parallel_velocity = np.array(parallel_velocity, dtype=np.float64, copy=False)

    nx = mesh.nx  # poloidal
    ny = mesh.ny  # radial
    ns = len(sim.species_list)  # number of species

    _check_shape('poloidal_flux', poloidal_flux, (ns, ny, nx))
    _check_shape('radial_flux', radial_flux, (ns, ny, nx))
    _check_shape('parallel_velocity', parallel_velocity, (ns, ny, nx))

    poloidal_area = mesh.poloidal_area
    radial_area = mesh.radial_area
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
    dens_neighb = sim.species_density[:, leftiy, leftix]  # density in the left neighbouring cell
    has_neighbour = ((leftix > -1) * (leftiy > -1))  # check if has left neighbour
    neg_flux = (poloidal_flux < 0) * (sim.species_density > 0)  # will use density in this cell if flux is negative
    pos_flux = (poloidal_flux > 0) * (dens_neighb > 0) * has_neighbour  # will use density in neighbouring cell if flux is positive
    velocity_left = np.divide(poloidal_flux, sim.species_density, out=np.zeros((ns, ny, nx)), where=neg_flux)
    velocity_left = np.divide(poloidal_flux, dens_neighb, out=velocity_left, where=pos_flux)
    poloidal_face_normal = mesh.radial_basis_vector[[1, 0]]
    poloidal_face_normal[1] *= -1
    velocity_left = velocity_left[:, None] * poloidal_face_normal  # to vector in RZ

    # Obtaining bottom velocity
    dens_neighb = sim.species_density[:, bottomiy, bottomix]
    has_neighbour = ((bottomix > -1) * (bottomiy > -1))
    neg_flux = (radial_flux < 0) * (sim.species_density > 0)
    pos_flux = (poloidal_flux > 0) * (dens_neighb > 0) * has_neighbour
    velocity_bottom = np.divide(radial_flux, sim.species_density, out=np.zeros((ns, ny, nx)), where=neg_flux)
    velocity_bottom = np.divide(radial_flux, dens_neighb, out=velocity_bottom, where=pos_flux)
    radial_face_normal = mesh.poloidal_basis_vector[[1, 0]]
    radial_face_normal[0] *= -1
    velocity_bottom = velocity_bottom[:, None] * radial_face_normal  # to RZ

    # Obtaining right and top velocities
    velocity_right = velocity_left[:, :, rightiy, rightix]
    velocity_right[:, :, (rightix < 0) + (rightiy < 0)] = 0

    velocity_top = velocity_bottom[:, :, topiy, topix]
    velocity_top[:, :, (topix < 0) + (topiy < 0)] = 0

    vcyl = np.zeros((ns, 3, ny, nx))  # velocities in cylindrical coordinates

    # Projection of velocity on RZ-plane
    vcyl[:, [0, 2]] = 0.25 * (velocity_bottom + velocity_left + velocity_top + velocity_right)

    # Obtaining toroidal velocity
    bmagn = np.sqrt((b * b).sum(0))
    vcyl[:, 1] = (parallel_velocity * bmagn - vcyl[:, 0] * b[0] - vcyl[:, 2] * b[2]) / b[1]

    return vcyl


def eirene_flux_to_velocity(sim, poloidal_flux, radial_flux, parallel_velocity):
    """
    Calculates velocities of neutral atoms using Eirene particle fluxes defined at cell centre.

    :param SOLPSSimulation sim: SOLPSSimulation instance.
    :param ndarray poloidal_flux: Poloidal flux of atoms in m-2 s-1. Must be a 3 dimensional array of
                                  shape (num_atoms, mesh.ny, mesh.nx).
    :param ndarray radial_flux: Radial flux of atoms in m-2 s-1. Must be a 3 dimensional array of
                                shape (num_atoms, mesh.ny, mesh.nx).
    :param ndarray parallel_velocity: Parallel velocity of atoms in m/s. Must be a 3 dimensional
                                      array of shape (num_atoms, mesh.ny, mesh.nx).
                                      Parallel velocity is a velocity projection on magnetic
                                      field direction.

    :return ndarray: Velocities of neutral atoms in cylindrical coordinates.
                     Array of shape (num_atoms, 3, mesh.ny, mesh.nx).
    """
    if sim.b_field_cylindrical is None:
        raise RuntimeError('Attribute "b_field_cylindrical" is not set.')
    if sim.species_density is None:
        raise RuntimeError('Attribute "species_density" is not set.')

    mesh = sim.mesh
    b = sim.b_field_cylindrical

    poloidal_flux = np.array(poloidal_flux, dtype=np.float64, copy=False)
    radial_flux = np.array(radial_flux, dtype=np.float64, copy=False)
    parallel_velocity = np.array(parallel_velocity, dtype=np.float64, copy=False)

    nx = mesh.nx  # poloidal
    ny = mesh.ny  # radial
    ns = len(sim.neutral_list)  # number of neutral atoms

    _check_shape('poloidal_flux', poloidal_flux, (ns, ny, nx))
    _check_shape('radial_flux', radial_flux, (ns, ny, nx))
    _check_shape('parallel_velocity', parallel_velocity, (ns, ny, nx))

    neutral_indx = [k for k, sp in enumerate(sim.species_list) if sp[1] == 0]
    density = sim.species_density[neutral_indx, :]

    # Obtaining velocity
    poloidal_velocity = np.divide(poloidal_flux, density, out=np.zeros_like(density), where=(density > 0))
    radial_velocity = np.divide(radial_flux, density, out=np.zeros_like(density), where=(density > 0))

    vcyl = np.zeros((ns, 3, ny, nx))  # velocities in cylindrical coordinates

    # Projection of velocity on RZ-plane
    vcyl[:, [0, 2]] = (poloidal_velocity[:, None] * mesh.poloidal_basis_vector + radial_velocity[:, None] * mesh.radial_basis_vector)

    # Obtaining toroidal velocity
    bmagn = np.sqrt((b * b).sum(0))
    vcyl[:, 1] = (parallel_velocity * bmagn - vcyl[:, 0] * b[0] - vcyl[:, 2] * b[2]) / b[1]

    return vcyl
