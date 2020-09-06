
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

import os
import numpy as np
from scipy.constants import elementary_charge
from raysect.core.math.function.float import Discrete2DMesh

from cherab.core.math.mappers import AxisymmetricMapper
from cherab.core.atomic.elements import lookup_isotope

from cherab.solps.eirene import load_fort44_file
from cherab.solps.b2.parse_b2_block_file import load_b2f_file
from cherab.solps.mesh_geometry import SOLPSMesh
from cherab.solps.solps_plasma import SOLPSSimulation, prefer_element


# Code based on script by Felix Reimold (2016)
def load_solps_from_raw_output(simulation_path, debug=False):
    """
    Load a SOLPS simulation from raw SOLPS output files.

    Required files include:
    * mesh description file (b2fgmtry)
    * B2 plasma state (b2fstate)
    * Eirene output file (fort.44), optional

    :param str simulation_path: String path to simulation directory.
    :rtype: SOLPSSimulation
    """

    if not os.path.isdir(simulation_path):
        RuntimeError("Simulation_path must be a valid directory.")

    mesh_file_path = os.path.join(simulation_path, 'b2fgmtry')
    b2_state_file = os.path.join(simulation_path, 'b2fstate')
    eirene_fort44_file = os.path.join(simulation_path, "fort.44")

    if not os.path.isfile(mesh_file_path):
        raise RuntimeError("No B2 b2fgmtry file found in SOLPS output directory.")

    if not(os.path.isfile(b2_state_file)):
        RuntimeError("No B2 b2fstate file found in SOLPS output directory.")

    if not(os.path.isfile(eirene_fort44_file)):
        print("Warning! No EIRENE fort.44 file found in SOLPS output directory. Assuming B2.5 stand-alone simulation.")
        b2_standalone = True
    else:
        # Load data for neutral species from EIRENE output file
        eirene = load_fort44_file(eirene_fort44_file, debug=debug)
        b2_standalone = False

    # Load SOLPS mesh geometry
    _, _, geom_data_dict = load_b2f_file(mesh_file_path, debug=debug)  # geom_data_dict is needed also for magnetic field

    mesh = SOLPSMesh(geom_data_dict['crx'], geom_data_dict['cry'], geom_data_dict['vol'])
    ni = mesh.nx
    nj = mesh.ny

    header_dict, sim_info_dict, mesh_data_dict = load_b2f_file(b2_state_file, debug=debug)

    # Load each plasma species in simulation
    species_list = []
    for i in range(len(sim_info_dict['zn'])):

        zn = int(sim_info_dict['zn'][i])  # Nuclear charge
        am = int(round(float(sim_info_dict['am'][i])))  # Atomic mass number
        charge = int(sim_info_dict['zamax'][i])  # Ionisation/charge
        isotope = lookup_isotope(zn, number=am)
        species = prefer_element(isotope)  # Prefer Element over Isotope if the mass number is the same
        species_list.append((species, charge))

    sim = SOLPSSimulation(mesh, species_list)

    # Load magnetic field    
    sim.b_field = geom_data_dict['bb'][:, :, :3]
    # sim.b_field_cartesian is created authomatically

    # Load electron species
    sim.electron_temperature = mesh_data_dict['te'] / elementary_charge
    sim.electron_density = mesh_data_dict['ne']

    # Load ion temperature
    sim.ion_temperature = mesh_data_dict['ti'] / elementary_charge

    # Load species density
    sim.species_density = mesh_data_dict['na']

    if not b2_standalone:
        # Replacing B2 neutral densities with EIRENE ones
        da_raw_data = eirene.da
        neutral_i = 0  # counter for neutral atoms
        for k, sp in enumerate(sim.species_list):
            charge = sp[1]
            if charge == 0:
                sim.species_density[1:-1, 1:-1, k] = da_raw_data[:, :, neutral_i]
                neutral_i += 1

    # TODO: Eirene data (TOP.SNAPSHOT.PFLA, TOP.SNAPSHOT.RFLA) should be used for neutral atoms.
    velocities = np.zeros((ni, nj, len(sim.species_list), 3))
    velocities[:, :, :, 0] = mesh_data_dict['ua']

    ################################################
    # Calculate the species' velocity distribution #

    # calculate field component ratios for velocity conversion
    bplane2 = sim.b_field[:, :, 0]**2 + sim.b_field[:, :, 2]**2
    parallel_to_toroidal_ratio = sim.b_field[:, :, 0] * sim.b_field[:, :, 2] / bplane2

    # Calculate toroidal velocity component
    velocities[:, :, :, 2] = velocities[:, :, :, 0] * parallel_to_toroidal_ratio[:, :, None]

    # Radial velocity is obtained from radial particle flux
    radial_particle_flux = mesh_data_dict['fna'][:, :, 1::2]

    vec_r = mesh.r[:, :, 1] - mesh.r[:, :, 0]
    vec_z = mesh.z[:, :, 1] - mesh.z[:, :, 0]
    radial_area = np.pi * (mesh.r[:, :, 1] + mesh.r[:, :, 0]) * np.sqrt(vec_r**2 + vec_z**2)

    for k, sp in enumerate(sim.species_list):
        i, j = np.where(sim.species_density[:, :, k] > 0)
        velocities[i, j, k, 1] = radial_particle_flux[i, j, k] / radial_area[i, j] / sim.species_density[i, j, k]

    sim.velocities = velocities
    # sim.velocities_cartesian is created authomatically

    if not b2_standalone:
        # Note EIRENE data grid is slightly smaller than SOLPS grid, for example (98, 38) => (96, 36)
        # Need to pad EIRENE data to fit inside larger B2 array

        # Obtaining neutral temperatures
        ta = np.zeros((ni, nj, eirene.ta.shape[2]))
        ta[1:-1, 1:-1, :] = eirene.ta
        for i in (0, -1):
            ta[i, 1:-1, :] = eirene.ta[i, :, :]
            ta[1:-1, i, :] = eirene.ta[:, i, :]
        for i, j in ((0, 0), (0, -1), (-1, 0), (-1, -1)):
            ta[i, j, :] = eirene.ta[i, j, :]
        sim.neutral_temperature = ta / elementary_charge

        # Obtaining total radiation
        eradt_raw_data = eirene.eradt.sum(2)
        sim.total_radiation = np.zeros((ni, nj))
        sim.total_radiation[1:-1, 1:-1] = eradt_raw_data

        sim.eirene_simulation = eirene

    return sim
