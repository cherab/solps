
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

    # Load SOLPS mesh geometry
    mesh = load_mesh_from_files(mesh_file_path=mesh_file_path, debug=debug)

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

    # TODO: add code to load SOLPS velocities and magnetic field from files

    # Load electron species
    sim.electron_temperature = mesh_data_dict['te'] / elementary_charge
    sim.electron_density = mesh_data_dict['ne']

    sim.species_density = mesh_data_dict['na']

    # Load total radiated power from EIRENE output file
    eirene = load_fort44_file(eirene_fort44_file, debug=debug)
    sim.eirene_simulation = eirene

    # Note EIRENE data grid is slightly smaller than SOLPS grid, for example (98, 38) => (96, 36)
    # Need to pad EIRENE data to fit inside larger B2 array
    nx = mesh.nx
    ny = mesh.ny

    # Replacing B2 neutral densities with EIRENE ones
    da_raw_data = eirene.da
    neutral_i = 0  # counter for neutral atoms
    for k, sp in enumerate(sim.species_list):
        charge = sp[1]
        if charge == 0:
            sim.species_density[1:-1, 1:-1, k] = da_raw_data[:, :, neutral_i]
            neutral_i += 1

    # Obtaining total radiation
    eradt_raw_data = eirene.eradt.sum(2)
    sim.total_radiation = np.zeros((nx, ny))
    sim.total_radiation[1:-1, 1:-1] = eradt_raw_data

    return sim


def load_mesh_from_files(mesh_file_path, debug=False):
    """
    Load SOLPS grid description from B2 Eirene output file.

    :param str filepath: full path for B2 eirene mesh description file
    :param bool debug: flag for displaying textual debugging information.
    :return: tuple of dictionaries. First is the header information such as the version, label, grid size, etc.
      Second dictionary has a ndarray for each piece of data found in the file.
    """
    _, _, geom_data_dict = load_b2f_file(mesh_file_path, debug=debug)

    cr_x = geom_data_dict['crx']
    cr_z = geom_data_dict['cry']
    vol = geom_data_dict['vol']

    # build mesh object
    return SOLPSMesh(cr_x, cr_z, vol)
