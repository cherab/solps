
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
from raysect.core.math.function import Discrete2DMesh

from cherab.core.math.mappers import AxisymmetricMapper
from cherab.core.atomic.elements import hydrogen, deuterium, helium, beryllium, carbon, nitrogen, oxygen, neon, \
    argon, krypton, xenon

from cherab.solps.eirene import Eirene
from cherab.solps.b2.parse_b2_block_file import load_b2f_file
from cherab.solps.mesh_geometry import SOLPSMesh
from cherab.solps.solps_plasma import SOLPSSimulation

Q = 1.602E-19

# key is nuclear charge Z and atomic mass AMU
_popular_species = {
    (1, 2): deuterium,
    (2, 4.003): helium,
    (2, 4.0): helium,
    (6, 12.011): carbon,
    (6, 12.0): carbon,
    (7, 14.007): nitrogen,
    (10, 20.0): neon,
    (10, 20.1797): neon,
    (18, 39.948): argon,
    (36, 83.798): krypton,
    (54, 131.293): xenon
}


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

    sim = SOLPSSimulation(mesh)
    ni = mesh.nx
    nj = mesh.ny

    # TODO: add code to load SOLPS velocities and magnetic field from files

    # Load electron species
    sim._electron_temperature = mesh_data_dict['te']/Q
    sim._electron_density = mesh_data_dict['ne']

    ##########################################
    # Load each plasma species in simulation #
    ##########################################

    sim._species_list = []
    for i in range(len(sim_info_dict['zn'])):

        zn = int(sim_info_dict['zn'][i])  # Nuclear charge
        am = float(sim_info_dict['am'][i])  # Atomic mass
        charge = int(sim_info_dict['zamax'][i])  # Ionisation/charge
        species = _popular_species[(zn, am)]
        sim.species_list.append(species.symbol + str(charge))

    sim._species_density = mesh_data_dict['na']

    # Make Mesh Interpolator function for inside/outside mesh test.
    inside_outside_data = np.ones(mesh.num_tris)
    inside_outside = AxisymmetricMapper(Discrete2DMesh(mesh.vertex_coords, mesh.triangles, inside_outside_data, limit=False))
    sim._inside_mesh = inside_outside

    # Load total radiated power from EIRENE output file
    eirene = Eirene.from_fort44(eirene_fort44_file, debug=debug)
    sim._eirene = eirene

    # Note EIRENE data grid is slightly smaller than SOLPS grid, for example (98, 38) => (96, 36)
    # Need to pad EIRENE data to fit inside larger B2 array
    nx = mesh.nx
    ny = mesh.ny
    eradt_raw_data = eirene.eradt.sum(2)
    eradt_data = np.zeros((nx, ny))
    eradt_data[1:nx-1, 1:ny-1] = eradt_raw_data
    sim._total_rad = eradt_data

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
