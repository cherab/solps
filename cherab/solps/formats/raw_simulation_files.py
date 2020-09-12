
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
        print("Warning! No EIRENE fort.44 file found in SOLPS output directory. Assuming B2 stand-alone simulation.")
        b2_standalone = True
    else:
        # Load data for neutral species from EIRENE output file
        eirene = load_fort44_file(eirene_fort44_file, debug=debug)
        b2_standalone = False

    # Load SOLPS mesh geometry
    _, _, geom_data_dict = load_b2f_file(mesh_file_path, debug=debug)  # geom_data_dict is needed also for magnetic field

    mesh = create_mesh_from_geom_data(geom_data_dict)

    ny = mesh.ny  # radial
    nx = mesh.nx  # poloidal

    header_dict, sim_info_dict, mesh_data_dict = load_b2f_file(b2_state_file, debug=debug)

    # Load each plasma species in simulation
    species_list = []
    neutral_indx = []
    for i in range(len(sim_info_dict['zn'])):

        zn = int(sim_info_dict['zn'][i])  # Nuclear charge
        am = int(round(float(sim_info_dict['am'][i])))  # Atomic mass number
        charge = int(sim_info_dict['zamax'][i])  # Ionisation/charge
        isotope = lookup_isotope(zn, number=am)
        species = prefer_element(isotope)  # Prefer Element over Isotope if the mass number is the same
        species_list.append((species.name, charge))
        if charge == 0:  # updating neutral index
            neutral_indx.append(i)

    sim = SOLPSSimulation(mesh, species_list)

    # Load magnetic field
    sim.b_field = geom_data_dict['bb'][:3]
    # sim.b_field_cylindrical is created authomatically

    # Load electron species
    sim.electron_temperature = mesh_data_dict['te'] / elementary_charge
    sim.electron_density = mesh_data_dict['ne']

    # Load ion temperature
    sim.ion_temperature = mesh_data_dict['ti'] / elementary_charge

    # Load species density
    sim.species_density = mesh_data_dict['na']

    # Load parallel velocity
    parallel_velocity = mesh_data_dict['ua']

    # Load poloidal and radial particle fluxes for velocity calculation
    poloidal_flux = mesh_data_dict['fna'][::2]
    radial_flux = mesh_data_dict['fna'][1::2]

    # Obtaining velocity from B2 flux
    sim.b2_flux_to_velocity(poloidal_flux, radial_flux, parallel_velocity)

    if not b2_standalone:
        # Obtaining additional data from EIRENE and replacing data for neutrals
        # Note EIRENE data grid is slightly smaller than SOLPS grid, for example (98, 38) => (96, 36)
        # Need to pad EIRENE data to fit inside larger B2 array

        neutral_density = np.zeros((len(neutral_indx), ny, nx))
        neutral_density[:, 1:-1, 1:-1] = eirene.da
        sim.species_density[neutral_indx] = neutral_density

        # Obtaining neutral atom velocity from EIRENE flux
        # Note that if the output for fluxes was turned off, eirene.ppa and eirene.rpa are all zeros
        if np.any(eirene.ppa) or np.any(eirene.rpa):
            neutral_poloidal_flux = np.zeros((len(neutral_indx), ny, nx))
            neutral_poloidal_flux[:, 1:-1, 1:-1] = eirene.ppa

            neutral_radial_flux = np.zeros((len(neutral_indx), ny, nx))
            neutral_radial_flux[:, 1:-1, 1:-1] = eirene.rpa

            neutral_parallel_velocity = np.zeros((len(neutral_indx), ny, nx))  # must be zero outside EIRENE grid
            neutral_parallel_velocity[:, 1:-1, 1:-1] = parallel_velocity[neutral_indx, 1:-1, 1:-1]

            sim.eirene_flux_to_velocity(neutral_poloidal_flux, neutral_radial_flux, neutral_parallel_velocity)

        # Obtaining neutral temperatures
        ta = np.zeros((eirene.ta.shape[0], ny, nx))
        ta[:, 1:-1, 1:-1] = eirene.ta
        # extrapolating
        for i in (0, -1):
            ta[:, i, 1:-1] = eirene.ta[:, i, :]
            ta[:, 1:-1, i] = eirene.ta[:, :, i]
        for i, j in ((0, 0), (0, -1), (-1, 0), (-1, -1)):
            ta[:, i, j] = eirene.ta[:, i, j]
        sim.neutral_temperature = ta / elementary_charge

        # Obtaining total radiation
        eradt_raw_data = eirene.eradt.sum(0)
        total_radiation = np.zeros((ny, nx))
        total_radiation[1:-1, 1:-1] = eradt_raw_data
        sim.total_radiation = total_radiation

        sim.eirene_simulation = eirene

    return sim

def create_mesh_from_geom_data(geom_data):

    r = geom_data['crx']
    z = geom_data['cry']
    vol = geom_data['vol']

    # Loading neighbouring cell indices
    neighbix = np.zeros(r.shape, dtype=np.int)
    neighbiy = np.zeros(r.shape, dtype=np.int)

    neighbix[0] = geom_data['leftix'].astype(np.int)  # poloidal prev.
    neighbix[1] = geom_data['bottomix'].astype(np.int)  # radial prev.
    neighbix[2] = geom_data['rightix'].astype(np.int)  # poloidal next
    neighbix[3] = geom_data['topix'].astype(np.int)  # radial next

    neighbiy[0] = geom_data['leftiy'].astype(np.int)
    neighbiy[1] = geom_data['bottomiy'].astype(np.int)
    neighbiy[2] = geom_data['rightiy'].astype(np.int)
    neighbiy[3] = geom_data['topiy'].astype(np.int)

    # In SOLPS cell indexing starts with -1 (guarding cell), but in SOLPSMesh -1 means no neighbour.
    neighbix += 1
    neighbiy += 1
    neighbix[neighbix == r.shape[2]] = -1
    neighbiy[neighbiy == r.shape[1]] = -1

    mesh = SOLPSMesh(r, z, vol, neighbix, neighbiy)

    return mesh
