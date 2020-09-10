
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

import numpy as np
from scipy.io import netcdf
from scipy import constants
from raysect.core.math.function.float import Discrete2DMesh

from cherab.core.math.mappers import AxisymmetricMapper
from cherab.core.atomic.elements import lookup_isotope

from cherab.solps.mesh_geometry import SOLPSMesh
from cherab.solps.solps_plasma import SOLPSSimulation, prefer_element


def load_solps_from_balance(balance_filename):
    """
    Load a SOLPS simulation from SOLPS balance.nc output files.
    """

    el_charge = constants.elementary_charge
    rydberg_energy = constants.value('Rydberg constant times hc in eV')

    # Open the file
    fhandle = netcdf.netcdf_file(balance_filename, 'r')

    # Load SOLPS mesh geometry
    mesh = load_mesh_from_netcdf(fhandle)

    # Load each plasma species in simulation

    species_list = []
    n_species = len(fhandle.variables['am'].data)
    for i in range(n_species):

        # Extract the nuclear charge
        if fhandle.variables['species'].data[i, 1] == b'D':
            zn = 1
        if fhandle.variables['species'].data[i, 1] == b'C':
            zn = 6
        if fhandle.variables['species'].data[i, 1] == b'N':
            zn = 7
        if fhandle.variables['species'].data[i, 1] == b'N' and fhandle.variables['species'].data[i, 2] == b'e':
            zn = 10
        if fhandle.variables['species'].data[i, 1] == b'A' and fhandle.variables['species'].data[i, 2] == b'r':
            zn = 18

        am = int(round(float(fhandle.variables['am'].data[i])))  # Atomic mass
        charge = int(fhandle.variables['za'].data[i])  # Ionisation/charge
        isotope = lookup_isotope(zn, number=am)
        species = prefer_element(isotope)  # Prefer Element over Isotope if the mass number is the same

        # If we only need to populate species_list, there is probably a faster way to do this...
        species_list.append((species.name, charge))

    sim = SOLPSSimulation(mesh, species_list)

    # TODO: add code to load SOLPS velocities and magnetic field from files

    # Load electron species
    sim.electron_temperature = fhandle.variables['te'].data.copy().T / el_charge
    sim.electron_density = fhandle.variables['ne'].data.copy().T

    # Load ion temperature
    sim.ion_temperature = fhandle.variables['ti'].data.copy().T / el_charge

    species_density = np.transpose(fhandle.variables['na'].data, (0, 2, 1))

    # Load the neutrals data
    try:
        D0_indx = sim.species_list.index(("deuterium", 0))
    except ValueError:
        D0_indx = None

    # Replace the deuterium neutrals density (from the fluid neutrals model by default) with
    # the values calculated by EIRENE - do the same for other neutrals?
    if 'dab2' in fhandle.variables.keys():
        if D0_indx is not None:
            b2_len = np.shape(species_density[D0_indx])[-2]
            dab2 = fhandle.variables['dab2'].data.copy()[0].T
            eirene_len = np.shape(fhandle.variables['dab2'].data)[-2]
            species_density[D0_indx] = dab2[0:b2_len - eirene_len, :]

        eirene_run = True
    else:
        eirene_run = False

    sim.species_density = species_density

    # Calculate the total radiated power
    if eirene_run:
        # Total radiated power from B2, not including neutrals
        b2_ploss = np.sum(fhandle.variables['b2stel_she_bal'].data, axis=0).T / mesh.vol

        # Electron energy loss due to interactions with neutrals
        if 'eirene_mc_eael_she_bal' in fhandle.variables.keys():
            eirene_ecoolrate = np.sum(fhandle.variables['eirene_mc_eael_she_bal'].data, axis=0).T / mesh.vol

        # Ionisation rate from EIRENE, needed to calculate the energy loss to overcome the ionisation potential of atoms
        if 'eirene_mc_papl_sna_bal' in fhandle.variables.keys():
            tmp = np.sum(fhandle.variables['eirene_mc_papl_sna_bal'].data, axis=(0))[1].T
            eirene_potential_loss = rydberg_energy * tmp * el_charge / mesh.vol

        # This will be negative (energy sink); multiply by -1
        sim.total_radiation = -1.0 * (b2_ploss + (eirene_ecoolrate - eirene_potential_loss))

    else:
        # Total radiated power from B2, not including neutrals
        b2_ploss = np.sum(fhandle.variables['b2stel_she_bal'].data, axis=0).T / mesh.vol

        potential_loss = np.sum(fhandle.variables['b2stel_sna_ion_bal'].data, axis=0).T / mesh.vol

        # Save total radiated power to the simulation object
        sim.total_radiation = rydberg_energy * el_charge * potential_loss - b2_ploss

    fhandle.close()

    return sim


def load_mesh_from_netcdf(fhandle):

    # Load SOLPS mesh geometry
    # Re-arrange the array dimensions in the way CHERAB expects...
    r = np.transpose(fhandle.variables['crx'].data, (0, 2, 1))
    z = np.transpose(fhandle.variables['cry'].data, (0, 2, 1))
    vol = fhandle.variables['vol'].data.copy().T

    # Loading neighbouring cell indices
    neighbix = np.zeros(r.shape, dtype=np.int)
    neighbiy = np.zeros(r.shape, dtype=np.int)

    neighbix[0] = fhandle.variables['leftix'].data.copy().astype(np.int).T  # poloidal prev.
    neighbix[1] = fhandle.variables['bottomix'].data.copy().astype(np.int).T  # radial prev.
    neighbix[2] = fhandle.variables['rightix'].data.copy().astype(np.int).T  # poloidal next
    neighbix[3] = fhandle.variables['topix'].data.copy().astype(np.int).T  # radial next

    neighbiy[0] = fhandle.variables['leftiy'].data.copy().astype(np.int).T
    neighbiy[1] = fhandle.variables['bottomiy'].data.copy().astype(np.int).T
    neighbiy[2] = fhandle.variables['rightiy'].data.copy().astype(np.int).T
    neighbiy[3] = fhandle.variables['topiy'].data.copy().astype(np.int).T

    # In SOLPS cell indexing starts with -1 (guarding cell), but in SOLPSMesh -1 means no neighbour.
    if neighbix.min() < -1 or neighbiy.min() < -1:
        neighbix += 1
        neighbiy += 1
    neighbix[neighbix == r.shape[2]] = -1
    neighbiy[neighbiy == r.shape[1]] = -1

    # Create the SOLPS mesh
    mesh = SOLPSMesh(r, z, vol, neighbix, neighbiy)

    return mesh
