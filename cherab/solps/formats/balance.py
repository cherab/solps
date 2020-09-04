
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
from cherab.core.atomic.elements import lookup_isotope, deuterium

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
    cr_x = fhandle.variables['crx'].data.copy()
    cr_z = fhandle.variables['cry'].data.copy()
    vol = fhandle.variables['vol'].data.copy()

    # Re-arrange the array dimensions in the way CHERAB expects...
    cr_x = np.moveaxis(cr_x, 0, -1)
    cr_z = np.moveaxis(cr_z, 0, -1)

    # Create the SOLPS mesh
    mesh = SOLPSMesh(cr_x, cr_z, vol)

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
        species_list.append((species, charge))

    sim = SOLPSSimulation(mesh, species_list)

    # TODO: add code to load SOLPS velocities and magnetic field from files

    # Load electron species
    sim.electron_temperature = fhandle.variables['te'].data.copy() / el_charge
    sim.electron_density = fhandle.variables['ne'].data.copy()

    # Load ion temperature
    sim.ion_temperature = fhandle.variables['ti'].data.copy() / el_charge

    tmp = fhandle.variables['na'].data.copy()
    tmp = np.moveaxis(tmp, 0, -1)
    sim.species_density = tmp

    # Load the neutrals data
    try:
        D0_indx = sim.species_list.index((deuterium, 0))
    except ValueError:
        D0_indx = None

    # Replace the deuterium neutrals density (from the fluid neutrals model by default) with
    # the values calculated by EIRENE - do the same for other neutrals?
    if 'dab2' in fhandle.variables.keys():
        if D0_indx is not None:
            b2_len = np.shape(sim.species_density[:, :, D0_indx])[-1]
            eirene_len = np.shape(fhandle.variables['dab2'].data)[-1]
            sim.species_density[:, :, D0_indx] = fhandle.variables['dab2'].data[0, :, 0:b2_len - eirene_len]

        eirene_run = True
    else:
        eirene_run = False

    # Calculate the total radiated power
    if eirene_run:
        # Total radiated power from B2, not including neutrals
        b2_ploss = np.sum(fhandle.variables['b2stel_she_bal'].data, axis=0) / vol

        # Electron energy loss due to interactions with neutrals
        if 'eirene_mc_eael_she_bal' in fhandle.variables.keys():
            eirene_ecoolrate = np.sum(fhandle.variables['eirene_mc_eael_she_bal'].data, axis=0) / vol

        # Ionisation rate from EIRENE, needed to calculate the energy loss to overcome the ionisation potential of atoms
        if 'eirene_mc_papl_sna_bal' in fhandle.variables.keys():
            eirene_potential_loss = rydberg_energy * np.sum(fhandle.variables['eirene_mc_papl_sna_bal'].data, axis=(0))[1, :, :] * el_charge / vol

        # This will be negative (energy sink); multiply by -1
        sim.total_radiation = -1.0 * (b2_ploss + (eirene_ecoolrate - eirene_potential_loss))

    else:
        # Total radiated power from B2, not including neutrals
        b2_ploss = np.sum(fhandle.variables['b2stel_she_bal'].data, axis=0) / vol

        potential_loss = np.sum(fhandle.variables['b2stel_sna_ion_bal'].data, axis=0) / vol

        # Save total radiated power to the simulation object
        sim.total_radiation = rydberg_energy * el_charge * potential_loss - b2_ploss

    fhandle.close()

    return sim
