
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

from cherab.core.atomic.elements import lookup_isotope

from cherab.solps.mesh_geometry import SOLPSMesh
from cherab.solps.solps_plasma import SOLPSSimulation, prefer_element, eirene_flux_to_velocity, b2_flux_to_velocity


def load_solps_from_balance(balance_filename):
    """
    Load a SOLPS simulation from SOLPS balance.nc output files.
    """

    el_charge = constants.elementary_charge
    rydberg_energy = constants.value('Rydberg constant times hc in eV')

    # Open the file
    with netcdf.netcdf_file(balance_filename, 'r') as fhandle:

        # Load SOLPS mesh geometry
        mesh = load_mesh_from_netcdf(fhandle)

        # Load each plasma species in simulation

        species_list = []
        neutral_indx = []
        am = np.round(fhandle.variables['am'].data).astype(int)  # Atomic mass number
        charge = fhandle.variables['za'].data.astype(int)   # Ionisation/charge
        species_names = fhandle.variables['species'].data.copy()
        ns = am.size
        for i in range(ns):
            symbol = ''.join([b.decode('utf-8').strip(' 0123456789+-') for b in species_names[i]])  # also strips isotope number
            if symbol not in ('D', 'T'):
                isotope = lookup_isotope(symbol, number=am[i])  # will throw an error for D or T
                species = prefer_element(isotope)  # Prefer Element over Isotope if the mass number is the same
            else:
                species = lookup_isotope(symbol)

            # If we only need to populate species_list, there is probably a faster way to do this...
            species_list.append((species.name, charge[i]))
            if charge[i] == 0:
                neutral_indx.append(i)

        sim = SOLPSSimulation(mesh, species_list)
        nx = mesh.nx
        ny = mesh.ny

        ##########################
        # Magnetic field vectors #
        sim.b_field = fhandle.variables['bb'].data.copy()[:3]
        # sim.b_field_cylindrical is created automatically

        # Load electron species
        sim.electron_temperature = fhandle.variables['te'].data.copy() / el_charge
        sim.electron_density = fhandle.variables['ne'].data.copy()

        # Load ion temperature
        sim.ion_temperature = fhandle.variables['ti'].data.copy() / el_charge

        # Load species density
        sim.species_density = fhandle.variables['na'].data.copy()

        # Load parallel velocity
        parallel_velocity = fhandle.variables['ua'].data.copy()

        # Load poloidal and radial particle fluxes for velocity calculation
        if 'fna_tot' in fhandle.variables:
            fna = fhandle.variables['fna_tot'].data.copy()
            # Obtaining velocity from B2 flux
            sim.velocities_cylindrical = b2_flux_to_velocity(sim, fna[:, 0], fna[:, 1], parallel_velocity)
        else:  # trying to obtain particle flux from components
            fna = 0
            for key in fhandle.variables.keys():
                if 'fna_' in key:
                    fna += fhandle.variables[key].data.copy()
            if np.any(fna != 0):
                # Obtaining velocity from B2 flux
                sim.velocities_cylindrical = b2_flux_to_velocity(sim, fna[:, 0], fna[:, 1], parallel_velocity)

        # Obtaining additional data from EIRENE and replacing data for neutrals
        if 'dab2' in fhandle.variables:
            sim.species_density[neutral_indx] = fhandle.variables['dab2'].data.copy()[:, :ny, :nx]  # in case of large grid size
            b2_standalone = False
        else:
            b2_standalone = True

        if not b2_standalone:

            # Obtaining neutral atom velocity from EIRENE flux
            # Note that if the output for fluxes was turned off, pfluxa and rfluxa' are all zeros
            if 'pfluxa' in fhandle.variables and 'rfluxa' in fhandle.variables:
                neutral_poloidal_flux = fhandle.variables['pfluxa'].data.copy()[:, :ny, :nx]
                neutral_radial_flux = fhandle.variables['rfluxa'].data.copy()[:, :ny, :nx]

                if np.any(neutral_poloidal_flux) or np.any(neutral_radial_flux):
                    neutral_velocities = eirene_flux_to_velocity(sim, neutral_poloidal_flux, neutral_radial_flux,
                                                                 parallel_velocity[neutral_indx])
                    if sim.velocities_cylindrical is not None:
                        sim.velocities_cylindrical[neutral_indx] = neutral_velocities
                        sim.velocities_cylindrical = sim.velocities_cylindrical  # Updating sim.velocities
                    else:
                        # No 'fna_*' keys in balance.nc and b2 species velocities are not set
                        velocities_cylindrical = np.zeros((len(sim.species_list, 3, ny, nx)))
                        velocities_cylindrical[neutral_indx] = neutral_velocities
                        sim.velocities_cylindrical = velocities_cylindrical

            # Obtaining neutral temperatures
            if 'tab2' in fhandle.variables:
                sim.neutral_temperature = fhandle.variables['tab2'].data.copy()[:, :ny, :nx]

            # Calculate the total radiated power
            b2_ploss = 0
            eirene_ecoolrate = 0
            eirene_potential_loss = 0

            # Total radiated power from B2, not including neutrals
            if 'b2stel_she_bal' in fhandle.variables:
                b2_ploss = np.sum(fhandle.variables['b2stel_she_bal'].data, axis=0) / mesh.vol

            # Electron energy loss due to interactions with neutrals
            if 'eirene_mc_eael_she_bal' in fhandle.variables:
                eirene_ecoolrate = np.sum(fhandle.variables['eirene_mc_eael_she_bal'].data, axis=0) / mesh.vol

            # Ionisation rate from EIRENE, needed to calculate the energy loss to overcome the ionisation potential of atoms
            if 'eirene_mc_papl_sna_bal' in fhandle.variables:
                tmp = np.sum(fhandle.variables['eirene_mc_papl_sna_bal'].data, axis=0)[1]
                eirene_potential_loss = rydberg_energy * tmp * el_charge / mesh.vol

            # This will be negative (energy sink); multiply by -1
            total_rad = -1.0 * (b2_ploss + (eirene_ecoolrate - eirene_potential_loss))

        else:
            # Total radiated power from B2, not including neutrals
            b2_ploss = 0
            potential_loss = 0

            if 'b2stel_she_bal' in fhandle.variables:
                b2_ploss = np.sum(fhandle.variables['b2stel_she_bal'].data, axis=0) / mesh.vol

            if 'b2stel_sna_ion_bal' in fhandle.variables:
                potential_loss = np.sum(fhandle.variables['b2stel_sna_ion_bal'].data, axis=0) / mesh.vol

            # Save total radiated power to the simulation object
            total_rad = rydberg_energy * el_charge * potential_loss - b2_ploss

        if np.any(total_rad != 0):
            sim.total_radiation = total_rad

    return sim


def load_mesh_from_netcdf(fhandle):

    # Load SOLPS mesh geometry
    # Re-arrange the array dimensions in the way CHERAB expects...
    r = fhandle.variables['crx'].data.copy()
    z = fhandle.variables['cry'].data.copy()
    vol = fhandle.variables['vol'].data.copy()

    # Loading neighbouring cell indices
    neighbix = np.zeros(r.shape, dtype=int)
    neighbiy = np.zeros(r.shape, dtype=int)

    neighbix[0] = fhandle.variables['leftix'].data.astype(int)  # poloidal prev.
    neighbix[1] = fhandle.variables['bottomix'].data.astype(int)  # radial prev.
    neighbix[2] = fhandle.variables['rightix'].data.astype(int)  # poloidal next
    neighbix[3] = fhandle.variables['topix'].data.astype(int)  # radial next

    neighbiy[0] = fhandle.variables['leftiy'].data.astype(int)
    neighbiy[1] = fhandle.variables['bottomiy'].data.astype(int)
    neighbiy[2] = fhandle.variables['rightiy'].data.astype(int)
    neighbiy[3] = fhandle.variables['topiy'].data.astype(int)

    # In SOLPS cell indexing starts with -1 (guarding cell), but in SOLPSMesh -1 means no neighbour.
    neighbix += 1
    neighbiy += 1
    neighbix[neighbix == r.shape[2]] = -1
    neighbiy[neighbiy == r.shape[1]] = -1

    # Create the SOLPS mesh
    mesh = SOLPSMesh(r, z, vol, neighbix, neighbiy)

    return mesh
