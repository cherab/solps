
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
import copy
import numpy as np
from scipy.io import netcdf
from scipy.constants import elementary_charge
from raysect.core.math import Discrete2DMesh

from cherab.core.math.mappers import AxisymmetricMapper
from cherab.core.atomic.elements import hydrogen, deuterium, helium, beryllium, carbon, nitrogen, oxygen, neon, \
    argon, krypton, xenon

from cherab.solps.eirene import Eirene
from cherab.solps.mesh_geometry import SOLPSMesh
from cherab.solps.solps_plasma import SOLPSSimulation

Q = elementary_charge

# key is nuclear charge Z and atomic mass AMU
_popular_species = {
    (1, 2): deuterium,
    (6, 12.0): carbon,
    (2, 4.003): helium,
    (7, 14.0): nitrogen,
    (10, 20.180): neon,
    (18, 39.948): argon,
    (18, 40.0): argon,
    (36, 83.798): krypton,
    (54, 131.293): xenon
}


# Code developed by J. Harrison 9/4/2019
def load_solps_from_balance(balance_filename):
    """
    Load a SOLPS simulation from SOLPS balance.nc output files.

    """

    # Open the file
    fhandle = netcdf.netcdf_file(balance_filename,'r')
	
    # Load SOLPS mesh geometry
    cr_x = copy.deepcopy(fhandle.variables['crx'].data)
    cr_z = copy.deepcopy(fhandle.variables['cry'].data)
    vol = copy.deepcopy(fhandle.variables['vol'].data)
	
    # Re-arrange the array dimensions in the way CHERAB expects...
    cr_x = np.moveaxis(cr_x, 0, -1)
    cr_z = np.moveaxis(cr_z, 0, -1)
	
    # Create the SOLPS mesh
    mesh = SOLPSMesh(cr_x,cr_z,vol)	

    sim = SOLPSSimulation(mesh)
    ni = mesh.nx
    nj = mesh.ny

    # TODO: add code to load SOLPS velocities and magnetic field from files

    # Load electron species
    sim._electron_temperature = copy.deepcopy(fhandle.variables['te'].data)/Q
    sim._electron_density = copy.deepcopy(fhandle.variables['ne'].data)

    ##########################################
    # Load each plasma species in simulation #
    ##########################################

    sim._species_list = []
    n_species = len(fhandle.variables['am'].data)
	
    for i in range(n_species):

        # Extract the nuclear charge	
        if fhandle.variables['species'].data[i,1] == b'D':
	        zn = 1
        if fhandle.variables['species'].data[i,1] == b'C':
            zn = 6
        if fhandle.variables['species'].data[i,1] == b'N':
            zn = 7
        if fhandle.variables['species'].data[i,1] == b'N' and fhandle.variables['species'].data[i,2] == b'e':
            zn = 10
        if fhandle.variables['species'].data[i,1] == b'A' and fhandle.variables['species'].data[i,2] == b'r':
            zn = 18
	
        am = float(fhandle.variables['am'].data[i])  # Atomic mass
        charge = int(fhandle.variables['za'].data[i])  # Ionisation/charge
        species = _popular_species[(zn, am)]
		
        # If we only need to populate species_list, there is probably a faster way to do this...		
        sim.species_list.append(species.symbol + str(charge))

    tmp = copy.deepcopy(fhandle.variables['na'].data)
    tmp = np.moveaxis(tmp, 0, -1)
    sim._species_density = tmp

    # Make Mesh Interpolator function for inside/outside mesh test.
    inside_outside_data = ones(mesh.num_tris)
    inside_outside = AxisymmetricMapper(Discrete2DMesh(mesh.vertex_coords, mesh.triangles, inside_outside_data, limit=False))
    sim._inside_mesh = inside_outside
	
    # Load the neutrals data
    if 'D0' in sim.species_list:
        for i in arange(len(sim.species_list)):
            if sim.species_list[i] == 'D0':
                D0_indx = i

    # Replace the deuterium neutrals density (from the fluid neutrals model by default) with
    # the values calculated by EIRENE - do the same for other neutrals?
    if 'dab2' in fhandle.variables.keys():
        sim.species_density[:,:,D0_indx] = copy.deepcopy(fhandle.variables['dab2'].data[0,:,0:-2])
        eirene_run = True
    else:
        eirene_run = False
    
    # Calculate the total radiated power
    
    if eirene_run:
        # Total radiated power from B2, not including neutrals
        b2_ploss = fhandle.variables['b2stel_she_bal'].data/vol
        
        # Electron energy loss due to interactions with neutrals
        if 'eirene_mc_eael_she_bal' in fhandle.variables.keys():
            eirene_ecoolrate = np.sum(fhandle.variables['eirene_mc_eael_she_bal'].data,axis=0)/vol
            
        # Ionisation rate from EIRENE, needed to calculate the energy loss to overcome the ionisation potential of atoms
        if 'eirene_mc_papl_sna_bal' in fhandle.variables.keys():
            eirene_potential_loss = 13.6*np.sum(fhandle.variables['eirene_mc_papl_sna_bal'].data,axis=(0))[1,:,:]*Q/vol
        
        # This will be negative (energy sink); multiply by -1
        sim._total_rad = -1.0*(b2_ploss+(eirene_ecoolrate-eirene_potential_loss))
        
    else:
 
        # Total radiated power from B2, not including neutrals
        b2_ploss = np.sum(fhandle.variables['b2stel_she_bal'].data,axis=0)/vol
        
        potential_loss = np.sum(fhandle.variables['b2stel_sna_ion_bal'].data,axis=0)/vol
        
	# Save total radiated power to the simulation object
        sim._total_rad = 13.6*Q*potential_loss-b2_ploss
	
    fhandle.close()	

    return sim
