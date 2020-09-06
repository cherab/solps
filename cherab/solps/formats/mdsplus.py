
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
from raysect.core import Point2D

from cherab.core.atomic.elements import lookup_isotope
from cherab.solps.mesh_geometry import SOLPSMesh
from cherab.solps.solps_plasma import SOLPSSimulation, prefer_element


# TODO: violates interface of SOLPSSimulation.... puts numpy arrays in the object where they should be function2D
def load_solps_from_mdsplus(mds_server, ref_number):
    """
    Load a SOLPS simulation from a MDSplus server.

    :param str mds_server: Server address.
    :param int ref_number: Simulation reference number.
    :rtype: SOLPSSimulation
    """

    from MDSplus import Connection as MDSConnection

    # Setup connection to server
    conn = MDSConnection(mds_server)
    conn.openTree('solps', ref_number)

    # Load SOLPS mesh geometry and lookup arrays
    mesh = load_mesh_from_mdsplus(conn)

    # Load each plasma species
    ns = conn.get('\SOLPS::TOP.IDENT.NS').data()  # Number of species
    zn = conn.get('\SOLPS::TOP.SNAPSHOT.GRID.ZN').data().astype(np.int)  # Nuclear charge
    am = np.round(conn.get('\SOLPS::TOP.SNAPSHOT.GRID.AM').data()).astype(np.int)  # Atomic mass number
    charge = conn.get('\SOLPS::TOP.SNAPSHOT.GRID.ZA').data().astype(np.int)   # Ionisation/charge

    species_list = []
    for i in range(ns):
        isotope = lookup_isotope(zn[i], number=am[i])
        species = prefer_element(isotope)  # Prefer Element over Isotope if the mass number is the same
        species_list.append((species, charge[i]))

    sim = SOLPSSimulation(mesh, species_list)
    ni = mesh.nx
    nj = mesh.ny

    ##########################
    # Magnetic field vectors #
    sim.b_field = np.swapaxes(conn.get('\SOLPS::TOP.SNAPSHOT.B').data(), 0, 2)[:, :, :3]
    # sim.b_field_cartesian is created authomatically

    # Load electron temperature and density
    sim.electron_temperature = np.swapaxes(conn.get('\SOLPS::TOP.SNAPSHOT.TE').data(), 0, 1)  # (32, 98) => (98, 32)
    sim.electron_density = np.swapaxes(conn.get('\SOLPS::TOP.SNAPSHOT.NE').data(), 0, 1)  # (32, 98) => (98, 32)

    # Load ion temperature
    sim.ion_temperature = np.swapaxes(conn.get('\SOLPS::TOP.SNAPSHOT.TI').data(), 0, 1)

    # Load species
    sim.species_density = np.swapaxes(conn.get('\SOLPS::TOP.SNAPSHOT.NA').data(), 0, 2)
    sim.radial_particle_flux = np.swapaxes(conn.get('\SOLPS::TOP.SNAPSHOT.FNAY').data(), 0, 2)  # radial particle flux
    sim.radial_area = np.swapaxes(conn.get('\SOLPS::TOP.SNAPSHOT.SY').data(), 0, 1)  # radial contact area

    # Load the neutral atom density from Eirene if available
    dab2 = conn.get('\SOLPS::TOP.SNAPSHOT.DAB2').data()
    if isinstance(dab2, np.ndarray):
        # Replace the species densities
        neutral_densities = np.swapaxes(dab2, 0, 2)

        neutral_i = 0  # counter for neutral atoms
        for k, sp in enumerate(sim.species_list):
            charge = sp[1]
            if charge == 0:
                sim.species_density[:, :, k] = neutral_densities[:, :, neutral_i]
                neutral_i += 1

    # Load the neutral atom temperature from Eirene if available
    tab2 = conn.get('\SOLPS::TOP.SNAPSHOT.TAB2').data()
    if isinstance(tab2, np.ndarray):
        sim.neutral_temperature = np.swapaxes(tab2, 0, 2)

    # TODO: Eirene data (TOP.SNAPSHOT.PFLA, TOP.SNAPSHOT.RFLA) should be used for neutral atoms.
    velocities = np.zeros((ni, nj, len(sim.species_list), 3))
    velocities[:, :, :, 0] = np.swapaxes(conn.get('\SOLPS::TOP.SNAPSHOT.UA').data(), 0, 2)
    ################################################
    # Calculate the species' velocity distribution #

    # calculate field component ratios for velocity conversion
    bplane2 = sim.b_field[:, :, 0]**2 + sim.b_field[:, :, 2]**2
    parallel_to_toroidal_ratio = sim.b_field[:, :, 0] * sim.b_field[:, :, 2] / bplane2

    # Calculate toroidal and radial velocity components
    velocities[:, :, :, 2] = velocities[:, :, :, 0] * parallel_to_toroidal_ratio[:, :, None]

    for k, sp in enumerate(sim.species_list):
        i, j = np.where(sim.species_density[:, :-1, k] > 0)
        velocities[i, j, k, 1] = sim.radial_particle_flux[i, j, k] / sim.radial_area[i, j] / sim.species_density[i, j, k]
    sim.velocities = velocities
    # sim.velocities_cartesian is created authomatically

    ###############################
    # Load extra data from server #
    ###############################

    ####################
    # Integrated power #
    linerad = np.swapaxes(conn.get('\SOLPS::TOP.SNAPSHOT.RQRAD').data(), 0, 2)
    linerad = np.sum(linerad, axis=2)
    brmrad = np.swapaxes(conn.get('\SOLPS::TOP.SNAPSHOT.RQBRM').data(), 0, 2)
    brmrad = np.sum(brmrad, axis=2)
    neurad = conn.get('\SOLPS::TOP.SNAPSHOT.ENEUTRAD').data()
    if neurad is not None:  # need to cope with fact that neurad may not be present!!!
        if len(neurad.shape) == 3:
            neurad = np.swapaxes(np.abs(np.sum(neurad, axis=0)), 0, 1)
        else:
            neurad = np.swapaxes(np.abs(neurad), 0, 1)
    else:
        neurad = np.zeros(brmrad.shape)

    sim.total_radiation = (linerad + brmrad + neurad) / mesh.vol

    return sim


def load_mesh_from_mdsplus(mds_connection):
    """
    Load the SOLPS mesh geometry for a given MDSplus connection.

    :param mds_connection: MDSplus connection object. Already set to the SOLPS tree with pulse #ID.
    """

    # Load the R, Z coordinates of the cell vertices, original coordinates are (4, 38, 98)
    # re-arrange axes (4, 38, 98) => (98, 38, 4)
    x = np.swapaxes(mds_connection.get('\TOP.SNAPSHOT.GRID:R').data(), 0, 2)
    z = np.swapaxes(mds_connection.get('\TOP.SNAPSHOT.GRID:Z').data(), 0, 2)

    vol = np.swapaxes(mds_connection.get('\SOLPS::TOP.SNAPSHOT.VOL').data(), 0, 1)

    # build mesh object
    mesh = SOLPSMesh(x, z, vol)

    #############################
    # Add additional parameters #
    #############################

    # add the vessel geometry
    vessel = mds_connection.get('\SOLPS::TOP.SNAPSHOT.GRID:VESSEL').data()
    if isinstance(vessel, np.ndarray):
        mesh.vessel = vessel

    return mesh
