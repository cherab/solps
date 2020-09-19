
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

from cherab.core.atomic.elements import lookup_isotope
from cherab.solps.mesh_geometry import SOLPSMesh
from cherab.solps.solps_plasma import SOLPSSimulation, prefer_element, eirene_flux_to_velocity, b2_flux_to_velocity


def load_solps_from_mdsplus(mds_server, ref_number):
    """
    Load a SOLPS simulation from a MDSplus server.

    :param str mds_server: Server address.
    :param int ref_number: Simulation reference number.
    :rtype: SOLPSSimulation
    """

    from MDSplus import Connection as MDSConnection, MdsException

    # Setup connection to server
    conn = MDSConnection(mds_server)
    conn.openTree('solps', ref_number)

    # Load SOLPS mesh geometry and lookup arrays
    mesh = load_mesh_from_mdsplus(conn, MdsException)

    # Load each plasma species in simulation
    ns = conn.get(r'\SOLPS::TOP.IDENT.NS').data()  # Number of species
    zn = conn.get(r'\SOLPS::TOP.SNAPSHOT.GRID.ZN').data().astype(np.int)  # Nuclear charge
    am = np.round(conn.get(r'\SOLPS::TOP.SNAPSHOT.GRID.AM').data()).astype(np.int)  # Atomic mass number
    charge = conn.get(r'\SOLPS::TOP.SNAPSHOT.GRID.ZA').data().astype(np.int)   # Ionisation/charge

    species_list = []
    neutral_indx = []
    for i in range(ns):
        isotope = lookup_isotope(zn[i], number=am[i])
        species = prefer_element(isotope)  # Prefer Element over Isotope if the mass number is the same
        species_list.append((species.name, charge[i]))
        if charge[i] == 0:
            neutral_indx.append(i)

    sim = SOLPSSimulation(mesh, species_list)
    nx = mesh.nx
    ny = mesh.ny

    ##########################
    # Magnetic field vectors #
    sim.b_field = conn.get(r'\SOLPS::TOP.SNAPSHOT.B').data()[:3]
    # sim.b_field_cylindrical is created automatically

    # Load electron temperature and density
    sim.electron_temperature = conn.get(r'\SOLPS::TOP.SNAPSHOT.TE').data()
    sim.electron_density = conn.get(r'\SOLPS::TOP.SNAPSHOT.NE').data()

    # Load ion temperature
    sim.ion_temperature = conn.get(r'\SOLPS::TOP.SNAPSHOT.TI').data()

    # Load species density
    sim.species_density = conn.get(r'\SOLPS::TOP.SNAPSHOT.NA').data()

    # Load parallel velocity
    parallel_velocity = conn.get(r'\SOLPS::TOP.SNAPSHOT.UA').data()

    # Load poloidal and radial particle fluxes for velocity calculation
    poloidal_flux = conn.get(r'\SOLPS::TOP.SNAPSHOT.FNAX').data()
    radial_flux = conn.get(r'\SOLPS::TOP.SNAPSHOT.FNAY').data()

    # B2 fluxes are defined between cells, so correcting array shapes if needed
    if poloidal_flux.shape[2] == nx - 1:
        poloidal_flux = np.concatenate((np.zeros((ns, ny, 1)), poloidal_flux), axis=2)

    if radial_flux.shape[1] == ny - 1:
        radial_flux = np.concatenate((np.zeros((ns, 1, nx)), radial_flux), axis=1)

    # Setting velocities from B2 flux
    sim.velocities_cylindrical = b2_flux_to_velocity(sim, poloidal_flux, radial_flux, parallel_velocity)

    # Obtaining additional data from EIRENE and replacing data for neutrals

    try:
        # Replace the species densities
        neutral_density = conn.get(r'\SOLPS::TOP.SNAPSHOT.DAB2').data()  # this will throw a TypeError is neutral_density is not an array
        # We can update the data without re-initialising interpolators because they use pointers
        sim.species_density[neutral_indx] = neutral_density[:]

    except (MdsException, TypeError):
        print("Warning! This is B2 stand-alone simulation.")
        b2_standalone = True
    else:
        b2_standalone = False

    if not b2_standalone:
        # Obtaining neutral atom velocity from EIRENE flux
        # Note that if the output for fluxes was turned off, PFLA and RFLA' are all zeros
        try:
            neutral_poloidal_flux = conn.get(r'\SOLPS::TOP.SNAPSHOT.PFLA').data()[:]
            neutral_radial_flux = conn.get(r'\SOLPS::TOP.SNAPSHOT.RFLA').data()[:]

            if np.any(neutral_poloidal_flux) or np.any(neutral_radial_flux):
                sim.velocities_cylindrical[neutral_indx] = eirene_flux_to_velocity(sim, neutral_poloidal_flux, neutral_radial_flux,
                                                                                   parallel_velocity[neutral_indx])
                sim.velocities_cylindrical = sim.velocities_cylindrical  # Updating sim.velocities

        except (MdsException, TypeError):
            pass

        # Obtaining neutral temperatures
        try:
            sim.neutral_temperature = conn.get(r'\SOLPS::TOP.SNAPSHOT.TAB2').data()[:]
        except (MdsException, TypeError):
            pass

    ###############################
    # Load extra data from server #
    ###############################

    ####################
    # Integrated power #
    try:
        linerad = np.sum(conn.get(r'\SOLPS::TOP.SNAPSHOT.RQRAD').data()[:], axis=0)
    except (MdsException, TypeError):
        linerad = 0

    try:
        brmrad = np.sum(conn.get(r'\SOLPS::TOP.SNAPSHOT.RQBRM').data()[:], axis=0)
    except (MdsException, TypeError):
        brmrad = 0

    try:
        eneutrad = conn.get(r'\SOLPS::TOP.SNAPSHOT.ENEUTRAD').data()[:]
        if np.ndim(eneutrad) == 3:
            neurad = np.abs(np.sum(eneutrad, axis=0))
        else:
            neurad = np.abs(eneutrad)
    except (MdsException, TypeError):
        neurad = 0

    total_rad = linerad + brmrad + neurad

    if total_rad is not 0:
        sim.total_radiation = total_rad / mesh.vol

    return sim


def load_mesh_from_mdsplus(mds_connection, MdsException):
    """
    Load the SOLPS mesh geometry for a given MDSplus connection.

    :param mds_connection: MDSplus connection object. Already set to the SOLPS tree with pulse #ID.
    :param mdsExceptions: MDSplus mdsExceptions module for error handling.
    """

    # Load the R, Z coordinates of the cell vertices, original coordinates are (4, 38, 98)
    r = mds_connection.get(r'\TOP.SNAPSHOT.GRID:R').data()
    z = mds_connection.get(r'\TOP.SNAPSHOT.GRID:Z').data()

    vol = mds_connection.get(r'\SOLPS::TOP.SNAPSHOT.VOL').data()

    # Loading neighbouring cell indices
    neighbix = np.zeros(r.shape, dtype=np.int)
    neighbiy = np.zeros(r.shape, dtype=np.int)

    neighbix[0] = mds_connection.get(r'\SOLPS::TOP.SNAPSHOT.GRID:LEFTIX').data().astype(np.int)
    neighbix[1] = mds_connection.get(r'\SOLPS::TOP.SNAPSHOT.GRID:BOTTOMIX').data().astype(np.int)
    neighbix[2] = mds_connection.get(r'\SOLPS::TOP.SNAPSHOT.GRID:RIGHTIX').data().astype(np.int)
    neighbix[3] = mds_connection.get(r'\SOLPS::TOP.SNAPSHOT.GRID:TOPIX').data().astype(np.int)

    neighbiy[0] = mds_connection.get(r'\SOLPS::TOP.SNAPSHOT.GRID:LEFTIY').data().astype(np.int)
    neighbiy[1] = mds_connection.get(r'\SOLPS::TOP.SNAPSHOT.GRID:BOTTOMIY').data().astype(np.int)
    neighbiy[2] = mds_connection.get(r'\SOLPS::TOP.SNAPSHOT.GRID:RIGHTIY').data().astype(np.int)
    neighbiy[3] = mds_connection.get(r'\SOLPS::TOP.SNAPSHOT.GRID:TOPIY').data().astype(np.int)

    neighbix[neighbix == r.shape[2]] = -1
    neighbiy[neighbiy == r.shape[1]] = -1

    # build mesh object
    mesh = SOLPSMesh(r, z, vol, neighbix, neighbiy)

    #############################
    # Add additional parameters #
    #############################

    # add the vessel geometry
    try:
        vessel = mds_connection.get(r'\SOLPS::TOP.SNAPSHOT.GRID:VESSEL').data()[:]
        mesh.vessel = vessel
    except (MdsException, TypeError):
        pass

    return mesh
