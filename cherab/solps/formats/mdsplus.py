
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
from cherab.solps.solps_plasma import SOLPSSimulation, prefer_element, b2_flux_to_velocity, eirene_flux_to_velocity

from matplotlib import pyplot as plt


def load_solps_from_mdsplus(mds_server, ref_number):
    """
    Load a SOLPS simulation from a MDSplus server.

    :param str mds_server: Server address.
    :param int ref_number: Simulation reference number.
    :rtype: SOLPSSimulation
    """

    from MDSplus import Connection as MDSConnection, mdsExceptions

    # Setup connection to server
    conn = MDSConnection(mds_server)
    conn.openTree('solps', ref_number)

    # Load SOLPS mesh geometry and lookup arrays
    mesh = load_mesh_from_mdsplus(conn, mdsExceptions)

    # Load each plasma species in simulation
    ns = conn.get('\SOLPS::TOP.IDENT.NS').data()  # Number of species
    zn = conn.get('\SOLPS::TOP.SNAPSHOT.GRID.ZN').data().astype(np.int)  # Nuclear charge
    am = np.round(conn.get('\SOLPS::TOP.SNAPSHOT.GRID.AM').data()).astype(np.int)  # Atomic mass number
    charge = conn.get('\SOLPS::TOP.SNAPSHOT.GRID.ZA').data().astype(np.int)   # Ionisation/charge

    species_list = []
    neutral_indx = []
    for i in range(ns):
        isotope = lookup_isotope(zn[i], number=am[i])
        species = prefer_element(isotope)  # Prefer Element over Isotope if the mass number is the same
        species_list.append((species.name, charge[i]))
        if charge[i] == 0:
            neutral_indx.append(i)

    sim = SOLPSSimulation(mesh, species_list)
    ni = mesh.nx
    nj = mesh.ny

    ##########################
    # Magnetic field vectors #
    sim.set_b_field(np.swapaxes(conn.get('\SOLPS::TOP.SNAPSHOT.B').data(), 0, 2)[:, :, :3])
    # sim.b_field_cartesian is created authomatically

    # Load electron temperature and density
    sim.set_electron_temperature(np.swapaxes(conn.get('\SOLPS::TOP.SNAPSHOT.TE').data(), 0, 1))  # (32, 98) => (98, 32)
    sim.set_electron_density(np.swapaxes(conn.get('\SOLPS::TOP.SNAPSHOT.NE').data(), 0, 1))  # (32, 98) => (98, 32)

    # Load ion temperature
    sim.set_ion_temperature(np.swapaxes(conn.get('\SOLPS::TOP.SNAPSHOT.TI').data(), 0, 1))

    # Load species density
    species_density = np.swapaxes(conn.get('\SOLPS::TOP.SNAPSHOT.NA').data(), 0, 2)

    # Load parallel velocity
    parallel_velocity = np.swapaxes(conn.get('\SOLPS::TOP.SNAPSHOT.UA').data(), 0, 2)

    # Load poloidal and radial particle fluxes for velocity calculation
    poloidal_flux = np.swapaxes(conn.get('\SOLPS::TOP.SNAPSHOT.FNAX').data(), 0, 2)
    radial_flux = np.swapaxes(conn.get('\SOLPS::TOP.SNAPSHOT.FNAY').data(), 0, 2)

    # B2 fluxes are defined between cells, so correcting array shapes if needed
    if poloidal_flux.shape[0] == ni - 1:
        poloidal_flux = np.vstack((np.zeros((1, nj, ns)), poloidal_flux))

    if radial_flux.shape[1] == nj - 1:
        radial_flux = np.hstack((np.zeros((ni, 1, ns)), radial_flux))

    # Obtaining velocity from B2 flux
    velocities_cartesian = b2_flux_to_velocity(mesh, species_density, poloidal_flux, radial_flux, parallel_velocity, sim.b_field_cartesian)

    # Obtaining additional data from EIRENE and replacing data for neutrals

    b2_standalone = False
    try:
        # Replace the species densities
        neutral_density = np.swapaxes(conn.get('\SOLPS::TOP.SNAPSHOT.DAB2').data(), 0, 2)
        species_density[:, :, neutral_indx] = neutral_density
    except (mdsExceptions.TreeNNF, np.AxisError):
        print("Warning! This is B2 stand-alone simulation.")
        b2_standalone = True

    if not b2_standalone:
        # Obtaining neutral atom velocity from EIRENE flux
        # Note that if the output for fluxes was turned off, PFLA and RFLA' are all zeros
        try:
            neutral_poloidal_flux = np.swapaxes(conn.get('\SOLPS::TOP.SNAPSHOT.PFLA').data(), 0, 2)
            neutral_radial_flux = np.swapaxes(conn.get('\SOLPS::TOP.SNAPSHOT.RFLA').data(), 0, 2)

            if np.any(neutral_poloidal_flux) or np.any(neutral_radial_flux):
                neutral_velocities_cartesian = eirene_flux_to_velocity(mesh, neutral_density, neutral_poloidal_flux, neutral_radial_flux,
                                                                       parallel_velocity[:, :, neutral_indx], sim.b_field_cartesian)

                velocities_cartesian[:, :, neutral_indx, :] = neutral_velocities_cartesian
        except (mdsExceptions.TreeNNF, np.AxisError):
            pass

        # Obtaining neutral temperatures
        try:
            sim.set_neutral_temperature(np.swapaxes(conn.get('\SOLPS::TOP.SNAPSHOT.TAB2').data(), 0, 2))
        except (mdsExceptions.TreeNNF, np.AxisError):
            pass

    sim.set_species_density(species_density)
    sim.set_velocities_cartesian(velocities_cartesian)  # this also updates sim.velocities

    ###############################
    # Load extra data from server #
    ###############################

    ####################
    # Integrated power #
    try:
        linerad = np.swapaxes(conn.get('\SOLPS::TOP.SNAPSHOT.RQRAD').data(), 0, 2)
        linerad = np.sum(linerad, axis=2)
    except (mdsExceptions.TreeNNF, np.AxisError):
        linerad = 0

    try:
        brmrad = np.swapaxes(conn.get('\SOLPS::TOP.SNAPSHOT.RQBRM').data(), 0, 2)
        brmrad = np.sum(brmrad, axis=2)
    except (mdsExceptions.TreeNNF, np.AxisError):
        brmrad = 0

    try:
        eneutrad = conn.get('\SOLPS::TOP.SNAPSHOT.ENEUTRAD').data()
        if np.ndim(eneutrad) == 3:  # this will not return error if eneutrad is not np.ndarray
            neurad = np.swapaxes(np.abs(np.sum(eneutrad, axis=0)), 0, 1)
        else:
            neurad = np.swapaxes(np.abs(eneutrad), 0, 1)
    except (mdsExceptions.TreeNNF, np.AxisError):
        neurad = 0

    sim.set_total_radiation((linerad + brmrad + neurad) / mesh.vol)

    return sim


def load_mesh_from_mdsplus(mds_connection, mdsExceptions):
    """
    Load the SOLPS mesh geometry for a given MDSplus connection.

    :param mds_connection: MDSplus connection object. Already set to the SOLPS tree with pulse #ID.
    :param mdsExceptions: MDSplus mdsExceptions module for error handling.
    """

    # Load the R, Z coordinates of the cell vertices, original coordinates are (4, 38, 98)
    # re-arrange axes (4, 38, 98) => (98, 38, 4)
    r = np.swapaxes(mds_connection.get('\TOP.SNAPSHOT.GRID:R').data(), 0, 2)
    z = np.swapaxes(mds_connection.get('\TOP.SNAPSHOT.GRID:Z').data(), 0, 2)

    vol = np.swapaxes(mds_connection.get('\SOLPS::TOP.SNAPSHOT.VOL').data(), 0, 1)

    # Loading neighbouring cell indices
    neighbix = np.zeros(r.shape, dtype=np.int)
    neighbiy = np.zeros(r.shape, dtype=np.int)

    neighbix[:, :, 0] = np.swapaxes(mds_connection.get('\SOLPS::TOP.SNAPSHOT.GRID:LEFTIX').data().astype(np.int), 0, 1)
    neighbix[:, :, 1] = np.swapaxes(mds_connection.get('\SOLPS::TOP.SNAPSHOT.GRID:BOTTOMIX').data().astype(np.int), 0, 1)
    neighbix[:, :, 2] = np.swapaxes(mds_connection.get('\SOLPS::TOP.SNAPSHOT.GRID:RIGHTIX').data().astype(np.int), 0, 1)
    neighbix[:, :, 3] = np.swapaxes(mds_connection.get('\SOLPS::TOP.SNAPSHOT.GRID:TOPIX').data().astype(np.int), 0, 1)

    neighbiy[:, :, 0] = np.swapaxes(mds_connection.get('\SOLPS::TOP.SNAPSHOT.GRID:LEFTIY').data().astype(np.int), 0, 1)
    neighbiy[:, :, 1] = np.swapaxes(mds_connection.get('\SOLPS::TOP.SNAPSHOT.GRID:BOTTOMIY').data().astype(np.int), 0, 1)
    neighbiy[:, :, 2] = np.swapaxes(mds_connection.get('\SOLPS::TOP.SNAPSHOT.GRID:RIGHTIY').data().astype(np.int), 0, 1)
    neighbiy[:, :, 3] = np.swapaxes(mds_connection.get('\SOLPS::TOP.SNAPSHOT.GRID:TOPIY').data().astype(np.int), 0, 1)

    neighbix[neighbix == r.shape[0]] = -1
    neighbiy[neighbiy == r.shape[1]] = -1

    # build mesh object
    mesh = SOLPSMesh(r, z, vol, neighbix, neighbiy)

    #############################
    # Add additional parameters #
    #############################

    # add the vessel geometry
    try:
        vessel = mds_connection.get('\SOLPS::TOP.SNAPSHOT.GRID:VESSEL').data()
        if isinstance(vessel, np.ndarray):
            mesh.vessel = vessel
    except mdsExceptions.TreeNNF:
        pass

    return mesh
