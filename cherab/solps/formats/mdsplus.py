
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

import re
import numpy as np
from math import sqrt
from raysect.core import Point2D
from raysect.core.math.interpolators import Discrete2DMesh

from cherab.core.math.mappers import AxisymmetricMapper
from cherab.solps.mesh_geometry import SOLPSMesh
from cherab.solps.solps_plasma import SOLPSSimulation


_SPECIES_REGEX = '([a-zA-z]+)\+?([0-9]+)'


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
    sim = SOLPSSimulation(mesh)
    ni = mesh.nx
    nj = mesh.ny

    ##########################
    # Magnetic field vectors #
    raw_b_field = np.swapaxes(conn.get('\SOLPS::TOP.SNAPSHOT.B').data(), 0, 2)
    b_field_vectors_cartesian = np.zeros((ni, nj, 3))
    b_field_vectors = np.zeros((ni, nj, 3))
    for i in range(ni):
        for j in range(nj):
            bparallel = raw_b_field[i, j, 0]
            bradial = raw_b_field[i, j, 1]
            btoroidal = raw_b_field[i, j, 2]
            b_field_vectors[i, j] = (bparallel, bradial, btoroidal)

            pv = mesh.poloidal_grid_basis[i, j, 0]  # parallel basis vector
            rv = mesh.poloidal_grid_basis[i, j, 1]  # radial basis vector

            bx = pv.x * bparallel + rv.x * bradial  # component of B along poloidal x
            by = pv.y * bparallel + rv.y * bradial  # component of B along poloidal y
            b_field_vectors_cartesian[i, j] = (bx, btoroidal, by)
    sim._b_field_vectors = b_field_vectors
    sim._b_field_vectors_cartesian = b_field_vectors_cartesian

    # Load electron species
    sim._electron_temperature = np.swapaxes(conn.get('\SOLPS::TOP.SNAPSHOT.TE').data(), 0, 1)  # (32, 98) => (98, 32)
    sim._electron_density = np.swapaxes(conn.get('\SOLPS::TOP.SNAPSHOT.NE').data(), 0, 1)  # (32, 98) => (98, 32)

    ############################
    # Load each plasma species #
    ############################

    # Master list of species, e.g. ['D0', 'D+1', 'C0', 'C+1', ...
    sim._species_list = conn.get('\SOLPS::TOP.IDENT.SPECIES').data().decode('UTF-8').split()
    sim._species_density = np.swapaxes(conn.get('\SOLPS::TOP.SNAPSHOT.NA').data(), 0, 2)
    sim._rad_par_flux = np.swapaxes(conn.get('\SOLPS::TOP.SNAPSHOT.FNAY').data(), 0, 2)  # radial particle flux
    sim._radial_area = np.swapaxes(conn.get('\SOLPS::TOP.SNAPSHOT.SY').data(), 0, 1)  # radial contact area

    # Load the neutral atom density from B2
    dab2 = conn.get('\SOLPS::TOP.SNAPSHOT.DAB2')
    if isinstance(dab2, np.ndarray):
        sim._b2_neutral_densities = np.swapaxes(dab2.data(), 0, 2)

    sim._velocities_parallel = np.swapaxes(conn.get('\SOLPS::TOP.SNAPSHOT.UA').data(), 0, 2)
    sim._velocities_radial = np.zeros((ni, nj, len(sim.species_list)))
    sim._velocities_toroidal = np.zeros((ni, nj, len(sim.species_list)))
    sim._velocities_cartesian = np.zeros((ni, nj, len(sim.species_list), 3), dtype=np.float64)

    ################################################
    # Calculate the species' velocity distribution #
    b2_neutral_i = 0  # counter for B2 neutrals
    for k, sp in enumerate(sim.species_list):

        # Identify the species based on its symbol
        symbol, charge = re.match(_SPECIES_REGEX, sp).groups()
        charge = int(charge)

        # If neutral and B" atomic density available,  use B2 density, otherwise use fluid species density.
        if charge == 0 and (sim._b2_neutral_densities != None):
            species_dens_data = sim.b2_neutral_densities[:, :, b2_neutral_i]
            b2_neutral_i += 1
        else:
            species_dens_data = sim.species_density[:, :, k]

        for i in range(ni):
            for j in range(nj):
                # Load grid basis vectors
                pv = mesh.poloidal_grid_basis[i, j, 0]  # parallel basis vector
                rv = mesh.poloidal_grid_basis[i, j, 1]  # radial basis vector

                # calculate field component ratios for velocity conversion
                bparallel = b_field_vectors[i, j, 0]
                btoroidal = b_field_vectors[i, j, 2]
                bplane = sqrt(bparallel**2 + btoroidal**2)
                parallel_to_toroidal_ratio = bparallel * btoroidal / (bplane**2)

                # Calculate toroidal and radial velocity components
                v_parallel = sim.velocities_parallel[i, j, k]  # straight from SOLPS 'UA' variable
                v_toroidal = v_parallel * parallel_to_toroidal_ratio
                sim.velocities_toroidal[i, j, k] = v_toroidal
                # Special case for edge of mesh, no radial velocity expected.
                try:
                    if species_dens_data[i, j] == 0:
                        v_radial = 0.0
                    else:
                        v_radial = sim.radial_particle_flux[i, j, k] / sim.radial_area[i, j] / species_dens_data[i, j]
                except IndexError:
                    v_radial = 0.0
                sim.velocities_radial[i, j, k] = v_radial

                # Convert velocities to cartesian coordinates
                vx = pv.x * v_parallel + rv.x * v_radial  # component of v along poloidal x
                vy = pv.y * v_parallel + rv.y * v_radial  # component of v along poloidal y
                sim.velocities_cartesian[i, j, k, :] = (vx, v_toroidal, vy)

    # Make Mesh Interpolator function for inside/outside mesh test.
    inside_outside_data = np.ones(mesh.num_tris)
    inside_outside = AxisymmetricMapper(Discrete2DMesh(mesh.vertex_coords, mesh.triangles, inside_outside_data, limit=False))
    sim._inside_mesh = inside_outside

    ###############################
    # Load extra data from server #
    ###############################

    ####################
    # Integrated power #
    vol = np.swapaxes(conn.get('\SOLPS::TOP.SNAPSHOT.VOL').data(), 0, 1)  # TODO - this should be a mesh property
    linerad = np.swapaxes(conn.get('\SOLPS::TOP.SNAPSHOT.RQRAD').data(), 0, 2)
    linerad = np.sum(linerad, axis=2)
    brmrad = np.swapaxes(conn.get('\SOLPS::TOP.SNAPSHOT.RQBRM').data(), 0, 2)
    brmrad = np.sum(brmrad, axis=2)

    total_rad_data = np.zeros(vol.shape)
    ni, nj = vol.shape
    for i in range(ni):
        for j in range(nj):
            total_rad_data[i, j] = (linerad[i, j] + brmrad[i, j]) / vol[i, j]
    sim._total_rad = total_rad_data

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
    mesh.vessel = mds_connection.get('\SOLPS::TOP.SNAPSHOT.GRID:VESSEL').data()

    # Load the centre points of the grid cells.
    cr = np.swapaxes(mds_connection.get('\TOP.SNAPSHOT.GRID:CR').data(), 0, 1)
    cz = np.swapaxes(mds_connection.get('\TOP.SNAPSHOT.GRID:CZ').data(), 0, 1)
    mesh._cr = cr
    mesh._cz = cz

    # Load cell basis vectors
    nx = mesh.nx
    ny = mesh.ny

    cell_poloidal_basis = np.empty((nx, ny, 2), dtype=object)
    for i in range(nx):
        for j in range(ny):

            # Work out cell's 2D parallel vector in the poloidal plane
            if i == nx - 1:
                # Special case for end of array, repeat previous calculation.
                # This is because I don't have access to the gaurd cells.
                xp_x = cr[i, j] - cr[i-1, j]
                xp_y = cz[i, j] - cz[i-1, j]
                norm = np.sqrt(xp_x**2 + xp_y**2)
                cell_poloidal_basis[i, j, 0] = Point2D(xp_x/norm, xp_y/norm)
            else:
                xp_x = cr[i+1, j] - cr[i, j]
                xp_y = cz[i+1, j] - cz[i, j]
                norm = np.sqrt(xp_x**2 + xp_y**2)
                cell_poloidal_basis[i, j, 0] = Point2D(xp_x/norm, xp_y/norm)

            # Work out cell's 2D radial vector in the poloidal plane
            if j == ny - 1:
                # Special case for end of array, repeat previous calculation.
                yr_x = cr[i, j] - cr[i, j-1]
                yr_y = cz[i, j] - cz[i, j-1]
                norm = np.sqrt(yr_x**2 + yr_y**2)
                cell_poloidal_basis[i, j, 1] = Point2D(yr_x/norm, yr_y/norm)
            else:
                yr_x = cr[i, j+1] - cr[i, j]
                yr_y = cz[i, j+1] - cz[i, j]
                norm = np.sqrt(yr_x**2 + yr_y**2)
                cell_poloidal_basis[i, j, 1] = Point2D(yr_x/norm, yr_y/norm)

    mesh._poloidal_grid_basis = cell_poloidal_basis

    return mesh
