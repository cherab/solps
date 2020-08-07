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

from cherab.solps.eirene import Eirene
from cherab.solps.eirene.parser.utility import read_block44


def load_fort44_2017(file_path, debug=False):
    """
    Read neutral species and wall flux information from fort.44.

    This is for fort.44 files with format ID 20170328.
    Specification of the data format is in Section 5.2 of the SOLPS
    manual (Running the coupled version -> Output Files).

    :param str file_path: path to the fort.44 file
    :param bool debug: status flag for printing debugging output
    :rtype:
    """

    with open(file_path, 'r') as file_handle:

        # Read sizes
        line = file_handle.readline().split()
        nx = int(line[0])
        ny = int(line[1])
        version = int(line[2])
        if debug:
            print('Geometry & Version : nx {}, ny {}, version {}'
                  .format(nx, ny, version))

        # Read Species numbers
        line = file_handle.readline().split()
        na = int(line[0])  # number of atoms
        nm = int(line[1])  # number of molecules
        ni = int(line[2])  # number of ions
        ns = na + nm + ni  # total number of species

        if debug:
            print('Species # : {} atoms, {} molecules, {} ions, {} total species'
                  .format(na, nm, ni, ns))

        # Read Species labels
        species_labels = []
        for _ in range(ns):
            line = file_handle.readline()
            species_labels.append(line.strip())
        if debug:
            print("Species labels => {}".format(species_labels))

        # create eirene object
        eirene = Eirene(nx, ny, na, nm, ni, ns, species_labels, version)

        # Read atomic species (da, ta)
        eirene.da = read_block44(file_handle, eirene.na, eirene.nx, eirene.ny)  # Atomic Neutral Density
        eirene.ta = read_block44(file_handle, eirene.na, eirene.nx, eirene.ny)  # Atomic Neutral Temperature
        if debug:
            print('Atomic Neutral Density nD0: ', eirene.da[0, :, 0])
            print('Atomic Neutral Temperature TD0: ', eirene.ta[0, :, 0])

        # Read molecular species (dm, tm)
        eirene.dm = read_block44(file_handle, eirene.nm, eirene.nx, eirene.ny)  # Molecular Neutral Density
        eirene.tm = read_block44(file_handle, eirene.nm, eirene.nx, eirene.ny)  # Molecular Neutral Temperature

        # Read ion species (di, ti)
        eirene.di = read_block44(file_handle, eirene.ni, eirene.nx, eirene.ny)  # Test Ion Density
        eirene.ti = read_block44(file_handle, eirene.ni, eirene.nx, eirene.ny)  # Test Ion Temperature

        # Read radial particle flux (rpa, rpm)
        eirene.rpa = read_block44(file_handle, eirene.na, eirene.nx, eirene.ny)  # Atomic Radial Particle Flux
        eirene.rpm = read_block44(file_handle, eirene.nm, eirene.nx, eirene.ny)  # Molecular Radial Particle Flux

        # Read poloidal particle flux (ppa, ppm)
        eirene.ppa = read_block44(file_handle, eirene.na, eirene.nx, eirene.ny)  # Atomic Poloidal Particle Flux
        eirene.ppm = read_block44(file_handle, eirene.nm, eirene.nx, eirene.ny)  # Molecular Poloidal Particle Flux

        # Read radial energy flux (rea, rem)
        eirene.rea = read_block44(file_handle, eirene.na, eirene.nx, eirene.ny)  # Atomic Radial Energy Flux
        eirene.rem = read_block44(file_handle, eirene.nm, eirene.nx, eirene.ny)  # Molecular Radial Energy Flux

        # Read poloidal energy flux (pea, pem)
        eirene.pea = read_block44(file_handle, eirene.na, eirene.nx, eirene.ny)  # Atomic Poloidal Energy Flux
        eirene.pem = read_block44(file_handle, eirene.nm, eirene.nx, eirene.ny)  # Molecular Poloidal Energy Flux

        # Halpha total & molecules (emist, emism)
        eirene.emist = read_block44(file_handle, 1, eirene.nx, eirene.ny)  # Total Halpha Emission (including molecules)
        eirene.emism = read_block44(file_handle, 1, eirene.nx, eirene.ny)  # Molecular Halpha Emission

        # Molecular source term, unused
        _ = read_block44(file_handle, eirene.nm, eirene.nx, eirene.ny)  # Molecule particle source

        # Radiated power (elosm, edism, eradt)
        eirene.edism = read_block44(file_handle, 1, eirene.nx, eirene.ny)  # Power loss due to molecule dissociation

        # Consume lines until eradt is reached
        while True:
            line = file_handle.readline().split()
            if line[0] == "*eirene" and line[3] == "eneutrad":
                break

        eirene.eradt = read_block44(file_handle, 1, eirene.nx, eirene.ny)  # Neutral radiated power

    return eirene
