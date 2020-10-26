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
from cherab.solps.eirene import Eirene
from cherab.solps.eirene.parser.utility import read_labelled_block44


def load_fort44_post_2016(file_path, debug=False):
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

        # Read all blocks of data from the file into a dictionary.
        raw_data = {}
        while True:
            try:
                raw_data.update(read_labelled_block44(file_handle))
            except EOFError:
                break

    # create eirene object
    eirene = Eirene(nx, ny, na, nm, ni, ns, species_labels, version)

    # Read atomic species data
    eirene.da = raw_data['dab2'].reshape(eirene.na, eirene.ny, eirene.nx)
    eirene.ta = raw_data['tab2'].reshape(eirene.na, eirene.ny, eirene.nx)
    eirene.rpa = raw_data['rfluxa'].reshape(eirene.na, eirene.ny, eirene.nx)
    eirene.ppa = raw_data['pfluxa'].reshape(eirene.na, eirene.ny, eirene.nx)
    eirene.rea = raw_data['refluxa'].reshape(eirene.na, eirene.ny, eirene.nx)
    eirene.pea = raw_data['pefluxa'].reshape(eirene.na, eirene.ny, eirene.nx)

    # Read molecular species data
    if eirene.nm > 0:
        eirene.dm = raw_data['dmb2'].reshape(eirene.nm, eirene.ny, eirene.nx)
        eirene.tm = raw_data['tmb2'].reshape(eirene.nm, eirene.ny, eirene.nx)
        eirene.rpm = raw_data['rfluxm'].reshape(eirene.nm, eirene.ny, eirene.nx)
        eirene.ppm = raw_data['pfluxm'].reshape(eirene.nm, eirene.ny, eirene.nx)
        eirene.rem = raw_data['refluxm'].reshape(eirene.nm, eirene.ny, eirene.nx)
        eirene.pem = raw_data['pefluxm'].reshape(eirene.nm, eirene.ny, eirene.nx)
        eirene.emism = raw_data['emissmol'].reshape(1, eirene.ny, eirene.nx)
        eirene.edism = raw_data['edissml'].reshape(eirene.nm, eirene.ny, eirene.nx)

    # Read ion species data
    if eirene.ni > 0:
        eirene.di = raw_data['dib2'].reshape(eirene.ni, eirene.ny, eirene.nx)
        eirene.ti = raw_data['tib2'].reshape(eirene.ni, eirene.ny, eirene.nx)

    # Halpha emission
    # fort.44 gives emission due to atoms and emission due to molecules separately.
    emiss_atoms = raw_data['emiss'].reshape(1, eirene.ny, eirene.nx)
    if eirene.nm > 0:
        eirene.emist = emiss_atoms + eirene.emism
    else:
        eirene.emist = emiss_atoms

    # Radiated power
    try:
        eneutrad = raw_data['eneutrad'].reshape(eirene.na, eirene.ny, eirene.nx)
    except ValueError:
        # Some Eirene/SOLPS combinations output eneutrad summed over atoms,
        # rather than resolved.
        eneutrad = raw_data['eneutrad'].reshape(1, eirene.ny, eirene.nx)

    try:
        emolrad = raw_data['emolrad'].reshape(eirene.nm, eirene.ny, eirene.nx)
    except KeyError:
        print("Warning: no emolrad data available in EIRENE simulation.")
        emolrad = 0
    except ValueError:
        # Data is summed over molecules rather than resolved.
        emolrad = raw_data['emolrad'].reshape(1, eirene.ny, eirene.nx)

    try:
        eionrad = raw_data['eionrad'].reshape(eirene.ni, eirene.ny, eirene.nx)
    except KeyError:
        print("Warning: no eionrad data available in EIRENE simulation.")
        eionrad = 0
    except ValueError:
        # Data is summed over ions rather than resolved.
        eionrad = raw_data['eionrad'].reshape(1, eirene.ny, eirene.nx)

    eirene.eradt = (np.sum(eneutrad, axis=0, keepdims=True)
                    + np.sum(emolrad, axis=0, keepdims=True)
                    + np.sum(eionrad, axis=0, keepdims=True))

    return eirene
