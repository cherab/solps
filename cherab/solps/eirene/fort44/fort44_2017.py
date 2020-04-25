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

from .parser import read_block44


def load_fort44_2017(file_handle, nx, ny, debug=False):
    """
    Read neutral species and wall flux information from fort.44.

    This is for fort.44 files with format ID 20170328.
    Specification of the data format is in Section 5.2 of the SOLPS
    manual (Running the coupled version -> Output Files).

    :param str file_handle: an open "fort.44" output file
    :param int nx: Number of grid cells in the x direction
    :param int ny: Number of grid cells in the y direction
    :param bool debug: status flag for printing debugging output
    :rtype:
    """

    data = {}

    # Read Species numbers
    line = file_handle.readline().split()
    data["na"] = int(line[0])  # number of atoms
    data["nm"] = int(line[1])  # number of molecules
    data["ni"] = int(line[2])  # number of ions
    data["ns"] = data["na"] + data["nm"] + data["ni"]  # total number of species
    if debug:
        print('Species # : {} atoms, {} molecules, {} ions, {} total species'
              .format(data["na"], data["nm"], data["ni"], data["ns"]))

    # Read Species labels
    data["species_labels"] = []
    for _ in range(data["ns"]):
        line = file_handle.readline()
        data["species_labels"].append(line.strip())
    if debug:
        print("Species labels => {}".format(data["species_labels"]))

    # Read atomic species (da, ta)
    data["da"] = read_block44(file_handle, data["na"], nx, ny)  # Atomic Neutral Density
    data["ta"] = read_block44(file_handle, data["na"], nx, ny)  # Atomic Neutral Temperature
    if debug:
        print('Atomic Neutral Density nD0: ', data["da"][0, :, 0])
        print('Atomic Neutral Temperature TD0: ', data["ta"][0, :, 0])

    # Read molecular species (dm, tm)
    data["dm"] = read_block44(file_handle, data["nm"], nx, ny)  # Molecular Neutral Density
    data["tm"] = read_block44(file_handle, data["nm"], nx, ny)  # Molecular Neutral Temperature

    # Read ion species (di, ti)
    data["di"] = read_block44(file_handle, data["ni"], nx, ny)  # Test Ion Density
    data["ti"] = read_block44(file_handle, data["ni"], nx, ny)  # Test Ion Temperature

    # Read radial particle flux (rpa, rpm)
    data["rpa"] = read_block44(file_handle, data["na"], nx, ny)  # Atomic Radial Particle Flux
    data["rpm"] = read_block44(file_handle, data["nm"], nx, ny)  # Molecular Radial Particle Flux

    # Read poloidal particle flux (ppa, ppm)
    data["ppa"] = read_block44(file_handle, data["na"], nx, ny)  # Atomic Poloidal Particle Flux
    data["ppm"] = read_block44(file_handle, data["nm"], nx, ny)  # Molecular Poloidal Particle Flux

    # Read radial energy flux (rea, rem)
    data["rea"] = read_block44(file_handle, data["na"], nx, ny)  # Atomic Radial Energy Flux
    data["rem"] = read_block44(file_handle, data["nm"], nx, ny)  # Molecular Radial Energy Flux

    # Read poloidal energy flux (pea, pem)
    data["pea"] = read_block44(file_handle, data["na"], nx, ny)  # Atomic Poloidal Energy Flux
    data["pem"] = read_block44(file_handle, data["nm"], nx, ny)  # Molecular Poloidal Energy Flux

    # Halpha total & molecules (emist, emism)
    data["emist"] = read_block44(file_handle, 1, nx, ny)  # Total Halpha Emission (including molecules)
    data["emism"] = read_block44(file_handle, 1, nx, ny)  # Molecular Halpha Emission

    # Molecular source term, unused
    _ = read_block44(file_handle, data["nm"], nx, ny)  # Molecule particle source

    # Radiated power (elosm, edism, eradt)
    data["edism"] = read_block44(file_handle, 1, nx, ny)  # Power loss due to molecule dissociation

    # Consume lines until eradt is reached
    while True:
        line = file_handle.readline().split()
        if line[0] == "*eirene" and line[3] == "eneutrad":
            break

    data["eradt"] = read_block44(file_handle, 1, nx, ny)  # Neutral radiated power

    return data
