
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
import numpy as np


SIM_INFO_DATA = 0
MESH_DATA = 1


# Code based on a script by Felix Reimold (2016)

def load_b2f_file(filepath, debug=False, header_dict=None):
    """
    File for parsing the 'b2fstate and b2fplasmf' B2 Eirene output files.

    :param str filepath: full path for file to load and parse
    :param bool debug: flag for displaying textual debugging information.
    :param dict header_dict: header information
    :return: tuple of dictionaries. First is the header information such
      as the version, label, grid size, etc. Second contains arrays of
      simulation outputs. Third contains information about the simulation
      itself.

    If header_dict is provided, it should contain the dimension info 'nx',
    'ny', 'nxg', 'nyg', 'nxy', 'nxyg', as obtained by reading a previous
    b2f* file. If header_dict is None, these values will be read from the
    top of the file.
    """

    # Inline function for mapping str data to floats, reshaping arrays, and loading into SOLPSData object.
    def _make_solps_data_object(_data):

        # Convert list of strings to list of floats
        for idx, item in enumerate(_data):
            _data[idx] = float(item)

        # Multiple 2D data field (e.g. na)
        if number > nxyg:
            _data = np.array(_data).reshape((number // nxyg, nyg, nxg))
            if debug:
                print('Mesh data field {} with dimensions:  {:d} x {:d} x {:d}'.format(name, number // nxyg, nyg, nxg))
            return MESH_DATA, _data

        # 2D data field (e.g. ne)
        elif number == nxyg:
            _data = np.array(_data).reshape((nyg, nxg))
            if debug:
                print('Mesh data field {} with dimensions:  {:d} x {:d}'.format(name, nyg, nxg))
            return MESH_DATA, _data

        # Additional information field (e.g. zamin)
        else:
            _data = np.array(_data)
            if debug:
                print('Sim info field {} with length:     {} '.format(name, _data.shape[0]))
            return SIM_INFO_DATA, _data

    if not(os.path.isfile(filepath)):
        raise IOError('File {} not found.'.format(filepath))

    # Open SOLPS geometry file for reading
    fh = open(filepath, 'r')

    # Version header
    version = fh.readline()

    if header_dict is None:
        # Read mesh size
        fh.readline()
        line = fh.readline().split()
        nx = int(line[0])
        ny = int(line[1])

        # Calculate guard cells
        nxg = nx + 2
        nyg = ny + 2

        # Flat vector size
        nxy = nx * ny
        nxyg = nxg * nyg

        # Read Label
        fh.readline()
        label = fh.readline()

        # Save header
        header_dict = {'version': version, 'nx': nx, 'ny': ny, 'nxg': nxg, 'nyg': nyg, 'nxy': nxy, 'nxyg': nxyg, 'label': label}
    else:
        nx = header_dict['nx']
        ny = header_dict['ny']
        nxg = header_dict['nxg']
        nyg = header_dict['nyg']
        nxy = header_dict['nxy']
        nxyg = header_dict['nxyg']

    # variables for file data
    name = ''
    number = 0
    data = []
    sim_info_dict = {}
    mesh_data_dict = {}

    # Repeat reading data blocks till EOF
    while True:
        # Read line
        line = fh.readline().split()

        # EOF --OR--  New block of similar data (vector qty, e.g. bb)
        if len(line) == 0:
            # Check if last line
            line = fh.readline().split()
            if len(line) == 0:
                break

        # New block found
        if line[0] == '*cf:':

            # If previous block read --> Convert data to float, reshape and save to Object
            if name != '':
                flag, shaped_data = _make_solps_data_object(data)
                if flag == SIM_INFO_DATA:
                    sim_info_dict[name] = shaped_data
                elif flag == MESH_DATA:
                    mesh_data_dict[name] = shaped_data

            # Read next field paramters
            data_type = str(line[1].strip())
            number = int(line[2])
            name = str(line[3].strip())
            data = []

        # Append line to vector of data
        else:
            data.extend(line)

    if name != '':
        flag, shaped_data = _make_solps_data_object(data)
        if flag == SIM_INFO_DATA:
            sim_info_dict[name] = shaped_data
        elif flag == MESH_DATA:
            mesh_data_dict[name] = shaped_data

    return header_dict, sim_info_dict, mesh_data_dict


# Test case
if __name__ == '__main__':

    p = load_b2f_file("/home/mcarr/mst1/aug_2016/solps_testcase/b2fstate", debug=True)
