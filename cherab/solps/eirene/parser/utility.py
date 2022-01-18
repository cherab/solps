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


def read_unlabelled_block44(file_handle, ns, nx, ny):
    """ Read standard block in EIRENE code output file 'fort.44'

    This function supports blocks in fort.44 format versions which do
    not contain a header line. Versions up to 20130210 should work with
    this function.

    :param file_handle: A python core file handle object as a result of a
      call to open('./fort.44').
    :param int ns: total number of species
    :param int nx: number of grid poloidal cells
    :param int ny: number of grid radial cells
    :return: ndarray of data with shape [ns, ny, nx]
    """
    data = []
    npoints = ns * nx * ny
    while len(data) < npoints:
        line = file_handle.readline().split()
        data.extend(line)
    data = np.asarray(data, dtype=float).reshape((ns, ny, nx))
    return data


def read_labelled_block44(file_handle):
    """ Read standard block in EIRENE code output file 'fort.44'

    This function supports blocks in fort.44 format versions which have
    a header line beginning with `*eirene` and containing the name of
    the variable and the number of items. Versions later than 2016
    should work with this function.

    :param file_handle: A python core file handle object as a result of a
      call to open('./fort.44').
    :return: a dictionary {name: data} where name is a string and data is
      a 1D array of the data.
    """
    header = file_handle.readline()
    if not header:
        raise EOFError()
    # Treat blank lines between blocks as null data.
    if header == "\n":
        return {}
    # Remove whitespace in indexed labels.
    pattern = re.compile(r"\(\s+(\d+)\)")
    header = pattern.sub(r"(\1)", header)
    header = header.split()
    # Deal with unlabelled printing of nlim, nsts, nstra.
    if len(header) == 3:
        return {'extra_dims': np.asarray(header, dtype=int)}
    if " ".join(header[:3]) != "*eirene data field":
        raise ValueError("fort.44 block format is not supported.")
    name = header[3]
    npoints = int(header[6])
    data = []
    while len(data) < npoints:
        file_pos = file_handle.tell()
        line = file_handle.readline().split()
        # A bug(?) in 20170328 means strata-resolved quantities are one value short.
        if line[0] == "*eirene":
            data.extend("0")
            file_handle.seek(file_pos)
            break
        # Some variables include species labels on the first line instead of numbers.
        try:
            float(line[0])
        except ValueError:
            continue
        data.extend(line)
    data = np.asarray(data, dtype=float)
    return {name: data}
