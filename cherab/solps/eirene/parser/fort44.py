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

from cherab.solps.eirene.parser.fort44_post_2016 import load_fort44_post_2016
from cherab.solps.eirene.parser.fort44_2013 import load_fort44_2013
from cherab.solps.eirene.parser.fort44_pre_2007 import load_fort44_pre_2007


def load_fort44_file(file_path, debug=False):
    """ Read neutral species and wall flux information from fort.44

    Template for reading is ngread.F in b2plot of B2.5 source.

    Specifications of data format are in Section 5.2 of the SOLPS
    manual (Running the coupled version -> Output Files).

    :param str file_path: path to EIRENE "fort.44" output file
    :param bool debug: status flag for printing debugging output
    :rtype:
    """
    with open(file_path, 'r') as file_handle:
        # dictionary with file data
        data = {}

        # Read sizes
        line = file_handle.readline().split()
        data["nx"] = int(line[0])
        data["ny"] = int(line[1])
        data["version"] = int(line[2])

    if debug:
        print('Geometry & Version : nx {}, ny {}, version {}'
              .format(data["nx"], data["ny"], data["version"]))

        # Look up file parsing function and call it to obtain for44 block and update data dictionary
    parser = assign_fort44_parser(data["version"])
    return parser(file_path, debug)


def assign_fort44_parser(file_version):
    """
    Looks up file parsing function for the parser Eirene file.
    :param file_version: Fort44 file version from the file header.
    :return: Parsing function object
    """

    # TODO write universal fort44 parser
    file_version = int(file_version)
    # Parser load_fort44_pre_2007 is used also for 20071209, 20080706 and 20081111
    # versions as a temporary solution. Parameter 'eneutrad' is not parsed from these file versions.
    if file_version <= 20081111:
        return load_fort44_pre_2007
    if file_version == 20130210:
        return load_fort44_2013
    if file_version >= 20160829:
        return load_fort44_post_2016
    raise ValueError("Can't read version {} fort.44 file".format(file_version))
