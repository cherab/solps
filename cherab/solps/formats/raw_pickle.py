
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


import pickle
from cherab.solps import SOLPSMesh, SOLPSSimulation


def load_solps_from_pickle(filename):

    file_handle = open(filename, 'rb')
    state = pickle.load(file_handle)
    mesh = SOLPSMesh(state['mesh']['cr_r'], state['mesh']['cr_z'], state['mesh']['vol'])
    simulation = SOLPSSimulation(mesh, state['species_list'])
    simulation.__setstate__(state)
    file_handle.close()

    return simulation
