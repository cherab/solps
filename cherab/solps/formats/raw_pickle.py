
import pickle
from cherab.solps import SOLPSMesh, SOLPSSimulation


def load_solps_from_pickle(filename):

    file_handle = open(filename, 'rb')
    state = pickle.load(file_handle)
    mesh = SOLPSMesh(state['mesh']['cr_r'], state['mesh']['cr_z'], state['mesh']['vol'])
    simulation = SOLPSSimulation(mesh)
    simulation.__setstate__(state)
    file_handle.close()

    return simulation
