
# Copyright 2014-2017 United Kingdom Atomic Energy Authority
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


# Code based on script by Felix Reimold (2016)
class Eirene:
    
    def __init__(self, file_path, debug=False):
        """ Class for holding EIRENE neutral simulation data

        :param str file_path: path to EIRENE "fort.44" output file
        :param bool debug: status flag for printing debugging output
        """

        self._load_fort44_file(file_path, debug=debug)

    @property
    def nx(self):
        """
        Number of grid cells in the x direction

        :rtype: int
        """
        return self._nx

    @property
    def ny(self):
        """
        Number of grid cells in the y direction

        :rtype: int
        """
        return self._ny

    @property
    def version(self):
        """
        Version of EIRENE

        :rtype: str
        """
        return self._version

    @property
    def na(self):
        """
        Number of atom species

        :rtype: int
        """
        return self._na

    @property
    def nm(self):
        """
        Number of molecule species

        :rtype: int
        """
        return self._nm

    @property
    def ni(self):
        """
        Number of ion species

        :rtype: int
        """
        return self._ni

    @property
    def ns(self):
        """
        Total number of species in simulation

        :rtype: int
        """
        return self._ns

    @property
    def species_labels(self):
        """
        Text descriptions for each species in simulation.

        :rtype: list
        """
        return self._species_labels

    @property
    def da(self):
        """
        Atomic Neutral Density

        :rtype: np.ndarray
        """
        return self._da

    @property
    def ta(self):
        """
        Atomic Neutral Temperature

        :rtype: np.ndarray
        """
        return self._ta

    @property
    def dm(self):
        """
        Molecular Neutral Density

        :rtype: np.ndarray
        """
        return self._dm

    @property
    def tm(self):
        """
        Molecular Neutral Temperature

        :rtype: np.ndarray
        """
        return self._tm

    @property
    def di(self):
        """
        Test Ion Density

        :rtype: np.ndarray
        """
        return self._di

    @property
    def ti(self):
        """
        Test Ion Temperature

        :rtype: np.ndarray
        """
        return self._ti

    @property
    def rpa(self):
        """
        Atomic Radial Particle Flux

        :rtype: np.ndarray
        """
        return self._rpa

    @property
    def rpm(self):
        """
        Molecular Radial Particle Flux

        :rtype: np.ndarray
        """
        return self._rpm

    @property
    def ppa(self):
        """
        Atomic Poloidal Particle Flux

        :rtype: np.ndarray
        """
        return self._ppa

    @property
    def ppm(self):
        """
        Molecular Poloidal Particle Flux

        :rtype: np.ndarray
        """
        return self._ppm

    @property
    def rea(self):
        """
        Atomic Radial Energy Flux

        :rtype: np.ndarray
        """
        return self._rea

    @property
    def rem(self):
        """
        Molecular Radial Energy Flux

        :rtype: np.ndarray
        """
        return self._rem

    @property
    def pea(self):
        """
        Atomic Poloidal Energy Flux

        :rtype: np.ndarray
        """
        return self._pea

    @property
    def pem(self):
        """
        Molecular Poloidal Energy Flux

        :rtype: np.ndarray
        """
        return self._pem

    @property
    def pea(self):
        """
        Atomic Poloidal Energy Flux

        :rtype: np.ndarray
        """
        return self._pea

    @property
    def pem(self):
        """
        Molecular Poloidal Energy Flux

        :rtype: np.ndarray
        """
        return self._pem

    @property
    def emist(self):
        """
        Total Halpha Emission (including molecules)

        :rtype: np.ndarray
        """
        return self._emist

    @property
    def emism(self):
        """
        Molecular Halpha Emission

        :rtype: np.ndarray
        """
        return self._emism

    @property
    def elosm(self):
        """
        Power loss due to molecules (including dissociation)

        :rtype: np.ndarray
        """
        return self._elosm

    @property
    def edism(self):
        """
        Power loss due to molecule dissociation

        :rtype: np.ndarray
        """
        return self._edism

    @property
    def eradt(self):
        """
        Neutral radiated power

        :rtype: np.ndarray
        """
        return self._eradt

    @staticmethod
    def _read_block44(file_handle, ns, nx, ny, debug=False):
        """ Read standard block in EIRENE code output file 'fort.44'
    
        :param file_handle: A python core file handle object as a result of a
          call to open('./fort.44').
        :param int ns: total number of species
        :param int nx: number of grid x cells
        :param int ny: number of grid y cells
        :param bool debug: status flag for printing debugging output
        :return: ndarray of data with shape [nx, ny, ns]
        """
    
        itl = 5  # items per line

        array = np.zeros((nx, ny, ns))
    
        for iss in range(ns):
            for iy in range(ny):
                for ix in range(nx):
                    if ix % itl == 0:
                        line = file_handle.readline().split()
                    array[ix, iy, iss] = float(line[ix%itl])
            if debug:
                print(line)
    
        return array

    def _load_fort44_file(self, file_path, debug=False):
        """ Read neutral species and wall flux information from fort.44
    
        Template for reading is ngread.F in b2plot of B2.5 source
    
        :param str file_path: path to EIRENE "fort.44" output file
        :param bool debug: status flag for printing debugging output
        :rtype:
        """
    
        file_handle = open(file_path, 'r')
    
        # Read sizes
        line = file_handle.readline().split()
        self._nx = int(line[0])
        self._ny = int(line[1])
        self._version = int(line[2])
        if debug:
            print('Geometry & Version : nx {}, ny {}, version {}'
                  ''.format(self._nx, self._ny, self._version))
            
        # Read Species numbers
        line = file_handle.readline().split()
        self._na = int(line[0])  # number of atoms
        self._nm = int(line[1])  # number of molecules
        self._ni = int(line[2])  # number of ions
        self._ns = self._na + self._nm + self._ni  # total number of species
        if debug:
            print('Species # : {} atoms, {} molecules, {} ions, {} total species'
                  ''.format(self._na, self._nm, self._ni, self._ns))
    
        # Read Species labels
        self._species_labels = []
        for i in range(self._ns):
            line = file_handle.readline()
            self._species_labels.append(line.strip())
        if debug:
            print("Species labels => {}".format(self._species_labels))
    
        # Read atomic species (da, ta)
        self._da = self._read_block44(file_handle, self._na, self._nx, self._ny)  # Atomic Neutral Density
        self._ta = self._read_block44(file_handle, self._na, self._nx, self._ny)  # Atomic Neutral Temperature
        if debug:
            print('Atomic Neutral Density nD0: ', self._da[0, :, 0])
        if debug:
            print('Atomic Neutral Temperature TD0: ', self._ta[0, :, 0])
    
        # Read molecular species (dm, tm)
        self._dm = self._read_block44(file_handle, self._nm, self._nx, self._ny)  # Molecular Neutral Density
        self._tm = self._read_block44(file_handle, self._nm, self._nx, self._ny)  # Molecular Neutral Temperature
        
        # Read ion species (di, ti)
        self._di = self._read_block44(file_handle, self._ni, self._nx, self._ny)  # Test Ion Density
        self._ti = self._read_block44(file_handle, self._ni, self._nx, self._ny)  # Test Ion Temperature
    
        # Read radial particle flux (rpa, rpm)
        self._rpa = self._read_block44(file_handle, self._na, self._nx, self._ny)  # Atomic Radial Particle Flux
        self._rpm = self._read_block44(file_handle, self._nm, self._nx, self._ny)  # Molecular Radial Particle Flux
    
        # Read poloidal particle flux (ppa, ppm)
        self._ppa = self._read_block44(file_handle, self._na, self._nx, self._ny)  # Atomic Poloidal Particle Flux
        self._ppm = self._read_block44(file_handle, self._nm, self._nx, self._ny)  # Molecular Poloidal Particle Flux
    
        # Read radial energy flux (rea, rem)
        self._rea = self._read_block44(file_handle, self._na, self._nx, self._ny)  # Atomic Radial Energy Flux
        self._rem = self._read_block44(file_handle, self._nm, self._nx, self._ny)  # Molecular Radial Energy Flux
    
        # Read poloidal energy flux (pea, pem)
        self._pea = self._read_block44(file_handle, self._na, self._nx, self._ny)  # Atomic Poloidal Energy Flux
        self._pem = self._read_block44(file_handle, self._nm, self._nx, self._ny)  # Molecular Poloidal Energy Flux
    
        # Halpha total & molecules (emist, emism)
        self._emist = self._read_block44(file_handle, 1, self._nx, self._ny)  # Total Halpha Emission (including molecules)
        self._emism = self._read_block44(file_handle, 1, self._nx, self._ny)  # Molecular Halpha Emission
    
        # Radiated power (elosm, edism, eradt)
        self._elosm = self._read_block44(file_handle, 1, self._nx, self._ny)  # Power loss due to molecules (including dissociation)
        self._edism = self._read_block44(file_handle, 1, self._nx, self._ny)  # Power loss due to molecule dissociation
        self._eradt = self._read_block44(file_handle, 1, self._nx, self._ny)  # Neutral radiated power
    
        # These lines are from Felix's original routines but didn't appear to be used.
        # -------------
        # Define arrays
        #  dat.wldnek = data.data()(1,dat.nx,dat.ny)) #Halpha emission of neutrals (including molecules)
        #  dat.wldnep = data.data()(1,dat.nx,dat.ny)) #Halpha emission of molecules
        #  dat.wldna  = data.data()(dat.na,dat.nx,dat.ny)) #Power loss due to molecules (including dissociation)
        #  dat.wleda = data.data()(dat.na,dat.nx,dat.ny)) #Power loss due to moelcule dissociation
        #  dat.wldnm = data.data()(dat.nm,dat.nx,dat.ny)) #Neutral radiated power
        #  dat.wledm = data.data()(dat.nm,dat.nx,dat.ny)) #Neutral radiated power
        #  dat.wldra = data.data()(dat.nm,dat.nx,dat.ny)) #Neutral radiated power
        #  dat.wldrm = data.data()(dat.na,dat.nx,dat.ny)) #Neutral radiated power
        #  dat.wledm = data.data()(dat.nm,dat.nx,dat.ny)) #Neutral radiated power

