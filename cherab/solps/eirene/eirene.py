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


# Code based on script by Felix Reimold (2016)
class Eirene:

    def __init__(self, nx, ny, na, nm, ni, ns, species_labels, version=None,
                 da=None, ta=None, dm=None, tm=None, di=None, ti=None, rpa=None, rpm=None, ppa=None, ppm=None, rea=None,
                 rem=None, pea=None, pem=None, emist=None, emism=None, elosm=None, edism=None, eradt=None,
                 *args, **kwargs):
        """ Class for holding EIRENE neutral simulation data

        :param file_path:
        :param nx: Number of grid cells in the x direction
        :param ny: Number of grid cells in the y direction
        :param na: Number of atom species
        :param nm: Number of molecule species
        :param ni: Number of ion species
        :param ns: Total number of species in simulation
        :param species_labels: Text descriptions for each species in simulation.
        :param version: Version of EIRENE
        :param da: Atomic Neutral Density
        :param ta: Atomic Neutral Temperature
        :param dm: Molecular Neutral Density
        :param tm: Molecular Neutral Temperature
        :param di: Test Ion Density
        :param ti: Test Ion Temperature
        :param rpa: Atomic Radial Particle Flux
        :param rpm: Molecular Radial Particle Flux
        :param ppa: Atomic Poloidal Particle Flux
        :param ppm: Molecular Poloidal Particle Flux
        :param rea: Atomic Radial Energy Flux
        :param rem: Molecular Radial Energy Flux
        :param pea: Atomic Poloidal Energy Flux
        :param pem: Molecular Poloidal Energy Flux
        :param emist: Total Halpha Emission (including molecules)
        :param emism: Molecular Halpha Emission
        :param elosm: Power loss due to molecules (including dissociation)
        :param edism: Power loss due to molecule dissociation
        :param eradt: Neutral radiated power
        :param debug: Print out debugging information.
        """

        self._nx = nx
        self._ny = ny
        self._na = na
        self._nm = nm
        self._ni = ni
        self._ns = ns

        self._species_labels = species_labels
        self._version = version
        self._da = da
        self._ta = ta
        self._dm = dm
        self._tm = tm
        self._di = di
        self._ti = ti
        self._rpa = rpa
        self._rpm = rpm
        self._ppa = ppa
        self._ppm = ppm
        self._rea = rea
        self._rem = rem
        self._pea = pea
        self._pem = pem
        self._emist = emist
        self._emism = emism
        self._elosm = elosm
        self._edism = edism
        self._eradt = eradt

    @property
    def nx(self):
        """
        Number of grid cells in the poloidal direction

        :rtype: int
        """
        return self._nx

    @property
    def ny(self):
        """
        Number of grid cells in the radial direction

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

    @da.setter
    def da(self, value):
        self._check_dimensions(value, self._na)
        self._da = value

    @property
    def ta(self):
        """
        Atomic Neutral Temperature

        :rtype: np.ndarray
        """
        return self._ta

    @ta.setter
    def ta(self, value):
        self._check_dimensions(value, self._na)
        self._ta = value

    @property
    def dm(self):
        """
        Molecular Neutral Density

        :rtype: np.ndarray
        """
        return self._dm

    @dm.setter
    def dm(self, value):
        self._check_dimensions(value, self._nm)
        self._dm = value

    @property
    def tm(self):
        """
        Molecular Neutral Temperature

        :rtype: np.ndarray
        """
        return self._tm

    @tm.setter
    def tm(self, value):
        self._check_dimensions(value, self._nm)
        self._tm = value

    @property
    def di(self):
        """
        Test Ion Density

        :rtype: np.ndarray
        """
        return self._di

    @di.setter
    def di(self, value):
        self._check_dimensions(value, self._ni)
        self._di = value

    @property
    def ti(self):
        """
        Test Ion Temperature

        :rtype: np.ndarray
        """
        return self._ti

    @ti.setter
    def ti(self, value):
        self._check_dimensions(value, self._ni)
        self._ti = value

    @property
    def rpa(self):
        """
        Atomic Radial Particle Flux

        :rtype: np.ndarray
        """
        return self._rpa

    @rpa.setter
    def rpa(self, value):
        self._check_dimensions(value, self._na)
        self._rpa = value

    @property
    def rpm(self):
        """
        Molecular Radial Particle Flux

        :rtype: np.ndarray
        """
        return self._rpm

    @rpm.setter
    def rpm(self, value):
        self._check_dimensions(value, self._nm)
        self._rpm = value

    @property
    def ppa(self):
        """
        Atomic Poloidal Particle Flux

        :rtype: np.ndarray
        """
        return self._ppa

    @ppa.setter
    def ppa(self, value):
        self._check_dimensions(value, self._na)
        self._ppa = value

    @property
    def ppm(self):
        """
        Molecular Poloidal Particle Flux

        :rtype: np.ndarray
        """
        return self._ppm

    @ppm.setter
    def ppm(self, value):
        self._check_dimensions(value, self._nm)
        self._ppm = value

    @property
    def rea(self):
        """
        Atomic Radial Energy Flux

        :rtype: np.ndarray
        """
        return self._rea

    @rea.setter
    def rea(self, value):
        self._check_dimensions(value, self._na)
        self._rea = value

    @property
    def rem(self):
        """
        Molecular Radial Energy Flux

        :rtype: np.ndarray
        """
        return self._rem

    @rem.setter
    def rem(self, value):
        self._check_dimensions(value, self._nm)
        self._rem = value

    @property
    def pea(self):
        """
        Atomic Poloidal Energy Flux

        :rtype: np.ndarray
        """
        return self._pea

    @pea.setter
    def pea(self, value):
        self._check_dimensions(value, self._na)
        self._pea = value

    @property
    def pem(self):
        """
        Molecular Poloidal Energy Flux

        :rtype: np.ndarray
        """
        return self._pem

    @pem.setter
    def pem(self, value):
        self._check_dimensions(value, self._nm)
        self._pem = value

    @property
    def emist(self):
        """
        Total Halpha Emission (including molecules)

        :rtype: np.ndarray
        """
        return self._emist

    @emist.setter
    def emist(self, value):
        self._check_dimensions(value, 1)
        self._emist = value

    @property
    def emism(self):
        """
        Molecular Halpha Emission

        :rtype: np.ndarray
        """
        return self._emism

    @emism.setter
    def emism(self, value):
        self._check_dimensions(value, 1)
        self._emism = value

    @property
    def elosm(self):
        """
        Power loss due to molecules (including dissociation)

        :rtype: np.ndarray
        """
        return self._elosm

    @elosm.setter
    def elosm(self, value):
        self._check_dimensions(value, self._nm)
        self._elosm = value

    @property
    def edism(self):
        """
        Power loss due to molecule dissociation

        :rtype: np.ndarray
        """
        return self._edism

    @edism.setter
    def edism(self, value):
        self._check_dimensions(value, self._nm)
        self._edism = value

    @property
    def eradt(self):
        """
        Neutral radiated power

        :rtype: np.ndarray
        """
        return self._eradt

    @eradt.setter
    def eradt(self, value):
        self._check_dimensions(value, 1)
        self._eradt = value

    def _check_dimensions(self, data, dim0):
        """
        Checks compatibility of the data array dimension with the species number and grid size.
        :param dim0: size of the 1st dimenion
        :return:
        """

        if not data.shape == (dim0, self._ny, self._nx):
            raise ValueError("Array with shape {0} obtained, but {1} expected".format(data.shape,
                                                                                      (dim0, self._ny, self._nx)))
