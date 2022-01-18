Project Changelog
=================

Release 1.2.0 (31 Dec 2021)
----------------------------

* Added support for Raysect 0.7.1.
* Replaced unsafe SOLPSFunction3D and SOLPSVectorFunction3D with safe SOLPSFunction2D and SOLPSVectorFunction2D (use AxisymmetricMapper(SOLPSFunction2D) and VectorAxisymmetricMapper(SOLPSVectorFunction2D) for 3D).
* Added correct initialisation of properties in SOLPSMesh and SOLPSSimulation.
* Added new attributes to SOLPSMesh for basis vectors, cell connection areas, indices of neighbouring cells and new methods to_poloidal() and to_cartesian() for converting vectors defined on a grid from/to (poloidal, radial)/(R, Z).
* Fixed incorrect calculation of cell basis vectors.
* Inverted the indexing of data arrays and made all arrays row-major.
* Added electron_velocities and neutral_listproperties to SOLPSSimulation.
* Added 2D and 3D interpolators for plasma parameters to SOLPSSimulation.
* Added parsing of additional quantities in load_solps_from_mdsplus(), load_solps_from_raw_output() and load_solps_from_balance().
* Added support for B2 stand-alone simulations.
* Added support for arbitrary plasma chemical composition.
* Fixed incorrect calculation of velocities.
* Add demos using the Generomak example machine, with no external data dependencies.
* Add support for reading more versions of EIRENE fort.44 output files.
* Fix loading of total radiated power from raw files.
* Add H-alpha emissivity calculated with EIRENE to SOLPSSimulation.
* Replace `solps_total_radiated_power` with `make_solps_emitter` and remove `SOLPSTotalRadiatedPower`.
* Fix broken serialisation with pickle.
* Allow the user to provide individual paths to SOLPS raw output files.
* Allow user-provided AtomicData when loading simulations.
* Other small fixes and improvements to code and documentation.

Release 1.1.0 (30 July 2020)
----------------------------

* Fix mesh loading when SOLPSPlasma is unpickled.
* Support Raysect 0.6 and Cherab-core 1.2.
* Add support for multiple versions of Eirene fort.44 files.
* Add support for loading a simulation from a balance.nc file.
* Fix MDSplus neutral radiation and density loading.
* Add boron to the species list when creating a plasma from a SOLPSSimulation.

Release 1.0.0 (13 Sept 2017)
----------------------------

Initial public release.
