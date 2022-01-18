# Cherab SOLPS

Cherab add-on module for SOLPS simulations.

This module enables the creation of Cherab plasma objects from SOLPS simulations.
Several SOLPS output formats are supported.
Please see the examples in the [demos](demos) directory for an illustration of how to use the module.

## Installation

It is recommended to install Cherab in a [virtual environment](https://docs.python.org/3/tutorial/venv.html).
This will enable installation of packages without modifying the system Python installation, which is particularly important on shared systems.
To create a virtual environment, do the following:

```bash
python3 -m venv ~/venvs/cherab-venv
```

After the virtual environment is created, it can be activated by running:

```bash
source ~/venvs/cherab-venv/bin/activate
```

Once activated, cherab-solps and its dependencies can be installed with:

```bash
pip install cherab-solps
```

## Building from source

### Users

This module depends on the core Cherab framework.
Cherab core, and all of its dependencies, are available on PyPI and can be installed using `pip`.

Note also that a [bug](https://github.com/cython/cython/issues/2918) in Cython prevents Cherab submodules from installing correctly.
This bug is fixed, but not yet released in the stable version of Cython.
As a result, you will need to install a pre-release version of Cython before installing this package.
Recent versions of `pip` which support [PEP 518](https://www.python.org/dev/peps/pep-0518/) will handle this automatically when you run `pip install cherab-solps` and a binary wheel is not available for your platform or Python version.

For older versions of `pip` you may need to manually install the build-time dependencies.
First, clone this repository, then do:

```bash
pip install -U cython==3.0a5
pip install cherab
pip install <path-to-cherab-solps>
```

This will pull in `cherab-core`, `raysect` `numpy` and other dependencies, then build and install the SOLPS module.

### Developers

For developing the code, it is recommended to use local checkouts of `cherab-core` and `raysect`, as well as `cherab-solps`.
Development should be done against the `development` branch of this repository, and any modifications submitted as pull requests to be merged back into `development`.

To install the package in develop mode, so that local changes are immediately visible without needing to reinstall, install with:

```
pip install -e <path-to-cherab-solps>
```

If you are modifying Cython files you will need to run `./dev/build.sh` from this directory in order to rebuild the extension modules.
They will then be used when Python is restarted.

