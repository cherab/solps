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

Recent versions of `pip` which support [PEP 518](https://www.python.org/dev/peps/pep-0518/) will handle this automatically when you run `pip install cherab-solps` and a binary wheel is not available for your platform or Python version.

For older versions of `pip` you may need to manually install the build-time dependencies.
First, clone this repository, then do:

```bash
pip install -r <path-to-cherab-solps>/requirements.txt
pip install <path-to-cherab-solps>
```

This will pull in `cherab-core`, `raysect` `numpy` and other dependencies, then build and install the SOLPS module.

### Developers

Development should be done against the `development` branch of this repository, and any modifications submitted as pull requests to be merged back into `development`.

To install the package in editable mode, so that local changes are immediately visible without needing to reinstall, install the project dependencies into your development environment.
You should also enable auto rebuilds.
From the cherab-solps directory, run:

```
pip install -r requirements.txt
pip install --no-build-isolation --config-settings=editable.rebuild=true -e .
```

If you are modifying Cython files these will then be automatically rebuilt and the modified versions used when Python is restarted.

Pure Python files will be automatically included in the distribution as long as they have been added to Git, but if you add any Cython files you will need to add (or modify) a CMakeLists.txt file in the same directory as the new files to ensure these modules are built and included in the distribution.
See the existing CMakeLists.txt files for examples of how to do this.
Also note that when adding new Cython files you will need to re-run the above `pip install` command to ensure these new modules will be available in the editable install.

#### Profiling

It is possible to turn on profiling and line tracing support in the Cython extensions, which may be useful for performance optimisation.
These features do incur a performance overhead so they are disabled by default.

To enable function-level profiling after installing the project in editable mode (see above), reinstall it with the following `pip` command:

```
pip install --no-build-isolation --config-settings=editable.rebuild=true --config-settings=cmake.define.profile=ON -e <path-to-cherab-solps>
```

To enable line-by-line profiling, use:

```
pip install --no-build-isolation --config-settings=editable.rebuild=true --config-settings=cmake.define.line-profile=ON -e <path-to-cherab-solps>
```

**Important:** the profile and line-profile settings will persist across subsequent (manual or automatic) rebuilds until they are turned off by running `pip install` with the corresponding definition set to `OFF`.
