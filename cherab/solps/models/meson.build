python_sources = [
    '__init__.py',
    '__init__.pxd',
    'line_emitter.pxd',
    'solps_emitter.py',
]

py3.install_sources(
    python_sources,
    subdir: 'cherab/solps/models',
)

# FIXME: this currently fails because line_emitter.pyx tries to cimport
# solps_2d_functions.pxd, and this is not present in the out-of-tree build
# that meson does.
# Scipy works around this by creating a generator object which has a dependency
# that copies the necessary files into the build location, and then uses that
# generator to build extensions which depend on the pxd files. See for example:
# https://github.com/scipy/scipy/blob/be11e296ba707157a1727dba8d67affc8f060a97/scipy/meson.build#L339
# https://github.com/scipy/scipy/blob/be11e296ba707157a1727dba8d67affc8f060a97/scipy/signal/meson.build#L60
# Unfortunately this doesn't scale to a project like Cherab with lots of
# deeply-nested submodules each defining a pxd file.
py3.extension_module(
    'line_emitter',
    'line_emitter.pyx',
    dependencies: py3_dep,
    include_directories: inc_np,
    install: true,
    subdir: 'cherab/solps/models',
)
