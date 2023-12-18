python_sources = [
    '__init__.py',
    '__init__.pxd',
    'solps_2d_functions.pxd',
    'mesh_geometry.py',
    'solps_plasma.py'
]

py3.install_sources(
    python_sources,
    subdir: 'cherab/solps',
)

py3.extension_module(
    'solps_2d_functions',
    'solps_2d_functions.pyx',
    dependencies: py3_dep,
    include_directories: inc_np,
    cython_args: cython_args,
    c_args: c_args,
    install: true,
    subdir: 'cherab/solps',
)

subdir('b2')
subdir('eirene')
subdir('formats')
subdir('models')
subdir('tests')
