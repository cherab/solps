project('cherab-solps', 'cython')

py3 = import('python').find_installation(pure: false)
py3_dep = py3.dependency()

# Numpy
incdir_numpy = meson.get_external_property('numpy-include-dir', 'not-given')
if incdir_numpy == 'not-given'
  # If not specified, try to query NumPy from the build python
  incdir_numpy = run_command(py3,
    [
      '-c',
      'import os; os.chdir(".."); import numpy; print(numpy.get_include())'
    ],
    check: true
  ).stdout().strip()
endif
inc_np = include_directories(incdir_numpy)

# Don't use deprecated Numpy API.
numpy_nodep_api = '-DNPY_NO_DEPRECATED_API=NPY_1_7_API_VERSION'
c_args = [numpy_nodep_api]

# Cython directives.
cython_args = ['-Xlanguage_level=3']
if get_option('buildtype') == 'debug'
    cython_args += ['-Xlinetrace=True']
    c_args += ['-DCYTHON_TRACE_NOGIL=1']
endif

subdir('cherab')
