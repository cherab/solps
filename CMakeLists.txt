cmake_minimum_required(VERSION 3.21)
project(${SKBUILD_PROJECT_NAME} LANGUAGES C)

find_package(
  Python
  COMPONENTS Interpreter Development.Module NumPy
  REQUIRED)

# Cython support uses cython-cmake 0.2.0:
# https://pypi.org/project/cython-cmake/.
# We use a vendored copy as upstream doesn't support Python 3.7, but this
# isn't a problem as the vendored files are cmake not python files.
# Ironically the vendoring scripts are the only part of the project which
# would potentially cause Python incompatibilities...
list(APPEND CMAKE_MODULE_PATH "${CMAKE_SOURCE_DIR}/cmake")
find_package(Cython MODULE REQUIRED VERSION 3.0)
include(UseCython)

# Out-of-tree build means Cython needs to be told to look in
# the source tree for pxd files.
set(CYTHON_ARGS -3 -Wextra -I${CMAKE_SOURCE_DIR})

# Don't use deprecated Numpy API.
add_compile_definitions(NPY_NO_DEPRECATED_API=NPY_1_7_API_VERSION)

# Demo files are stored separately and must be copied manually.
file(GLOB_RECURSE demos RELATIVE ${CMAKE_SOURCE_DIR} "${CMAKE_SOURCE_DIR}/demos/*")
foreach(demo_file ${demos})
  get_filename_component(dir ${demo_file} DIRECTORY)
  install(FILES ${demo_file} DESTINATION ${SKBUILD_DATA_DIR}/share/cherab/demos/solps/${dir})
endforeach()

# Dynamic options ported over from original setup.py.
# CMake's caching means when we use option(<name>) here along with
# -Ccmake.define.<name>=<value> (pypa-build)
# or --config-settings=cmake.define.<name>=<value> (pip),
# these persist across rebuilds until specified again on the command line with
# different values.
#
# This means --force won't work properly, since specifying --force for two builds in
# succession would be treated as the same build (the cache value doesn't change) and
# would not lead to calling cython at all. For other options such as --profile,
# again they will only be turned on or off if the command line arguments are changed.

option(profile "Enable profiling in Cython modules." OFF)
if(profile)
  list(APPEND CYTHON_ARGS -Xprofile=True)
endif()

option(line-profile "Enable line-by-line profiling in Cython modules." OFF)
if(line-profile)
  list(APPEND CYTHON_ARGS -Xlinetrace=True)
  add_compile_definitions(CYTHON_TRACE=1)
  add_compile_definitions(CYTHON_TRACE_NOGIL=1)
endif()

# Cython extensions are all stored in cherab/solps and subdirectories.  Note: all
# modifications to CYTHON_ARGS should be done before the call to add_subdirectory to
# ensure subdirectories pick up the final list of arguments.
add_subdirectory(cherab/solps)
