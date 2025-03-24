cmake_path(RELATIVE_PATH CMAKE_CURRENT_BINARY_DIR BASE_DIRECTORY ${CMAKE_BINARY_DIR} OUTPUT_VARIABLE libdir)

cython_transpile(solps_2d_functions.pyx LANGUAGE C OUTPUT_VARIABLE solps_2d_functions_c)
python_add_library(solps_2d_functions MODULE "${solps_2d_functions_c}" WITH_SOABI)
target_link_libraries(solps_2d_functions PRIVATE Python::NumPy)

install(TARGETS solps_2d_functions DESTINATION ${libdir})

add_subdirectory(models)
