cmake_path(RELATIVE_PATH CMAKE_CURRENT_BINARY_DIR BASE_DIRECTORY ${CMAKE_BINARY_DIR} OUTPUT_VARIABLE libdir)

cython_transpile(line_emitter.pyx LANGUAGE C OUTPUT_VARIABLE line_emitter_c)
python_add_library(line_emitter MODULE "${line_emitter_c}" WITH_SOABI)
target_link_libraries(line_emitter PRIVATE Python::NumPy)

install(TARGETS line_emitter DESTINATION ${libdir})
