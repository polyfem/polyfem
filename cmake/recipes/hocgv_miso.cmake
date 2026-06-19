# hocgv-miso: miso-generated Jacobian validity solvers
# Replaces the deprecated bezier library.
# License: see hocgv-miso repo

if(TARGET miso)
    return()
endif()

message(STATUS "Third-party: creating target 'miso' via hocgv-miso")

add_subdirectory("${HOCGV_MISO_DIR}" "${CMAKE_BINARY_DIR}/hocgv-miso-build")
