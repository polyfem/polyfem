# Jacobian Positivity Check (https://github.com/fsichetti/jacobian)
# License: MIT

if(TARGET jacobian)
    return()
endif()

message(STATUS "Third-party: creating target 'jacobian'")

option(PARAVIEW_OUTPUT "Export elements to Paraview" OFF)
option(HDF5_INTERFACE "Process HDF5 datasets" OFF)
option(IPRED_ARITHMETIC "Use the efficient Indirect Predicates library" ON)
if (IPRED_ARITHMETIC)
    add_compile_definitions(IPRED_ARITHMETIC)
endif()

include(CPM)
CPMAddPackage("gh:fsichetti/jacobian#db13092c42437c0238c38dadb3826c2c5f7845bc")
