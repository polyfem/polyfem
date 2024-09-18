# Robust Bezier subdivision (https://gitlab.com/fsichetti/robust-bezier-subdivision)
# License: MIT

if(TARGET bezier)
    return()
endif()

message(STATUS "Third-party: creating target 'bezier'")

option(PARAVIEW_OUTPUT "Export elements to Paraview" OFF)
option(HDF5_INTERFACE "Process HDF5 datasets" ON)
option(EIGEN_INTERFACE "Read data from Eigen classes" ON)
option(IPRED_ARITHMETIC "Use the efficient Indirect Predicates library" ON)
if (IPRED_ARITHMETIC)
    add_compile_definitions(IPRED_ARITHMETIC)
endif()
if (EIGEN_INTERFACE)
    add_compile_definitions(EIGEN_INTERFACE)
endif()

include(CPM)
CPMAddPackage("https://gitlab.com/fsichetti/robust-bezier-subdivision.git#495f74c2bcef3f69e55c08c543a11a0db349c166")