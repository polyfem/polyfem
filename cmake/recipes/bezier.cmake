# Robust Bezier subdivision (git@github.com:fsichetti/bezier)
# License: MIT

if(TARGET bezier)
    return()
endif()

message(STATUS "Third-party: creating target 'bezier'")

option(HDF5_INTERFACE "Process HDF5 datasets" OFF)
option(EIGEN_INTERFACE "Read data from Eigen classes" ON)
option(IPRED_ARITHMETIC "Use the efficient Indirect Predicates library" ON)
option(GMP_INTERFACE "Rational numbers" OFF)

if (IPRED_ARITHMETIC)
    add_compile_definitions(IPRED_ARITHMETIC)
endif()
if (EIGEN_INTERFACE)
    add_compile_definitions(EIGEN_INTERFACE)
endif()

include(CPM)
CPMAddPackage("gh:Huangzizhou/bezier#cdc1e4e2ae9109382b21c515c0865e1ba02646ed")
