# Jacobian Positivity Check (https://github.com/fsichetti/jacobian)
# License: MIT

if(TARGET jacobian)
    return()
endif()

message(STATUS "Third-party: creating target 'jacobian'")

option(PARAVIEW_OUTPUT "Export elements to Paraview" OFF)
option(IPRED_ARITHMETIC "Use the efficient Indirect Predicates library" ON)

include(CPM)
CPMAddPackage("gh:fsichetti/jacobian#e0716e28d0a86ce84d92bf923f1f7b71ab12fb16")
