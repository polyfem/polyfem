# Jacobian Positivity Check (https://github.com/fsichetti/jacobian)
# License: MIT

if(TARGET jacobian)
    return()
endif()

message(STATUS "Third-party: creating target 'jacobian'")

option(PARAVIEW_OUTPUT "Export elements to Paraview" OFF)

include(CPM)
CPMAddPackage("gh:fsichetti/jacobian#a33f3eb924974fe93b55f5fab3b160fa44f33681")
