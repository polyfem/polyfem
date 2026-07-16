# PolySolve (https://github.com/MeshFEM/catamari2polysolve)
# License: MIT

if(TARGET polysolve)
    return()
endif()

message(STATUS "Third-party: creating target 'polysolve'")

include(CPM)
CPMAddPackage("gh:MeshFEM/catamari2polysolve#integrate")
