# PolySolve (https://github.com/MeshFEM/catamari2polysolve)
# License: MIT

if(TARGET polysolve)
    return()
endif()

message(STATUS "Third-party: creating target 'polysolve'")

include(CPM)
CPMAddPackage(
    NAME polysolve
    GIT_REPOSITORY https://github.com/MeshFEM/polysolve.git
    GIT_TAG integrate
)
