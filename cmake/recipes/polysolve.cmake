# PolySolve (https://github.com/polyfem/polysolve)
# License: MIT

if(TARGET polysolve)
    return()
endif()

message(STATUS "Third-party: creating target 'polysolve'")

# TODO: this requires a conflicting version of Eigen. Reenable when Eigen 3.4+ is available.
set(POLYSOLVE_WITH_ACCELERATE OFF CACHE BOOL "Enable Apple Accelerate" FORCE)

include(CPM)
# CPMAddPackage("gh:polyfem/polysolve#e91bd511e583a3aff9a05e7ce51296f28bf3f324")
CPMAddPackage("gh:arvigj/polysolve#ab7091d20ae52795117c959958506a38305921fe")
