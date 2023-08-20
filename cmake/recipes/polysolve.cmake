# PolySolve (https://github.com/polyfem/polysolve)
# License: MIT

if(TARGET polysolve)
    return()
endif()

message(STATUS "Third-party: creating target 'polysolve'")

# TODO: this requires a conflicting version of Eigen. Reenable when Eigen 3.4+ is available.
set(POLYSOLVE_WITH_ACCELERATE OFF CACHE BOOL "Enable Apple Accelerate" FORCE)

include(CPM)
CPMAddPackage("gh:polyfem/polysolve#91f7e3a4a44df4ca541cb46923d5593a49254efd")
