# PolySolve (https://github.com/polyfem/polysolve)
# License: MIT

if(TARGET polysolve)
    return()
endif()

message(STATUS "Third-party: creating target 'polysolve'")

# TODO: this requires a conflicting version of Eigen. Reenable when Eigen 3.4+ is available.
set(POLYSOLVE_WITH_ACCELERATE OFF CACHE BOOL "Enable Apple Accelerate" FORCE)

include(CPM)
CPMAddPackage("gh:polyfem/polysolve#1b142d3ae2a91522bea964b7f9f63753dec8739b")
