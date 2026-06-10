# PolySolve (https://github.com/polyfem/polysolve)
# License: MIT

if(TARGET polysolve)
    return()
endif()

message(STATUS "Third-party: creating target 'polysolve'")

include(CPM)
CPMAddPackage("gh:iiiian/polysolve#mas-only")
