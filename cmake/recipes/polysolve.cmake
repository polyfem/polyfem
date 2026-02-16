# PolySolve (https://github.com/polyfem/polysolve)
# License: MIT

if(TARGET polysolve)
    return()
endif()

message(STATUS "Third-party: creating target 'polysolve'")

include(CPM)
CPMAddPackage("gh:maxpaik16/polysolve#7e16454a7147ceadd35b9848a99e4e616c9e4df0")
