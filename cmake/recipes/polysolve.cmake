# PolySolve (https://github.com/polyfem/polysolve)
# License: MIT

if(TARGET polysolve)
    return()
endif()

message(STATUS "Third-party: creating target 'polysolve'")

include(CPM)
CPMAddPackage("gh:polyfem/polysolve#9f8a5e18d064a38f12a9abf5bcf1ff87a7432219")
