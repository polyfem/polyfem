# PolySolve (https://github.com/polyfem/polysolve)
# License: MIT

if(TARGET polysolve)
    return()
endif()

message(STATUS "Third-party: creating target 'polysolve'")

include(CPM)

CPMAddPackage("gh:polyfem/polysolve#deb683d7c4264fbb5b1bae66f631dbe9bc88730e")
