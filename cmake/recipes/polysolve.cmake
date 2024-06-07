# PolySolve (https://github.com/polyfem/polysolve)
# License: MIT

if(TARGET polysolve)
    return()
endif()

message(STATUS "Third-party: creating target 'polysolve'")

include(CPM)
CPMAddPackage("gh:Huangzizhou/polysolve#e7e41add311f86d91b313c1fef88a09e3c8bf278")
