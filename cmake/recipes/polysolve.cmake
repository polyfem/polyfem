# PolySolve (https://github.com/polyfem/polysolve)
# License: MIT

if(TARGET polysolve)
    return()
endif()

message(STATUS "Third-party: creating target 'polysolve'")

include(CPM)
CPMAddPackage("gh:Huangzizhou/polysolve#4438ac2478ee8a3ff59bc841176c499bd9ad7621")
