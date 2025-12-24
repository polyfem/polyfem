# Polyfem Solvers (https://github.com/polyfem/paraviewo)
# License: MIT

if(TARGET paraviewo::paraviewo)
    return()
endif()

message(STATUS "Third-party: creating target 'paraviewo::paraviewo'")

include(CPM)
CPMAddPackage("gh:polyfem/paraviewo#95379e6075b47394c2cc31c83512dd84b5f39d1c")
