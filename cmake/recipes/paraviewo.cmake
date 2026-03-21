# Polyfem Solvers (https://github.com/polyfem/paraviewo)
# License: MIT

if(TARGET paraviewo::paraviewo)
    return()
endif()

message(STATUS "Third-party: creating target 'paraviewo::paraviewo'")

include(CPM)
CPMAddPackage("gh:polyfem/paraviewo#d402e9f623f005f8c224ebe3273261fc3f409209")
