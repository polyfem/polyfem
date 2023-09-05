# Polyfem Solvers (https://github.com/polyfem/paraviewo)
# License: MIT

if(TARGET paraviewo::paraviewo)
    return()
endif()

message(STATUS "Third-party: creating target 'paraviewo::paraviewo'")

include(CPM)
CPMAddPackage("gh:Huangzizhou/paraviewo#bd590f71dd8db60e72b3aae7fb1ee9d2e14d96b3")