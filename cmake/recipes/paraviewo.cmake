# Polyfem Solvers (https://github.com/polyfem/paraviewo)
# License: MIT

if(TARGET paraviewo::paraviewo)
    return()
endif()

message(STATUS "Third-party: creating target 'paraviewo::paraviewo'")

include(CPM)
CPMAddPackage("gh:polyfem/paraviewo#61685ee7f924873d1fd13068b6f0b691b295992d")