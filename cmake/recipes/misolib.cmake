# misolib (https://gitlab.com/minimize-solve/misolib)
# License: none specified upstream

if(TARGET misolib::misolib)
    return()
endif()

message(STATUS "Third-party: creating target 'misolib::misolib'")

include(CPM)
CPMAddPackage("gl:minimize-solve/misolib#f6dcd1af85ec55d3933364b4afc7fc317a9cf0c4")
