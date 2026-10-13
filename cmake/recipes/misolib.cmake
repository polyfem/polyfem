# misolib (https://gitlab.com/minimize-solve/misolib)
# License: none specified upstream

if(TARGET misolib::misolib)
    return()
endif()

message(STATUS "Third-party: creating target 'misolib::misolib'")

include(CPM)
CPMAddPackage("gl:minimize-solve/misolib#a6a12c679f7ab5b0beb5b01b7db5def70c52a9b8")
