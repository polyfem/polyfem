# SDL3::SDL3
# License: Zlib

if(TARGET SDL3::SDL3)
    return()
endif()

message(STATUS "Third-party: creating target 'SDL3::SDL3'")

include(CPM)
CPMAddPackage("gh:libsdl-org/SDL#release-3.2.16")
