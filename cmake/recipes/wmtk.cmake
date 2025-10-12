# WMTK (https://github.com/wildmeshing/wildmeshing-toolkit)
# License: MIT

if(TARGET wmtk::toolkit)
    return()
endif()

message(STATUS "Third-party: creating target 'wmtk::toolkit'")

include(CPM)
CPMAddPackage("gh:polyfem/wildmeshing-toolkit#e6202e326c51e83168c536663efbed69f57501a3")
