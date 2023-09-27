# WMTK (https://github.com/wildmeshing/wildmeshing-toolkit)
# License: MIT

if(TARGET wmtk::toolkit)
    return()
endif()

message(STATUS "Third-party: creating target 'wmtk::toolkit'")

include(CPM)
CPMAddPackage("gh:zfergus/wildmeshing-toolkit#7605bc6482eb06d2387a383f27dc662dffd3e5e0")