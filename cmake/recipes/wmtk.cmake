# WMTK (https://github.com/wildmeshing/wildmeshing-toolkit)
# License: MIT

if(TARGET wmtk::toolkit)
    return()
endif()

message(STATUS "Third-party: creating target 'wmtk::toolkit'")

include(CPM)
CPMAddPackage("gh:zfergus/wildmeshing-toolkit#1ced6df948a29cd3f3028279c163ae582f44d37e")