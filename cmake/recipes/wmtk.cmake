# WMTK (https://github.com/wildmeshing/wildmeshing-toolkit)
# License: MIT

if(TARGET wmtk::toolkit)
    return()
endif()

message(STATUS "Third-party: creating target 'wmtk::toolkit'")


include(CPM)
CPMAddPackage("gh:wildmeshing/wildmeshing-toolkit#3c2364a84613861d3314ae089abd40a54ef08929")