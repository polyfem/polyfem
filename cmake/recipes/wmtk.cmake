# WMTK
# License: MIT

if(TARGET wmtk::toolkit)
    return()
endif()

message(STATUS "Third-party: creating target 'wmtk::toolkit'")


include(FetchContent)
FetchContent_Declare(
    wildmeshing_toolkit
    GIT_REPOSITORY https://github.com/wildmeshing/wildmeshing-toolkit.git
    GIT_TAG 3c2364a84613861d3314ae089abd40a54ef08929
    GIT_SHALLOW FALSE
)
FetchContent_MakeAvailable(wildmeshing_toolkit)
