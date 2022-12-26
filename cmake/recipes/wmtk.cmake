# WMTK
# License: MIT

if(TARGET wmtk::toolkit)
    return()
endif()

message(STATUS "Third-party: creating target 'wmtk::toolkit'")


include(FetchContent)
FetchContent_Declare(
    wildmeshing_toolkit
    GIT_REPOSITORY https://github.com/zfergus/wildmeshing-toolkit.git
    GIT_TAG 1d9109af7d3bce14c49c4dcc624c9975fef130c4
    GIT_SHALLOW FALSE
)
FetchContent_MakeAvailable(wildmeshing_toolkit)
