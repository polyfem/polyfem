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
    GIT_TAG c800e75ce283e43918702fb7c8875a9911f48e1e
    GIT_SHALLOW FALSE
)
FetchContent_MakeAvailable(wildmeshing_toolkit)
