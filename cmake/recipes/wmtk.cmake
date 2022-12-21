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
    GIT_TAG 94d2947f0696afdeafa6909a4ac86a392a67f0ab
    GIT_SHALLOW FALSE
)
FetchContent_MakeAvailable(wildmeshing_toolkit)
