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
    GIT_TAG 7605bc6482eb06d2387a383f27dc662dffd3e5e0
    GIT_SHALLOW FALSE
)
FetchContent_MakeAvailable(wildmeshing_toolkit)
