# MshIO
# License: Apache-2.0

if(TARGET mshio)
    return()
endif()

message(STATUS "Third-party: creating target 'mshio'")


include(FetchContent)
FetchContent_Declare(
    mshio
    GIT_REPOSITORY https://github.com/qnzhou/MshIO.git
    GIT_TAG 29d0263b45bbbb2931ecbe892d0d7f0f3a493d0c
    GIT_SHALLOW FALSE
)
FetchContent_MakeAvailable(mshio)
